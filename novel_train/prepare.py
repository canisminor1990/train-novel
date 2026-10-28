"""Turn raw novels into premise → chapter SFT samples and cleaned CPT books."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from novel_train.envload import ROOT, configure_stdio
from novel_train.prompts import build_user_input, instruction_for
from novel_train.textutil import (
    clean_document,
    count_chars,
    iter_passages,
    load_document,
    self_check,
    tail_text,
)

SUFFIXES = {".txt", ".text", ".md", ".epub"}
MIN_PASSAGE_CHARS = 200
# 第一遍按约 6000 字切，短于 4000 字的章节先合并。模型一次能写多长，取决于训练回答有多长，
# 所以两遍都停在 4000 字以上。第二遍按约 4300 字换一组切点，篇幅有差别，模型才听「字数」这个要求。
SECOND_PASS_BELOW = 300
SECOND_PASS_MAX = 4300
SECOND_PASS_MIN = 4000


def passage_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def premises_path(output_dir: Path) -> Path:
    """Premises outlive re-slicing: they sit next to processed/, keyed by passage text."""
    return output_dir.parent / "premises.jsonl"


def load_premises(path: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    if not path.exists():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("id") and str(row.get("premise") or "").strip():
            found[str(row["id"])] = str(row["premise"]).strip()
    return found


def is_sample(path: Path) -> bool:
    stem = path.stem.lower()
    return path.stem.startswith("示例") or stem.startswith("example") or path.stem.startswith("_")


def list_books(folder: Path) -> list[Path]:
    files = [
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUFFIXES and not path.name.startswith(".")
    ]
    return sorted(files)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def eval_chars_for(total: int, args: argparse.Namespace) -> int:
    """Hold out the end of the book. Too short a book keeps everything for training."""
    if total >= 20 * args.passage_max:
        return max(args.passage_max, int(total * args.eval_ratio))
    if total >= 8 * args.passage_max:
        return args.passage_max
    return 0


def split_tail(chapters: list[tuple[str, list[str]]], eval_chars: int):
    """Split off the last eval_chars as a held-out region that no training passage overlaps."""
    lengths = [count_chars(paragraph) for _, paragraphs in chapters for paragraph in paragraphs]
    cut = len(lengths)
    acc = 0
    while eval_chars > 0 and cut > 1 and acc < eval_chars:
        cut -= 1
        acc += lengths[cut]
    train: list[tuple[str, list[str]]] = []
    held: list[tuple[str, list[str]]] = []
    offset = 0
    for title, paragraphs in chapters:
        end = offset + len(paragraphs)
        if end <= cut:
            train.append((title, paragraphs))
        elif offset >= cut:
            held.append((title, paragraphs))
        else:
            train.append((title, paragraphs[: cut - offset]))
            held.append((title, paragraphs[cut - offset :]))
        offset = end
    return train, held, cut


def make_sample(passage: dict, flat: list[str], book: str, index: int, premises: dict, args) -> dict | None:
    output = "\n".join(flat[passage["start"] : passage["end"]]).strip()
    size = count_chars(output)
    if size < MIN_PASSAGE_CHARS:
        return None
    pid = passage_id(output)
    premise = premises.get(pid, "")
    title = passage["title"] if passage["title"] != "正文" else ""
    chapter = title
    if title and passage["parts"] > 1:
        chapter = f"{title}（第 {passage['part']}/{passage['parts']} 部分）"
    # 一半样本带前情，一半只给命题，两种写法都学到。
    # 还没有命题、又不是一章开头的段落，没有前情就无从下笔，所以一定带上。
    starts_chapter = bool(title) and passage["part"] == 1
    wants_context = index % 2 == 1 or (not premise and not starts_chapter)
    context = ""
    if passage["start"] > 0 and wants_context:
        context = tail_text("\n".join(flat[max(0, passage["start"] - 40) : passage["start"]]), args.context_chars)
    return {
        "instruction": instruction_for(index),
        "input": build_user_input(premise, chapter, size, context),
        "output": output,
        "book": book,
        "chapter": chapter,
        "id": pid,
        "has_premise": bool(premise),
    }


def build(args: argparse.Namespace) -> int:
    source = Path(args.input)
    if not source.is_absolute():
        source = ROOT / source
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    if not source.exists():
        print(f"找不到输入目录：{source}")
        return 2

    files = list_books(source)
    samples = [path for path in files if is_sample(path)]
    real = [path for path in files if not is_sample(path)]
    chosen = real or samples
    if not chosen:
        print(f"{source} 里没有 txt / epub。把小说放进去后再运行。")
        return 2
    if real and samples:
        print(f"已跳过 {len(samples)} 个示例文件。正式训练不会混入示例。")

    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    books_rows: list[dict] = []
    passage_rows: list[dict] = []
    seen_outputs: set[str] = set()
    stats_books = []
    premises = load_premises(premises_path(output_dir))
    with_premise = 0

    for path in chosen:
        try:
            raw = load_document(path)
        except Exception as exc:
            print(f"跳过 {path.name}：{exc}")
            continue
        chapters, report = clean_document(raw)
        junk = report["junk_lines"]
        paragraphs_all = [paragraph for _, paragraphs in chapters for paragraph in paragraphs]
        char_count = sum(count_chars(paragraph) for paragraph in paragraphs_all)
        if char_count < args.min_book_chars:
            print(f"跳过 {path.name}：清洗后只有 {char_count} 字，少于 {args.min_book_chars}。")
            continue

        book_name = path.stem
        book_text = "\n\n".join(
            (f"{title}\n" if title and title != "正文" else "") + "\n".join(paragraphs)
            for title, paragraphs in chapters
        )
        books_rows.append({"book": book_name, "text": book_text, "chars": char_count})

        flat = [paragraph for _, paragraphs in chapters for paragraph in paragraphs]
        breaks = frozenset(report.get("scene_breaks") or [])
        train_chapters, held_chapters, cut = split_tail(chapters, eval_chars_for(char_count, args))

        passes = [list(iter_passages(train_chapters, args.passage_max, args.passage_min, breaks))]
        if len(passes[0]) < SECOND_PASS_BELOW:
            passes.append(list(iter_passages(train_chapters, SECOND_PASS_MAX, SECOND_PASS_MIN, breaks)))
        held = list(iter_passages(held_chapters, args.passage_max, args.passage_min, breaks, offset=cut))

        book_train: list[dict] = []
        book_eval: list[dict] = []
        planned = [(book_train, passage) for chunk in passes for passage in chunk]
        planned += [(book_eval, passage) for passage in held]
        for target, passage in planned:
            if target is book_train and args.max_samples_per_book and len(book_train) >= args.max_samples_per_book:
                continue
            row = make_sample(passage, flat, book_name, len(book_train) + len(book_eval), premises, args)
            if row is None or row["output"] in seen_outputs:
                continue
            seen_outputs.add(row["output"])
            with_premise += row.pop("has_premise")
            passage_rows.append({"id": row["id"], "book": book_name, "chapter": row["chapter"], "text": row["output"]})
            target.append(row)
        train_rows.extend(book_train)
        eval_rows.extend(book_eval)
        titled = sum(1 for title, _ in chapters if title != "正文")
        stats_books.append(
            {
                "book": book_name,
                "chars": char_count,
                "chapters": titled,
                "scene_breaks": len(breaks),
                "raw_chars": count_chars(raw),
                "junk_lines": junk,
                "duplicate_lines": report["duplicate_lines"],
                "merged_lines": report["merged_lines"],
                "duplicate_chapters": report["duplicate_chapters"],
                "train": len(book_train),
                "eval": len(book_eval),
            }
        )
        layout = f"{titled} 章" if titled else f"不分章，{len(breaks)} 处分隔线"
        print(
            f"{book_name}  {char_count} 字  {layout}  "
            f"训练 {len(book_train)}  验证 {len(book_eval)}  去掉杂行 {junk}"
        )

    if not train_rows:
        print("没有切出续写样本。文本可能太短，或章节被广告过滤清掉了。")
        return 2

    rng = random.Random(args.seed)
    rng.shuffle(train_rows)

    sft_dir = output_dir / "sft"
    cpt_dir = output_dir / "cpt"
    write_jsonl(sft_dir / "train.jsonl", train_rows)
    write_jsonl(sft_dir / "eval.jsonl", eval_rows)
    write_jsonl(cpt_dir / "books.jsonl", books_rows)
    write_jsonl(output_dir / "passages.jsonl", passage_rows)

    stats = {
        "books": stats_books,
        "train_samples": len(train_rows),
        "eval_samples": len(eval_rows),
        "passages": len(passage_rows),
        "with_premise": with_premise,
        "passage_max": args.passage_max,
        "passage_min": args.passage_min,
        "context_chars": args.context_chars,
    }
    stats_path = output_dir / "stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("")
    print(f"SFT 训练 {len(train_rows)} 条，验证 {len(eval_rows)} 条，其中 {with_premise} 条有命题")
    print(f"CPT 书稿 {len(books_rows)} 本")
    print(f"写出 {sft_dir / 'train.jsonl'}")
    if args.preview > 0:
        print("\n----- 样本预览 -----")
        for row in train_rows[: args.preview]:
            print(f"[{row['book']}] {row['instruction']}")
            print(row["input"][:240].replace("\n", " / "))
            print("=> " + row["output"][:180].replace("\n", " / "))
            print("")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把小说原文切成「命题 → 一章正文」训练集，不需要人工打标。")
    parser.add_argument("--self-check", action="store_true", help="用一段合成文本检查清洗和切分")
    parser.add_argument("--project", default=None, help="项目 ID，读 projects/<ID>/raw，写 projects/<ID>/processed")
    parser.add_argument("--input", default="data/raw")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--passage-max", type=int, default=6000, help="一条样本大约写多少字。更长的正文会拆开")
    parser.add_argument("--passage-min", type=int, default=4000, help="短于这个字数的章节和后面的章节合并")
    parser.add_argument("--context-chars", type=int, default=600, help="带前情的样本取多少字前情")
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--min-book-chars", type=int, default=800)
    parser.add_argument("--max-samples-per-book", type=int, default=0, help="0 表示不限制")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--preview", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    args = parse_args(argv)
    if args.self_check:
        self_check()
        return 0
    if args.project:
        from novel_train.projects import get

        project = get(args.project)
        args.input = str(project.raw)
        args.output = str(project.processed)
    return build(args)


if __name__ == "__main__":
    sys.exit(main())
