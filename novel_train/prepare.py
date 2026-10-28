"""Turn raw novels into SFT continuation pairs and cleaned CPT books."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from novel_train.envload import ROOT, configure_stdio
from novel_train.prompts import build_user_input, instruction_for
from novel_train.textutil import (
    clean_document,
    count_chars,
    iter_windows,
    load_document,
    self_check,
)

SUFFIXES = {".txt", ".text", ".md", ".epub"}


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
    seen_outputs: set[str] = set()
    stats_books = []

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

        flat: list[str] = []
        owners: list[str] = []
        for title, paragraphs in chapters:
            for paragraph in paragraphs:
                flat.append(paragraph)
                owners.append(title)

        pairs: list[dict] = []
        for sample_index, (context, output, start) in enumerate(
            iter_windows(
                flat,
                context_chars=args.context_chars,
                context_min=args.context_min,
                output_min=args.output_min,
                output_max=args.output_max,
                stride_chars=args.stride_chars,
            )
        ):
            if output in seen_outputs:
                continue
            seen_outputs.add(output)
            title = owners[start]
            chapter_name = title if title != "正文" and sample_index % 2 == 0 else ""
            pairs.append(
                {
                    "instruction": instruction_for(sample_index),
                    "input": build_user_input(chapter_name, context),
                    "output": output,
                    "book": book_name,
                    "chapter": title,
                }
            )
            if args.max_samples_per_book and len(pairs) >= args.max_samples_per_book:
                break

        eval_count = 0
        if len(pairs) >= 20:
            eval_count = max(1, int(len(pairs) * args.eval_ratio))
        elif len(pairs) >= 8:
            eval_count = 1
        book_train = pairs[:-eval_count] if eval_count else pairs
        book_eval = pairs[-eval_count:] if eval_count else []
        train_rows.extend(book_train)
        eval_rows.extend(book_eval)
        stats_books.append(
            {
                "book": book_name,
                "chars": char_count,
                "chapters": len(chapters),
                "raw_chars": count_chars(raw),
                "junk_lines": junk,
                "duplicate_lines": report["duplicate_lines"],
                "merged_lines": report["merged_lines"],
                "duplicate_chapters": report["duplicate_chapters"],
                "train": len(book_train),
                "eval": len(book_eval),
            }
        )
        print(
            f"{book_name}  {char_count} 字  {len(chapters)} 章  "
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

    stats = {
        "books": stats_books,
        "train_samples": len(train_rows),
        "eval_samples": len(eval_rows),
        "context_chars": args.context_chars,
        "output_min": args.output_min,
        "output_max": args.output_max,
        "stride_chars": args.stride_chars,
    }
    stats_path = output_dir / "stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("")
    print(f"SFT 训练 {len(train_rows)} 条，验证 {len(eval_rows)} 条")
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
    parser = argparse.ArgumentParser(description="把小说原文切成续写训练集，不需要人工打标。")
    parser.add_argument("--self-check", action="store_true", help="用一段合成文本检查清洗和切窗")
    parser.add_argument("--project", default=None, help="项目 ID，读 projects/<ID>/raw，写 projects/<ID>/processed")
    parser.add_argument("--input", default="data/raw")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--context-chars", type=int, default=1200)
    parser.add_argument("--context-min", type=int, default=300)
    parser.add_argument("--output-min", type=int, default=200)
    parser.add_argument("--output-max", type=int, default=800)
    parser.add_argument("--stride-chars", type=int, default=400)
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
