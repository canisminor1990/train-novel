"""Back-translate a premise for every passage, so training can learn premise → chapter.

The base model reads each passage and writes the prompt an author would have started from
(instruction back-translation, as in Humpback / LongForm / Suri). About 15 s per passage on the 4090.
Results are appended to projects/<project>/premises.jsonl as they come, so a stopped job resumes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path

from novel_train.envload import configure_stdio, load_project_env

load_project_env()

from novel_train.jsonio import read_json, read_jsonl, write_json
from novel_train.models import PREMISE_MODEL, SEQ
from novel_train.prepare import load_premises, premises_path
from novel_train.textutil import count_chars, cut_at_sentence

EDITOR = "你是小说编辑，擅长替作者把一段正文还原成动笔前的写作命题。"
ASK = (
    "下面是小说《{book}》{where}的正文，约 {chars} 字。\n\n{text}\n\n"
    "为这段正文写一个写作命题：作者只看这个命题，就能写出这段情节。\n"
    "要求：60 到 150 字，一段话；写清楚出场人物、地点、发生了什么、情绪怎么变、这一段停在哪里；"
    "用自己的话概括，不抄原句，不评价文笔，不加标题、序号和引号。"
)
_LEAD = re.compile(r"^(?:写作)?命题[:：]\s*")


def status_path(project_root: Path) -> Path:
    return project_root / "premise_status.json"


def read_status(project_root: Path) -> dict | None:
    payload = read_json(status_path(project_root), None)
    return payload if isinstance(payload, dict) else None


def _write_status(project_root: Path, **fields) -> None:
    payload = read_status(project_root) or {}
    payload.update(fields, updated=time.time())
    write_json(status_path(project_root), payload)


def pending_passages(processed: Path) -> tuple[list[dict], int]:
    passages = read_jsonl(processed / "passages.jsonl")
    known = load_premises(premises_path(processed))
    todo = [row for row in passages if row.get("id") and row["id"] not in known]
    return todo, len(passages)


def tidy(text: str) -> str:
    text = _LEAD.sub("", text.strip())
    text = re.sub(r"\s*\n+\s*", "", text).strip("「」“”\"' ")
    return cut_at_sentence(text, 220)


def ask_for(row: dict) -> str:
    chapter = str(row.get("chapter") or "")
    return ASK.format(
        book=row.get("book") or "",
        where=f"「{chapter}」" if chapter else "中一段",
        chars=count_chars(row["text"]),
        text=row["text"],
    )


def generate(project_id: str) -> int:
    from novel_train.ollama_bin import unload_all
    from novel_train.projects import get, prepare

    project = get(project_id)
    todo, total = pending_passages(project.processed)
    _write_status(
        project.root,
        state="running",
        done=total - len(todo),
        total=total,
        started=time.time(),
        detail="正在把 Qwen3.5-4B 载入显存",
        error=None,
    )
    if todo:
        unload_all()
        from unsloth import FastLanguageModel

        from novel_train.speak import continue_prose

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=PREMISE_MODEL,
            max_seq_length=SEQ,
            load_in_4bit=False,
            load_in_16bit=True,
            dtype=None,
        )
        FastLanguageModel.for_inference(model)
        out = premises_path(project.processed)
        began = time.time()
        with out.open("a", encoding="utf-8", newline="\n") as handle:
            for index, row in enumerate(todo, start=1):
                text = continue_prose(
                    model,
                    tokenizer,
                    ask_for(row),
                    instruction=EDITOR,
                    max_seq_length=SEQ,
                    max_new_tokens=240,
                    temperature=0.6,
                    top_p=0.9,
                    repetition_penalty=1.05,
                )
                premise = tidy(text)
                if count_chars(premise) >= 20:
                    handle.write(json.dumps({"id": row["id"], "premise": premise}, ensure_ascii=False) + "\n")
                    handle.flush()
                else:
                    print(f"跳过一段：命题太短 {premise!r}", flush=True)
                rate = (time.time() - began) / index
                _write_status(
                    project.root,
                    done=total - len(todo) + index,
                    eta=rate * (len(todo) - index),
                    detail=f"正在写命题：{row.get('book')} {row.get('chapter') or ''}".strip(),
                )
                if index == 1 or index % 20 == 0:
                    print(f"[{index}/{len(todo)}] {premise}", flush=True)
        del model
    _write_status(project.root, detail="命题写好了，正在重新切样本", eta=None)
    result = prepare(project)
    if not result.get("ok"):
        raise RuntimeError("重新切样本失败：" + str(result.get("log") or "")[-300:])
    stats = result.get("dataset") or {}
    _write_status(
        project.root,
        state="done",
        done=total,
        detail=f"{stats.get('with_premise', 0)} / {stats.get('passages', total)} 段有命题，样本已重新切好",
        ended=time.time(),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    parser = argparse.ArgumentParser(description="用基座模型给每段正文反推一个写作命题。")
    parser.add_argument("--project", required=True)
    args = parser.parse_args(argv)

    from novel_train import jobs
    from novel_train.projects import get

    project = get(args.project)
    pid = jobs.hold_gpu(project.id, "", "premise")
    if pid is None:
        print(jobs.busy_reason() or "显卡同一时间只跑一个任务。")
        return 2
    try:
        return generate(project.id)
    except BaseException as exc:
        traceback.print_exc()
        message = str(exc) or type(exc).__name__
        if "out of memory" in message.lower():
            message = "显存不够。关掉其他占显卡的程序后再试。"
        _write_status(project.root, state="failed", detail=message, error=message, ended=time.time())
        return 1
    finally:
        jobs.release_gpu(pid)


if __name__ == "__main__":
    sys.exit(main())
