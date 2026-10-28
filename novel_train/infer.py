"""Continue a scene with a trained LoRA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from novel_train.envload import ROOT, configure_stdio, load_project_env

load_project_env()

from unsloth import FastLanguageModel

from novel_train.prompts import INSTRUCTIONS
from novel_train.speak import prose_from_context
from novel_train.textutil import count_chars, tail_text


def resolve(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


def read_context(args: argparse.Namespace) -> str:
    if args.context_file:
        return resolve(args.context_file).read_text(encoding="utf-8")
    if args.text:
        return args.text
    if not sys.stdin.isatty():
        return sys.stdin.read()
    print("输入前情。单独一行输入 /// 后回车，开始续写。")
    lines: list[str] = []
    while True:
        line = input()
        if line.strip() == "///":
            break
        lines.append(line)
    return "\n".join(lines)


def project_adapter(project_id: str | None) -> Path:
    from novel_train.projects import get, list_projects
    from novel_train.runs import list_run_dirs

    if project_id:
        project = get(project_id)
    else:
        found = list_projects()
        if len(found) != 1:
            raise SystemExit("有多个项目。用 --project 指定一个，或用 --adapter 指定 LoRA 目录。")
        project = found[0]
    mounted = project.meta().get("mounted_run")
    candidates = ([project.runs / mounted] if mounted else []) + list_run_dirs(project)
    for run_dir in candidates:
        if (run_dir / "lora" / "adapter_model.safetensors").exists():
            print(f"使用 {run_dir.relative_to(ROOT)}")
            return run_dir / "lora"
    raise SystemExit("这个项目还没有训练好的 LoRA。")


def generate(args: argparse.Namespace) -> int:
    context = tail_text(read_context(args).strip(), args.context_chars)
    if count_chars(context) < 20:
        print("前情太短。")
        return 2

    adapter = resolve(args.adapter) if args.adapter else project_adapter(args.project)
    if not (adapter / "adapter_model.safetensors").exists():
        print(f"{adapter} 里没有 LoRA。")
        return 2
    load_in_4bit = False
    load_in_16bit = True
    saved = adapter.parent / "train_config.json"
    if saved.exists():
        saved_cfg = json.loads(saved.read_text(encoding="utf-8"))
        load_in_4bit = bool(saved_cfg.get("load_in_4bit", False))
        load_in_16bit = bool(saved_cfg.get("load_in_16bit", not load_in_4bit))
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter),
        max_seq_length=args.max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_16bit=load_in_16bit,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)
    text = prose_from_context(
        model,
        tokenizer,
        context,
        instruction=args.instruction,
        max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    print(text)
    if args.output:
        out = resolve(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    parser = argparse.ArgumentParser(description="用训好的 LoRA 按前文续写。")
    parser.add_argument("--project", default=None, help="项目 ID。默认用它当前挂到 Ollama 的那次运行")
    parser.add_argument("--adapter", default=None, help="直接指定 LoRA 目录，优先于 --project")
    parser.add_argument("--context-file", default=None)
    parser.add_argument("--text", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--instruction", default=INSTRUCTIONS[0])
    parser.add_argument("--context-chars", type=int, default=1200)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    return generate(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
