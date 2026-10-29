"""Domain continued pretraining on cleaned novel text."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from novel_train.envload import ROOT, configure_stdio, load_project_env

load_project_env()

from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from novel_train.guards import reject_sample_only
from novel_train.metrics import attach_jsonl_callback, write_status
from novel_train.speak import text_tokenizer
from novel_train.train_sft import OWN_KEYS, sft_overrides

CPT_KEYS = OWN_KEYS | {"books_file"}


def resolve(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


def load_books(path: Path) -> list[dict]:
    books = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                books.append(json.loads(line))
    return books


def chunk_book(text: str, tokenizer, max_seq_length: int) -> list[str]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        token_ids.append(tokenizer.eos_token_id)
    chunks = []
    for start in range(0, len(token_ids), max_seq_length):
        piece = token_ids[start : start + max_seq_length]
        if len(piece) < 128:
            continue
        chunks.append(tokenizer.decode(piece, skip_special_tokens=False))
    return chunks


def train(config_path: str, resume: str | None) -> None:
    import torch

    cfg = json.loads(resolve(config_path).read_text(encoding="utf-8"))
    books_path = resolve(cfg["books_file"])
    output_dir = resolve(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if not books_path.exists():
        raise SystemExit(f"找不到 {books_path}。先运行 python -m novel_train.prepare")

    books = load_books(books_path)
    if not books:
        raise SystemExit("books.jsonl 是空的。")
    reject_sample_only(books_path)
    overrides = sft_overrides(cfg, CPT_KEYS)
    (output_dir / "train_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_status(output_dir, "running", "正在加载模型或训练")

    use_bf16 = torch.cuda.is_bf16_supported()
    print(f"CPT 模型 {cfg['model_name']}  书 {len(books)} 本")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg.get("load_in_4bit", False),
        load_in_16bit=cfg.get("load_in_16bit", True),
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_rank"],
        target_modules=cfg.get("lora_target_modules"),
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg["seed"],
        use_rslora=cfg.get("use_rslora", False),
        loftq_config=None,
    )
    model.print_trainable_parameters()
    text_tok = text_tokenizer(tokenizer)

    train_texts: list[str] = []
    eval_texts: list[str] = []
    for book in books:
        chunks = chunk_book(book["text"], text_tok, cfg["max_seq_length"])
        if len(chunks) >= 8:
            train_texts.extend(chunks[:-1])
            eval_texts.append(chunks[-1])
        else:
            train_texts.extend(chunks)
        print(f"{book['book']}  {len(chunks)} 段")
    if not train_texts:
        raise SystemExit("切不出 CPT 片段。")

    train_ds = Dataset.from_list([{"text": text} for text in train_texts])
    eval_ds = Dataset.from_list([{"text": text} for text in eval_texts]) if eval_texts else None
    print(f"CPT 训练片段 {len(train_texts)}  验证片段 {len(eval_texts)}")

    args = SFTConfig(
        **{
            **overrides,
            "output_dir": str(output_dir),
            "dataset_text_field": "text",
            "max_length": cfg["max_seq_length"],
            "packing": False,
            "eval_strategy": "epoch" if eval_ds is not None else "no",
            "save_strategy": "epoch",
            "bf16": use_bf16,
            "fp16": not use_bf16,
        }
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    attach_jsonl_callback(trainer, output_dir / "metrics.jsonl")
    try:
        trainer.train(resume_from_checkpoint=resume)
        adapter_dir = output_dir / "lora"
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        print(f"CPT LoRA 已保存到 {adapter_dir}")
        write_status(output_dir, "finished", "CPT LoRA 已保存")
    except Exception as exc:
        write_status(output_dir, "failed", str(exc))
        raise


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    parser = argparse.ArgumentParser(description="用小说原文做领域续训（CPT）。")
    parser.add_argument("--config", default="configs/cpt_4090.json")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args(argv)
    train(args.config, args.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
