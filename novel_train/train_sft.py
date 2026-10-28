"""Style SFT with Unsloth bf16 LoRA. Import this module only after the venv is installed."""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path

from novel_train.envload import ROOT, configure_stdio, load_project_env

load_project_env()

from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from trl import SFTConfig, SFTTrainer

from novel_train.compat import call_accepting
from novel_train.guards import reject_sample_only
from novel_train.metrics import attach_jsonl_callback
from novel_train.pipeline import end_current, estimate_remaining, log_progress, set_stage
from novel_train.prompts import render_chat, to_messages
from novel_train.speak import continue_prose, text_tokenizer

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def resolve(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


def load_config(path: str) -> dict:
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def make_text(tokenizer, rows: dict) -> dict:
    texts = []
    for instruction, user, output in zip(rows["instruction"], rows["input"], rows["output"]):
        messages = to_messages(instruction, user, output)
        texts.append(render_chat(tokenizer, messages, add_generation_prompt=False))
    return {"text": texts}


def _stage_callback(output_dir: Path):
    from transformers import TrainerCallback

    class StageCallback(TrainerCallback):
        began = 0.0
        first_step = 0

        def on_train_begin(self, args, state, control, **kwargs):
            self.began = time.time()
            self.first_step = int(state.global_step or 0)

        def on_log(self, args, state, control, logs=None, **kwargs):
            total = int(state.max_steps or 0)
            step = int(state.global_step or 0)
            progress = (step / total) if total else None
            eta = None
            done = step - self.first_step
            elapsed = time.time() - self.began
            if total and done > 0 and elapsed > 0:
                eta = (total - step) * elapsed / done
            detail = f"第 {step} / {total} 步" if total else f"第 {step} 步"
            set_stage(output_dir, "train", "running", progress, detail, step=step, total=total, eta=eta)

    return StageCallback()


def checkpoint_interval(total_steps: int) -> int:
    return max(10, math.ceil(total_steps / 8))


def make_trainer(model, tokenizer, train_ds, eval_ds, collator, args):
    common = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": collator,
    }
    attempts = [
        {**common, "processing_class": tokenizer},
        {**common, "tokenizer": tokenizer},
    ]
    last_error: TypeError | None = None
    for kwargs in attempts:
        try:
            return SFTTrainer(**kwargs)
        except TypeError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def check_response_mask(trainer) -> None:
    row = trainer.train_dataset[0]
    labels = row.get("labels") if isinstance(row, dict) else None
    if labels is None:
        print("掩码检查：数据集里还没有 labels，训练时由 collator 处理。")
        return
    if hasattr(labels, "tolist"):
        labels = labels.tolist()
    kept = sum(1 for token in labels if token != -100)
    masked = sum(1 for token in labels if token == -100)
    print(f"掩码检查：保留回答 token {kept}，掩掉前文 token {masked}")
    if kept < 8:
        raise SystemExit("回答掩码几乎没留下 token。先不要开训，检查对话模板是否匹配。")


def write_preview(model, tokenizer, example: dict, output_dir: Path, max_seq_length: int) -> None:
    import torch

    FastLanguageModel.for_inference(model)
    text = continue_prose(
        model,
        tokenizer,
        example["input"],
        instruction=example["instruction"],
        max_seq_length=max_seq_length,
        max_new_tokens=256,
    )
    preview = output_dir / "sample_preview.txt"
    preview.write_text(
        "【指令】\n"
        + example["instruction"]
        + "\n\n【输入】\n"
        + example["input"]
        + "\n\n【模型续写】\n"
        + text
        + "\n\n【原文】\n"
        + example["output"]
        + "\n",
        encoding="utf-8",
    )
    print(f"写了一段续写预览：{preview}")


def train(config_path: str, resume: str | None) -> None:
    cfg = load_config(config_path)
    train_path = resolve(cfg["train_file"])
    eval_path = resolve(cfg["eval_file"])
    output_dir = resolve(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if not train_path.exists():
        raise SystemExit(f"找不到训练集 {train_path}。先运行 python -m novel_train.prepare")
    if cfg.get("model_name") != "unsloth/Qwen3.5-4B":
        raise SystemExit("这张卡只训练 unsloth/Qwen3.5-4B。")
    if cfg.get("load_in_4bit") or not cfg.get("load_in_16bit", True):
        raise SystemExit("Qwen3.5 使用 bf16 LoRA。")
    reject_sample_only(train_path)

    (output_dir / "train_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    set_stage(output_dir, "load", "running", None, "正在把 Qwen3.5-4B 载入显存")
    try:
        _train_body(cfg, train_path, eval_path, output_dir, resume)
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            end_current(output_dir, "stopped", "已中断")
        else:
            end_current(output_dir, "failed", str(exc) or type(exc).__name__)
        raise


def _watch_load(output_dir: Path, stop: threading.Event) -> None:
    while not stop.wait(2):
        progress = log_progress(output_dir, "load")
        if progress is None:
            continue
        extra = {}
        remaining = estimate_remaining(output_dir, "load", progress)
        if remaining is not None:
            extra["eta"] = remaining
        set_stage(output_dir, "load", "running", progress, "正在把 Qwen3.5-4B 载入显存", **extra)


def _train_body(cfg: dict, train_path: Path, eval_path: Path, output_dir: Path, resume: str | None) -> None:
    import torch

    stop = threading.Event()
    threading.Thread(target=_watch_load, args=(output_dir, stop), daemon=True).start()
    try:
        loaded = _prepare(cfg, train_path, eval_path, output_dir, resume, torch)
    finally:
        stop.set()
    _run_trainer(cfg, output_dir, resume, *loaded)


def _prepare(cfg: dict, train_path: Path, eval_path: Path, output_dir: Path, resume: str | None, torch) -> tuple:
    torch.manual_seed(cfg["seed"])
    use_bf16 = torch.cuda.is_bf16_supported()
    print(f"模型 {cfg['model_name']}")
    print(f"bf16 {use_bf16}  seq {cfg['max_seq_length']}  rank {cfg['lora_rank']}")

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
        target_modules=TARGET_MODULES,
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg["seed"],
        max_seq_length=cfg["max_seq_length"],
        use_rslora=False,
        loftq_config=None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_preview = json.loads(train_path.read_text(encoding="utf-8").splitlines()[0])
    if eval_path.exists() and eval_path.stat().st_size > 0:
        raw_preview = json.loads(eval_path.read_text(encoding="utf-8").splitlines()[0])
    train_ds = load_dataset("json", data_files=str(train_path), split="train")
    train_ds = train_ds.map(lambda rows: make_text(tokenizer, rows), batched=True, remove_columns=train_ds.column_names)
    eval_ds = None
    if eval_path.exists() and eval_path.stat().st_size > 0:
        eval_ds = load_dataset("json", data_files=str(eval_path), split="train")
        if len(eval_ds) > 0:
            eval_ds = eval_ds.map(
                lambda rows: make_text(tokenizer, rows),
                batched=True,
                remove_columns=eval_ds.column_names,
            )
        else:
            eval_ds = None

    seq = int(cfg["max_seq_length"])
    text_tok = text_tokenizer(tokenizer)

    def fits(rows):
        encoded = text_tok(rows["text"], add_special_tokens=False)
        return [len(ids) <= seq for ids in encoded["input_ids"]]

    before = len(train_ds)
    train_ds = train_ds.filter(fits, batched=True)
    if eval_ds is not None:
        eval_ds = eval_ds.filter(fits, batched=True)
        if len(eval_ds) == 0:
            eval_ds = None
    if len(train_ds) < before:
        print(f"去掉 {before - len(train_ds)} 条超过 {seq} token 的样本，截断会切掉续写结尾。")
    if len(train_ds) == 0:
        raise SystemExit(f"所有样本都超过 {seq} token。调大序列长度，或把前情切短。")

    has_eval = eval_ds is not None
    accumulation = max(1, int(cfg["gradient_accumulation_steps"]))
    max_steps = int(cfg.get("max_steps") or 0)
    total_steps = math.ceil(len(train_ds) / accumulation) * int(cfg["num_train_epochs"])
    if max_steps > 0:
        total_steps = min(total_steps, max_steps)
    interval = checkpoint_interval(total_steps)
    patience = int(cfg.get("early_stopping_patience") or 0) if has_eval else 0
    print(f"约 {total_steps} 步，每 {interval} 步存断点{'并验证' if has_eval else ''}")
    trainer = make_trainer(
        model,
        tokenizer,
        train_ds,
        eval_ds,
        DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        call_accepting(
            SFTConfig,
            dict(
                output_dir=str(output_dir),
                dataset_text_field="text",
                max_seq_length=cfg["max_seq_length"],
                max_length=cfg["max_seq_length"],
                packing=False,
                dataset_num_proc=1,
                per_device_train_batch_size=cfg["per_device_train_batch_size"],
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
                num_train_epochs=cfg["num_train_epochs"],
                max_steps=max_steps if max_steps > 0 else -1,
                learning_rate=cfg["learning_rate"],
                warmup_ratio=cfg["warmup_ratio"],
                lr_scheduler_type=cfg["lr_scheduler_type"],
                weight_decay=cfg["weight_decay"],
                logging_steps=max(1, min(int(cfg["logging_steps"]), total_steps // 40)),
                eval_strategy="steps" if has_eval else "no",
                eval_steps=interval if has_eval else None,
                save_strategy="steps",
                save_steps=interval,
                save_total_limit=cfg["save_total_limit"],
                load_best_model_at_end=patience > 0,
                metric_for_best_model="eval_loss" if patience > 0 else None,
                greater_is_better=False if patience > 0 else None,
                bf16=use_bf16,
                fp16=not use_bf16,
                optim="adamw_8bit",
                seed=cfg["seed"],
                report_to="none",
                dataloader_num_workers=0,
                remove_unused_columns=True,
            ),
        ),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    check_response_mask(trainer)
    attach_jsonl_callback(trainer, output_dir / "metrics.jsonl")
    trainer.add_callback(_stage_callback(output_dir))
    if patience > 0:
        from transformers import EarlyStoppingCallback

        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=patience))
    return trainer, model, tokenizer, raw_preview, total_steps, patience


def _run_trainer(cfg, output_dir, resume, trainer, model, tokenizer, raw_preview, total_steps, patience) -> None:
    set_stage(output_dir, "train", "running", 0, "训练开始", step=0, total=total_steps)
    trainer.train(resume_from_checkpoint=resume)
    state = trainer.state
    finished = int(state.global_step or 0)
    planned = int(state.max_steps or total_steps)
    train_detail = f"跑完 {finished} 步"
    if patience > 0 and finished < planned:
        train_detail = f"验证 loss 连续 {patience} 次没有变好，第 {finished} 步提前停下"
    set_stage(output_dir, "train", "done", 1, train_detail, step=finished, total=planned)
    save_detail = "LoRA 已保存"
    if patience > 0 and state.best_model_checkpoint:
        save_detail = f"保存的是验证 loss 最低的一版：{Path(state.best_model_checkpoint).name}"
    set_stage(output_dir, "save", "running", None, "正在保存 LoRA")
    adapter_dir = output_dir / "lora"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"LoRA 已保存到 {adapter_dir}")
    set_stage(output_dir, "save", "done", 1, save_detail)
    try:
        write_preview(model, tokenizer, raw_preview, output_dir, cfg["max_seq_length"])
    except Exception as exc:
        print(f"训练已完成。续写预览跳过：{exc}")


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    parser = argparse.ArgumentParser(description="在 4090 上做小说文风 SFT。")
    parser.add_argument("--config", default="configs/sft_4090.json")
    parser.add_argument("--resume", default=None, help="checkpoint 目录，用于断点续训")
    args = parser.parse_args(argv)
    train(args.config, args.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
