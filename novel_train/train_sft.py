"""Style SFT with Unsloth bf16 LoRA. Import this module only inside the pixi environment."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import threading
import time
from pathlib import Path

from novel_train.envload import ROOT, configure_stdio, load_project_env

load_project_env()

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from novel_train.guards import reject_sample_only
from novel_train.metrics import attach_jsonl_callback
from novel_train.models import MODELS, model_key, model_label, model_spec
from novel_train.ollama_bin import tokens_for
from novel_train.pipeline import end_current, estimate_remaining, log_progress, set_stage
from novel_train.prompts import to_messages
from novel_train.speak import continue_prose, text_tokenizer

# 配置里除了这些，其余字段原样交给 SFTConfig。导入 Unsloth 后 SFTConfig 会吞掉不认识的参数，
# 所以要先按 dataclass 字段核对。
OWN_KEYS = {
    "model_name",
    "max_seq_length",
    "load_in_4bit",
    "load_in_16bit",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "lora_target_modules",
    "use_rslora",
    "train_file",
    "eval_file",
    "early_stopping_patience",
}
SFT_DEFAULTS = {
    "optim": "adamw_8bit",
    "max_grad_norm": 1.0,
    "per_device_eval_batch_size": 1,
    "dataset_num_proc": 1,
    "dataloader_num_workers": 0,
    "report_to": "none",
}
NO_THINKING = {"enable_thinking": False}


def resolve(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


def load_config(path: str) -> dict:
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def sft_overrides(cfg: dict, own: set[str] = OWN_KEYS) -> dict:
    known = {field.name for field in dataclasses.fields(SFTConfig)}
    passthrough = {key: value for key, value in cfg.items() if key not in own}
    # 旧运行冻结的配置里是 transformers 5 删掉的 warmup_ratio。
    if "warmup_ratio" in passthrough:
        passthrough.setdefault("warmup_steps", passthrough.pop("warmup_ratio"))
    unknown = sorted(set(passthrough) - known)
    if unknown:
        raise SystemExit("配置里有 SFTConfig 不认识的字段：" + "、".join(unknown))
    if (passthrough.get("max_steps") or 0) <= 0:
        passthrough["max_steps"] = -1
    return {**SFT_DEFAULTS, **passthrough}


def to_prompt_completion(row: dict) -> dict:
    messages = to_messages(row["instruction"], row["input"], row["output"])
    return {"prompt": messages[:-1], "completion": messages[-1:], "chat_template_kwargs": NO_THINKING}


def load_split(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    ds = load_dataset("json", data_files=str(path), split="train")
    if len(ds) == 0:
        return None
    return ds.map(to_prompt_completion, remove_columns=ds.column_names)


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


def check_response_mask(trainer) -> None:
    row = trainer.train_dataset[0]
    if "labels" in row:
        keep = [token != -100 for token in row["labels"]]
    elif "completion_mask" in row:
        keep = [bool(flag) for flag in row["completion_mask"]]
    else:
        raise SystemExit("训练集里既没有 labels 也没有 completion_mask，回答掩码没有生效。")
    kept = sum(keep)
    masked = len(keep) - kept
    print(f"掩码检查：保留回答 token {kept}，掩掉前文 token {masked}")
    if kept < 8:
        raise SystemExit("回答掩码几乎没留下 token。先不要开训，检查对话模板是否匹配。")


def write_preview(model, tokenizer, example: dict, output_dir: Path, max_seq_length: int) -> None:
    FastLanguageModel.for_inference(model)
    text = continue_prose(
        model,
        tokenizer,
        example["input"],
        instruction=example["instruction"],
        max_seq_length=max_seq_length,
        max_new_tokens=min(2400, tokens_for(len(example["output"]))),
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
    if model_key(cfg.get("model_name")) is None:
        raise SystemExit("这张卡只训练 " + "、".join(spec["name"] for spec in MODELS.values()) + "。")
    if cfg.get("load_in_4bit") or not cfg.get("load_in_16bit", True):
        raise SystemExit("Qwen3.5 使用 bf16 LoRA。")
    reject_sample_only(train_path)
    sft_overrides(cfg)

    (output_dir / "train_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    set_stage(output_dir, "load", "running", None, _loading(cfg))
    try:
        _train_body(cfg, train_path, eval_path, output_dir, resume)
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            end_current(output_dir, "stopped", "已中断")
        else:
            end_current(output_dir, "failed", _failure(cfg, exc))
        raise


def _loading(cfg: dict) -> str:
    return f"正在把 {model_label(cfg['model_name'])} 载入显存"


def _failure(cfg: dict, exc: BaseException) -> str:
    text = str(exc) or type(exc).__name__
    if "out of memory" in text.lower():
        if model_key(cfg["model_name"]) == "9b":
            return "显存不够。9B 要约 22GB：关掉游戏、浏览器硬件加速和别的模型后从断点继续，还不够就换回 4B。"
        return "显存不够。关掉其他占显卡的程序后从断点继续。"
    return text


def _check_vram(cfg: dict, torch) -> None:
    from novel_train.ollama_bin import unload_all

    unload_all()
    if not torch.cuda.is_available():
        return
    free, _total = torch.cuda.mem_get_info()
    need = model_spec(model_key(cfg["model_name"]))["vram_gb"]
    free_gb = free / 1024**3
    print(f"可用显存 {free_gb:.1f} GB，{model_label(cfg['model_name'])} 约需 {need} GB", flush=True)
    if free_gb < need:
        print("可用显存少于需要的量，训练中可能爆显存。先关掉其他占显卡的程序。", flush=True)


def _watch_load(output_dir: Path, stop: threading.Event, detail: str) -> None:
    while not stop.wait(2):
        progress = log_progress(output_dir, "load")
        if progress is None:
            continue
        extra = {}
        remaining = estimate_remaining(output_dir, "load", progress)
        if remaining is not None:
            extra["eta"] = remaining
        set_stage(output_dir, "load", "running", progress, detail, **extra)


def _train_body(cfg: dict, train_path: Path, eval_path: Path, output_dir: Path, resume: str | None) -> None:
    import torch

    _check_vram(cfg, torch)
    stop = threading.Event()
    threading.Thread(target=_watch_load, args=(output_dir, stop, _loading(cfg)), daemon=True).start()
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
    # Qwen3.5 的 32 层里有 24 层是线性注意力（in_proj_qkv、in_proj_z、out_proj），
    # 不写 target_modules 时由 Unsloth 选出语言部分的全部线性层，并跳过视觉塔。
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
    if text_tok.pad_token is None:
        text_tok.pad_token = text_tok.eos_token

    raw_preview = json.loads(train_path.read_text(encoding="utf-8").splitlines()[0])
    if eval_path.exists() and eval_path.stat().st_size > 0:
        raw_preview = json.loads(eval_path.read_text(encoding="utf-8").splitlines()[0])
    train_ds = load_split(train_path)
    if train_ds is None:
        raise SystemExit(f"训练集 {train_path} 是空的。")
    eval_ds = load_split(eval_path)

    seq = int(cfg["max_seq_length"])

    def fits(row):
        encoded = text_tok.apply_chat_template(
            row["prompt"] + row["completion"],
            tokenize=True,
            return_dict=True,
            **row["chat_template_kwargs"],
        )
        return len(encoded["input_ids"]) <= seq

    before = len(train_ds)
    train_ds = train_ds.filter(fits)
    if eval_ds is not None:
        eval_ds = eval_ds.filter(fits)
        if len(eval_ds) == 0:
            eval_ds = None
    if len(train_ds) < before:
        print(f"去掉 {before - len(train_ds)} 条超过 {seq} token 的样本，截断会切掉章末。")
    if len(train_ds) == 0:
        raise SystemExit(f"所有样本都超过 {seq} token。调大序列长度，或把前情切短。")

    overrides = sft_overrides(cfg)
    has_eval = eval_ds is not None
    accumulation = max(1, int(overrides.get("gradient_accumulation_steps", 1)))
    batch = max(1, int(overrides.get("per_device_train_batch_size", 1)))
    total_steps = math.ceil(len(train_ds) / (batch * accumulation)) * math.ceil(overrides.get("num_train_epochs", 1))
    if overrides["max_steps"] > 0:
        total_steps = min(total_steps, overrides["max_steps"])
    interval = checkpoint_interval(total_steps)
    patience = int(cfg.get("early_stopping_patience") or 0) if has_eval else 0
    print(f"约 {total_steps} 步，每 {interval} 步存断点{'并验证' if has_eval else ''}")
    args = SFTConfig(
        **{
            **overrides,
            "output_dir": str(output_dir),
            # 超长样本在上面已经丢掉，这里不让 TRL 再截断章末。
            "max_length": None,
            "packing": False,
            "completion_only_loss": True,
            "bf16": use_bf16,
            "fp16": not use_bf16,
            "logging_steps": max(1, min(int(overrides.get("logging_steps", 10)), total_steps // 40)),
            "eval_strategy": "steps" if has_eval else "no",
            "eval_steps": interval if has_eval else None,
            "save_strategy": "steps",
            "save_steps": interval,
            "load_best_model_at_end": patience > 0,
            "metric_for_best_model": "eval_loss" if patience > 0 else None,
            "greater_is_better": False if patience > 0 else None,
        }
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
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
