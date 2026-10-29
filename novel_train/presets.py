"""Three SFT presets. The base model (Qwen3.5-4B or 9B, bf16 LoRA) is chosen per project."""

from __future__ import annotations

from novel_train.models import SEQ

# 一条样本是一整段正文（约 4000 到 6500 字），是旧版两三百字续写的十几倍，
# 所以梯度累积从 8 降到 4：每步看到的 token 仍比旧版多，步数不至于太少。
PRESETS = {
    "quick": {
        "label": "试跑",
        "summary": "十来分钟走完训练、导出和挂载，确认样本、显存和 Ollama 都没问题。换模型后也先跑一次。",
        "config": {
            "max_seq_length": SEQ,
            "lora_rank": 8,
            "lora_alpha": 8,
            "lora_dropout": 0.0,
            "learning_rate": 0.0002,
            "num_train_epochs": 1,
            "max_steps": 20,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 0.1,
            "weight_decay": 0.0,
            "lr_scheduler_type": "constant_with_warmup",
            "seed": 3407,
            "logging_steps": 1,
            "save_total_limit": 1,
            "early_stopping_patience": 0,
        },
    },
    "standard": {
        "label": "标准",
        "summary": "按这张 4090 的默认安排训练，适合先跑通一本长篇。",
        "config": {
            "max_seq_length": SEQ,
            "lora_rank": 16,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "learning_rate": 0.0001,
            "num_train_epochs": 2,
            "max_steps": 0,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 0.05,
            "weight_decay": 0.01,
            "lr_scheduler_type": "cosine",
            "seed": 3407,
            "logging_steps": 5,
            "save_total_limit": 2,
            "early_stopping_patience": 2,
        },
    },
    "careful": {
        "label": "细读",
        "summary": "多训一轮、适配更宽，想更贴原文时用。样本少于 150 条时容易背原文。",
        "config": {
            "max_seq_length": SEQ,
            "lora_rank": 32,
            "lora_alpha": 32,
            "lora_dropout": 0.0,
            "learning_rate": 0.0001,
            "num_train_epochs": 3,
            "max_steps": 0,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 0.05,
            "weight_decay": 0.01,
            "lr_scheduler_type": "cosine",
            "seed": 3407,
            "logging_steps": 5,
            "save_total_limit": 2,
            "early_stopping_patience": 2,
        },
    },
}
DEFAULT_PRESET = "quick"


def preset_config(name: str) -> dict:
    if name not in PRESETS:
        raise ValueError("预设只能是试跑、标准或细读。")
    return dict(PRESETS[name]["config"])


def preset_label(name: str | None) -> str:
    spec = PRESETS.get(name or "")
    return spec["label"] if spec else "自定义"


def estimate_steps(name: str, train_samples: int) -> int:
    cfg = PRESETS[name]["config"]
    per_epoch = -(-max(0, train_samples) // max(1, cfg["gradient_accumulation_steps"]))
    total = per_epoch * cfg["num_train_epochs"]
    if cfg.get("max_steps"):
        total = min(total, cfg["max_steps"]) if total else cfg["max_steps"]
    return total


def public_presets(selected: str | None, train_samples: int = 0) -> dict:
    items = []
    for name, spec in PRESETS.items():
        cfg = spec["config"]
        items.append(
            {
                "id": name,
                "label": spec["label"],
                "summary": spec["summary"],
                "epochs": cfg["num_train_epochs"],
                "max_steps": cfg.get("max_steps") or None,
                "rank": cfg["lora_rank"],
                "seq": cfg["max_seq_length"],
                "steps": estimate_steps(name, train_samples) if train_samples else None,
                "early_stop": bool(cfg.get("early_stopping_patience")),
                "selected": name == selected,
            }
        )
    return {"items": items, "selected": selected if selected in PRESETS else None}
