"""Base models a project can train on this RTX 4090. Both are Qwen3.5 bf16 LoRA."""

from __future__ import annotations

# 正文 4000 到 6500 字约 3000 到 4800 token。加上命题和前情后，用 6144 才不会把这些样本整段丢掉。
SEQ = 6144

MODELS = {
    "4b": {
        "label": "Qwen3.5-4B",
        "name": "unsloth/Qwen3.5-4B",
        "vram_gb": 10,
        "summary": "约 10GB 显存，训得快。新项目先用它跑通。",
    },
    "9b": {
        "label": "Qwen3.5-9B",
        "name": "unsloth/Qwen3.5-9B",
        "vram_gb": 22,
        "summary": "约 22GB 显存，长文更连贯，训练要两倍多的时间。序列加到 6144 后更贴这张卡的上限，开训前关掉其他占显存的程序。",
    },
}
DEFAULT_MODEL = "4b"
# 反推命题只做概括，用已经缓存的 4B 就够了。
PREMISE_MODEL = MODELS["4b"]["name"]


def model_key(name: str | None) -> str | None:
    for key, spec in MODELS.items():
        if spec["name"] == name:
            return key
    return None


def model_spec(key: str | None) -> dict:
    return MODELS.get(key or "", MODELS[DEFAULT_MODEL])


def model_label(name: str | None) -> str:
    key = model_key(name)
    return MODELS[key]["label"] if key else str(name or "")


def public_models(selected: str | None) -> dict:
    current = selected if selected in MODELS else DEFAULT_MODEL
    items = [
        {"id": key, "label": spec["label"], "summary": spec["summary"], "vram_gb": spec["vram_gb"], "selected": key == current}
        for key, spec in MODELS.items()
    ]
    return {"items": items, "selected": current}
