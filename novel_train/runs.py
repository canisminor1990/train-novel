"""Runs inside a project. Each run owns a frozen copy of its config and samples."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from novel_train.envload import ROOT
from novel_train.export_ollama import build_dir, merged_model_ready
from novel_train.ollama_bin import think_model_name
from novel_train.jsonio import dir_size, read_json, read_jsonl, tail_text, write_json
from novel_train.models import MODELS, model_key, model_label
from novel_train.pipeline import view as stage_view
from novel_train.presets import preset_config, preset_label
from novel_train.projects import Project, dataset, readiness

BASE_CONFIG = ROOT / "configs" / "sft_4090.json"
EVAL_CAP = 200
LIVE_STATES = {"queued", "running"}


def run_meta(run_dir: Path) -> dict:
    return read_json(run_dir / "run.json", {}) or {}


def update_run(run_dir: Path, **fields) -> dict:
    payload = run_meta(run_dir)
    payload.update(fields)
    write_json(run_dir / "run.json", payload)
    return payload


def base_config() -> dict:
    payload = read_json(BASE_CONFIG, None)
    if not isinstance(payload, dict):
        raise ValueError("configs/sft_4090.json 读不出来。")
    return payload


def locked(cfg: dict) -> dict:
    if model_key(cfg.get("model_name")) is None:
        allowed = "、".join(spec["name"] for spec in MODELS.values())
        raise ValueError(f"模型只能是 {allowed}，这次没有开训。")
    if cfg.get("load_in_4bit") or cfg.get("load_in_16bit") is False:
        raise ValueError("Qwen3.5 使用 bf16 LoRA，这次没有开训。")
    cfg["load_in_4bit"] = False
    cfg["load_in_16bit"] = True
    cfg["per_device_train_batch_size"] = 1
    return cfg


def _new_run_id(project: Project) -> str:
    base = "r-" + time.strftime("%Y%m%d-%H%M%S")
    candidate = base
    suffix = 1
    while (project.runs / candidate).exists():
        suffix += 1
        candidate = f"{base}-{suffix}"
    return candidate


def create_run(project: Project) -> Path:
    ready, reason = readiness(project)
    if not ready:
        raise ValueError(reason)
    meta = project.meta()
    preset = meta.get("preset")
    base = project.base_model
    cfg = locked({**base_config(), **preset_config(preset), "model_name": MODELS[base]["name"]})
    run_dir = project.runs / _new_run_id(project)
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True)
    shutil.copy2(project.train_file, data_dir / "train.jsonl")
    eval_rows = read_jsonl(project.eval_file)[:EVAL_CAP]
    with (data_dir / "eval.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in eval_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    cfg["train_file"] = str((data_dir / "train.jsonl").relative_to(ROOT)).replace("\\", "/")
    cfg["eval_file"] = str((data_dir / "eval.jsonl").relative_to(ROOT)).replace("\\", "/")
    cfg["output_dir"] = str(run_dir.relative_to(ROOT)).replace("\\", "/")
    write_json(run_dir / "config.json", cfg)
    stats = dataset(project) or {}
    now = time.time()
    write_json(
        run_dir / "run.json",
        {
            "id": run_dir.name,
            "project": project.id,
            "preset": preset,
            "base_model": base,
            "created": now,
            "started": None,
            "ended": None,
            "state": "queued",
            "detail": "正在启动",
            "pid": None,
            "dataset": {
                "books": [row.get("book") for row in stats.get("books") or []],
                "train_samples": stats.get("train_samples") or 0,
                "eval_samples": len(eval_rows),
            },
        },
    )
    return run_dir


def list_run_dirs(project: Project) -> list[Path]:
    if not project.runs.exists():
        return []
    dirs = [path for path in project.runs.iterdir() if path.is_dir() and path.name.startswith("r-")]
    return sorted(dirs, key=lambda path: path.name, reverse=True)


def latest_checkpoint(run_dir: Path) -> Path | None:
    found = [path for path in run_dir.glob("checkpoint-*") if path.is_dir()]

    def step(path: Path) -> int:
        try:
            return int(path.name.split("-")[-1])
        except ValueError:
            return -1

    return max(found, key=step) if found else None


def load_metrics(run_dir: Path) -> list[dict]:
    merged: dict[tuple, dict] = {}
    sources = []
    for state_path in run_dir.glob("checkpoint-*/trainer_state.json"):
        sources.extend((read_json(state_path, {}) or {}).get("log_history", []))
    sources.extend(read_jsonl(run_dir / "metrics.jsonl"))
    for row in sources:
        if not isinstance(row, dict) or not isinstance(row.get("step"), int):
            continue
        kind = "eval" if "eval_loss" in row else "train" if "loss" in row else "other"
        merged[(row["step"], kind)] = row
    rows = [row for (_, kind), row in merged.items() if kind != "other"]
    rows.sort(key=lambda row: (row["step"], 1 if "eval_loss" in row else 0))
    return rows


def _state(meta: dict, alive: bool, has_adapter: bool) -> str:
    state = str(meta.get("state") or "queued")
    if state in LIVE_STATES and not alive:
        created = float(meta.get("queued_at") or meta.get("created") or 0)
        if state == "queued" and time.time() - created < 20:
            return "queued"
        # LoRA is already saved. Dying here is an export that stopped, not a training to resume.
        if meta.get("phase") == "export" or (has_adapter and meta.get("phase") != "train"):
            return "trained"
        return "interrupted"
    return state


def _summary_numbers(metrics: list[dict]) -> dict:
    train = [row for row in metrics if isinstance(row.get("loss"), (int, float))]
    evals = [row for row in metrics if isinstance(row.get("eval_loss"), (int, float))]
    best = min(evals, key=lambda row: row["eval_loss"]) if evals else None
    return {
        "last_step": metrics[-1]["step"] if metrics else 0,
        "last_loss": train[-1]["loss"] if train else None,
        "first_loss": train[0]["loss"] if train else None,
        "last_eval": evals[-1]["eval_loss"] if evals else None,
        "best_eval": best["eval_loss"] if best else None,
        "best_eval_step": best["step"] if best else None,
        "eval_rising": bool(best and len(evals) >= 2 and evals[-1]["eval_loss"] > best["eval_loss"] * 1.02),
    }


def decide_action(
    *,
    alive: bool,
    state: str,
    has_adapter: bool,
    has_checkpoint: bool,
    has_gguf: bool,
    mounted: bool,
) -> str | None:
    """The one next control for this run. Finished work is never the thing that gets repeated."""
    if alive or state in LIVE_STATES:
        return "pause"
    if has_gguf:
        return None if mounted else "mount"
    if has_adapter or has_checkpoint:
        return "continue"
    if state == "failed":
        return "retry"
    if state in {"stopped", "interrupted"}:
        return "start"
    return None


def _model_label(run_dir: Path, meta: dict) -> str:
    key = meta.get("base_model")
    if key in MODELS:
        return MODELS[key]["label"]
    cfg = read_json(run_dir / "config.json", {}) or {}
    return model_label(cfg.get("model_name") or MODELS["4b"]["name"])


def summary(project: Project, run_dir: Path, active_run: str | None, mounted_run: str | None) -> dict:
    meta = run_meta(run_dir)
    alive = active_run == run_dir.name
    has_adapter = (run_dir / "lora" / "adapter_model.safetensors").exists()
    stages = stage_view(run_dir, alive)
    metrics = load_metrics(run_dir)
    train_stage = next(stage for stage in stages if stage["id"] == "train")
    current = next((stage for stage in stages if stage["state"] in {"running", "failed", "stopped"}), None)
    state = _state(meta, alive, has_adapter)
    detail = str(meta.get("detail") or "")
    if state == "trained" and str(meta.get("state") or "") in LIVE_STATES and not alive:
        detail = "导出进程已经不在。LoRA 还在，可以继续导出。"
    has_merged = merged_model_ready(build_dir(run_dir))
    has_checkpoint = latest_checkpoint(run_dir) is not None
    has_gguf = gguf_file(run_dir) is not None
    mounted = mounted_run == run_dir.name
    return {
        "id": run_dir.name,
        "project": project.id,
        "preset": meta.get("preset"),
        "preset_label": preset_label(meta.get("preset")),
        "model_label": _model_label(run_dir, meta),
        "state": state,
        "detail": detail,
        "created": meta.get("created"),
        "started": meta.get("started"),
        "ended": meta.get("ended"),
        "alive": alive,
        "mounted": mounted,
        "phase": meta.get("phase"),
        "has_adapter": has_adapter,
        "has_checkpoint": has_checkpoint,
        "has_merged": has_merged,
        "has_gguf": has_gguf,
        "next_action": decide_action(
            alive=alive,
            state=state,
            has_adapter=has_adapter,
            has_checkpoint=has_checkpoint,
            has_gguf=has_gguf,
            mounted=mounted,
        ),
        "stage": current["id"] if current else None,
        "progress": train_stage.get("progress"),
        "step": train_stage.get("step"),
        "total": train_stage.get("total"),
        "eta": train_stage.get("eta"),
        "dataset": meta.get("dataset"),
        **_summary_numbers(metrics),
    }


def gguf_dir(run_dir: Path) -> Path:
    return run_dir / "ollama_build_gguf"


def gguf_file(run_dir: Path) -> Path | None:
    directory = gguf_dir(run_dir)
    if not directory.exists():
        return None
    files = [path for path in directory.glob("*.gguf") if "mmproj" not in path.name.lower() and path.stat().st_size > 50_000_000]
    quantized = [path for path in files if any(tag in path.name.lower() for tag in ("q4", "q5", "q8"))]
    return (quantized or files or [None])[0]


def detail(project: Project, run_dir: Path, active_run: str | None, mounted_run: str | None) -> dict:
    alive = active_run == run_dir.name
    preview_path = run_dir / "sample_preview.txt"
    gguf = gguf_file(run_dir)
    return {
        **summary(project, run_dir, active_run, mounted_run),
        "stages": stage_view(run_dir, alive),
        "metrics": load_metrics(run_dir),
        "config": read_json(run_dir / "train_config.json", None) or read_json(run_dir / "config.json", None),
        "preview": preview_path.read_text(encoding="utf-8") if preview_path.exists() else "",
        "train_log": tail_text(run_dir / "train.log", 6000),
        "export_log": tail_text(run_dir / "export.log", 3000),
        "artifacts": {
            "dir": str(run_dir),
            "lora_bytes": dir_size(run_dir / "lora"),
            "gguf": gguf.name if gguf else None,
            "gguf_bytes": gguf.stat().st_size if gguf else 0,
            "disk_bytes": dir_size(run_dir),
        },
        "ollama_model": project.ollama_model,
        "ollama_think_model": think_model_name(project.ollama_model),
    }


def delete_run(run_dir: Path) -> None:
    shutil.rmtree(run_dir)
