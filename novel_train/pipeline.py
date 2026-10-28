"""Stage progress for one run: load, train, save, then the Ollama export."""

from __future__ import annotations

import re
import time
from pathlib import Path

from novel_train.jsonio import read_json, write_json

STAGES = [
    ("load", "加载模型"),
    ("train", "训练"),
    ("save", "保存 LoRA"),
    ("merge", "合并权重"),
    ("gguf", "导出 GGUF"),
    ("ollama", "挂载到 Ollama"),
]
ORDER = [item[0] for item in STAGES]
LABELS = dict(STAGES)
EXPORT_STAGES = ("merge", "gguf", "ollama")


def stage_path(run_dir: Path) -> Path:
    return run_dir / "stage.json"


def set_stage(
    run_dir: Path,
    stage_id: str,
    state: str,
    progress: float | None = None,
    detail: str = "",
    **extra,
) -> None:
    if stage_id not in ORDER:
        raise ValueError(f"未知阶段 {stage_id}")
    path = stage_path(run_dir)
    payload = read_json(path, {}) or {}
    stages = payload.get("stages") if isinstance(payload.get("stages"), dict) else {}
    now = time.time()
    if state in {"running", "done"}:
        for earlier in ORDER[: ORDER.index(stage_id)]:
            entry = stages.get(earlier) or {}
            if entry.get("state") != "done":
                stages[earlier] = {**entry, "state": "done", "progress": 1.0, "updated": now}
    previous = stages.get(stage_id) or {}
    entry = {
        **previous,
        "state": state,
        "progress": None if progress is None else max(0.0, min(1.0, float(progress))),
        "detail": detail,
        "updated": now,
        **extra,
    }
    entry.setdefault("started", now)
    if state == "running" and previous.get("state") != "running":
        entry["started"] = now
    stages[stage_id] = entry
    write_json(path, {"stages": stages, "current": stage_id, "updated": now})


def end_current(run_dir: Path, state: str, detail: str) -> None:
    payload = read_json(stage_path(run_dir), {}) or {}
    stages = payload.get("stages") if isinstance(payload.get("stages"), dict) else {}
    running = [stage_id for stage_id in ORDER if (stages.get(stage_id) or {}).get("state") == "running"]
    target = running[-1] if running else payload.get("current") or "load"
    entry = stages.get(target) or {}
    set_stage(run_dir, target, state, entry.get("progress"), detail)


_PERCENT = re.compile(r"(\d+(?:\.\d+)?)%")


def estimate_remaining(run_dir: Path, stage_id: str, progress: float | None) -> float | None:
    """Seconds left, from elapsed time and a real 0–1 progress. No progress, no guess."""
    if progress is None or progress <= 0.02 or progress >= 1:
        return None
    payload = read_json(stage_path(run_dir), {}) or {}
    stages = payload.get("stages") if isinstance(payload.get("stages"), dict) else {}
    started = (stages.get(stage_id) or {}).get("started")
    if not isinstance(started, (int, float)):
        return None
    elapsed = time.time() - float(started)
    if elapsed < 2:
        return None
    return max(0.0, elapsed * (1.0 - progress) / progress)


def _last_percent(text: str) -> float | None:
    found = list(_PERCENT.finditer(text))
    if not found:
        return None
    value = float(found[-1].group(1))
    if value > 100:
        return None
    return value / 100


def log_progress(run_dir: Path, stage_id: str) -> float | None:
    """Read the latest tqdm percent for a stage that does not report one itself."""
    if stage_id == "load":
        path, marker, end = run_dir / "train.log", "Loading weights", None
    elif stage_id == "merge":
        path, marker, end = run_dir / "export.log", "Loading weights", "Converting to GGUF"
    elif stage_id == "gguf":
        path, marker, end = run_dir / "export.log", "Converting to GGUF", None
    else:
        return None
    if not path.exists():
        return None
    text = path.read_bytes()[-40000:].decode("utf-8", "replace")
    if stage_id == "load":
        begun = max(text.rfind("===== 开始训练 ====="), text.rfind("===== 继续训练 ====="))
    else:
        begun = text.rfind("===== 导出 Ollama =====")
    if begun >= 0:
        text = text[begun:]
    start = text.rfind(marker)
    if start < 0:
        return None
    chunk = text[start:]
    if end:
        stop = chunk.find(end)
        if stop > 0:
            chunk = chunk[:stop]
    return _last_percent(chunk)


def reset_stages(run_dir: Path, stage_ids: tuple[str, ...]) -> None:
    path = stage_path(run_dir)
    payload = read_json(path, {}) or {}
    stages = payload.get("stages") if isinstance(payload.get("stages"), dict) else {}
    for stage_id in stage_ids:
        stages.pop(stage_id, None)
    current = payload.get("current")
    if current in stage_ids:
        current = next((stage_id for stage_id in reversed(ORDER) if stage_id not in stage_ids and stage_id in stages), None)
    write_json(path, {**payload, "stages": stages, "current": current})


def reset_export(run_dir: Path) -> None:
    reset_stages(run_dir, EXPORT_STAGES)


def reset_train(run_dir: Path) -> None:
    reset_stages(run_dir, ("load", "train", "save"))


def view(run_dir: Path, alive: bool) -> list[dict]:
    payload = read_json(stage_path(run_dir), {}) or {}
    stages = payload.get("stages") if isinstance(payload.get("stages"), dict) else {}
    rows = []
    for stage_id, label in STAGES:
        entry = stages.get(stage_id) or {}
        state = entry.get("state") or "pending"
        detail = str(entry.get("detail") or "")
        progress = entry.get("progress")
        eta = entry.get("eta") if state == "running" else None
        if state == "running" and not isinstance(progress, (int, float)):
            progress = log_progress(run_dir, stage_id)
            if isinstance(progress, float):
                eta = estimate_remaining(run_dir, stage_id, progress)
        if state == "running" and not alive:
            state = "stopped"
            detail = detail or "进程已经不在"
            eta = None
        rows.append(
            {
                "id": stage_id,
                "label": label,
                "state": state,
                "progress": progress,
                "detail": detail,
                "started": entry.get("started"),
                "updated": entry.get("updated"),
                "eta": eta,
                "step": entry.get("step"),
                "total": entry.get("total"),
            }
        )
    return rows
