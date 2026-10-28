"""Append trainer logs and record whether a run is alive."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


def write_status(output_dir: Path, state: str, detail: str = "") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "state": state,
        "pid": os.getpid(),
        "detail": detail,
        "updated": time.time(),
    }
    (output_dir / "run_status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def attach_jsonl_callback(trainer, path: Path) -> None:
    from transformers import TrainerCallback

    class JsonlLogCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            row = {"step": int(state.global_step), "epoch": state.epoch, "time": time.time()}
            for key, value in logs.items():
                if isinstance(value, (int, float)) and key != "epoch":
                    row[key] = value
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    trainer.add_callback(JsonlLogCallback())
