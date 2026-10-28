"""Machine-level status: the GPU and the one-off environment check."""

from __future__ import annotations

import subprocess
import sys
import time

from novel_train.envload import ROOT
from novel_train.jsonio import read_json, write_json
from novel_train.procs import child_env

CHECK_PATH = ROOT / "outputs" / "check.json"
_gpu_cache: dict = {"at": 0.0, "value": None}


def gpu_status() -> dict | None:
    now = time.time()
    if now - float(_gpu_cache["at"]) < 2:
        return _gpu_cache["value"]
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        result = None
    value = None
    if result is not None and result.returncode == 0 and result.stdout.strip():
        name, used, total, util, temp = [part.strip() for part in result.stdout.strip().splitlines()[0].split(",")]
        value = {
            "name": name,
            "used_mib": int(float(used)),
            "total_mib": int(float(total)),
            "util": int(float(util)),
            "temp_c": int(float(temp)),
        }
    _gpu_cache.update(at=now, value=value)
    return value


def last_check() -> dict | None:
    return read_json(CHECK_PATH, None)


def parse_check(log: str) -> dict:
    fields: dict = {}
    for line in log.splitlines():
        key, _, value = line.partition(" ")
        if key in {"torch", "gpu", "vram_free_mib", "unsloth"}:
            fields["vram" if key == "vram_free_mib" else key] = value
        elif key in {"cuda", "bf16"}:
            fields[key] = value == "True"
    return fields


def run_check() -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "novel_train.check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
        env=child_env(),
    )
    log = ((result.stdout or "") + (result.stderr or "")).strip()
    payload = {
        "ok": result.returncode == 0,
        "code": result.returncode,
        "log": log,
        "fields": parse_check(log),
        "updated": time.time(),
    }
    write_json(CHECK_PATH, payload)
    return payload
