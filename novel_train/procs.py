"""Windows process helpers. Children are detached so a dashboard restart does not kill training."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from novel_train.envload import ROOT, writable_runtime

DETACHED = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)


def child_env() -> dict:
    writable_runtime()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def pid_alive(pid) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    result = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
        capture_output=True,
        text=True,
        check=False,
    )
    return f'"{pid}"' in result.stdout


def kill_tree(pid: int) -> tuple[bool, str]:
    result = subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0, (result.stderr or result.stdout or "").strip()


def spawn_module(
    module: str,
    args: list[str],
    log_path: Path,
    banner: str,
    env_extra: dict | None = None,
) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = child_env()
    if env_extra:
        env.update(env_extra)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n===== {banner} =====\n")
        handle.flush()
        return subprocess.Popen(
            [sys.executable, "-m", module, *args],
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=DETACHED,
        )


def run_module(module: str, args: list[str], log_path: Path, banner: str) -> int:
    return spawn_module(module, args, log_path, banner).wait()
