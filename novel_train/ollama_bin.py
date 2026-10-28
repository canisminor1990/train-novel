"""Find the Ollama CLI and talk to the local Ollama server. Each project has its own model name."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

THINK_SUFFIX = "-think"
OLLAMA_URL = "http://127.0.0.1:11434"
_cache: dict = {"at": 0.0, "names": set()}


def ollama_executable() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found
    local = os.environ.get("LOCALAPPDATA", "")
    candidates = []
    if local:
        candidates.append(Path(local) / "Programs" / "Ollama" / "ollama.exe")
    candidates.append(Path(r"C:\Program Files\Ollama\ollama.exe"))
    for path in candidates:
        if path.is_file():
            return str(path)
    return None


def ollama_models(max_age: float = 0.0) -> set[str]:
    now = time.time()
    if max_age and now - float(_cache["at"]) < max_age:
        return set(_cache["names"])
    exe = ollama_executable()
    names: set[str] = set()
    if exe is not None:
        try:
            result = subprocess.run(
                [exe, "list"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=3,
            )
        except (OSError, subprocess.TimeoutExpired):
            result = None
        for line in ((result.stdout if result else "") or "").splitlines()[1:]:
            parts = line.split()
            if parts:
                names.add(parts[0])
    _cache.update(at=now, names=names)
    return names


MODEL_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{1,62}$")


def clean_model_name(name: str) -> str:
    """Ollama model names: lowercase letters, digits, '.', '_' and '-'. No tag."""
    cleaned = str(name or "").strip().lower()
    if cleaned.endswith(":latest"):
        cleaned = cleaned[: -len(":latest")]
    if not MODEL_PATTERN.fullmatch(cleaned):
        raise ValueError("Ollama 模型名只能用小写字母、数字、点、下划线和短横线，2 到 63 个字符，以字母或数字开头。")
    return cleaned


def think_model_name(model: str) -> str:
    """The sibling Ollama model that keeps Qwen's thinking trace."""
    name = f"{model}{THINK_SUFFIX}"
    if not MODEL_PATTERN.fullmatch(name):
        raise ValueError("模型名加上 -think 后不能超过 63 个字符。请把名字缩短到 57 个字符以内。")
    return name


def _cli(*args: str, timeout: float = 60) -> tuple[bool, str]:
    exe = ollama_executable()
    if exe is None:
        return False, "没有找到 ollama.exe。"
    try:
        result = subprocess.run(
            [exe, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    forget_cache()
    return result.returncode == 0, (result.stderr or result.stdout or "").strip()


def copy_model(source: str, target: str) -> tuple[bool, str]:
    return _cli("cp", source, target)


def remove_model(name: str) -> tuple[bool, str]:
    return _cli("rm", name)


def model_ready(name: str, max_age: float = 0.0) -> bool:
    return any(item == name or item.startswith(name + ":") for item in ollama_models(max_age))


def forget_cache() -> None:
    _cache["at"] = 0.0


def chat(model: str, context: str, temperature: float = 0.8) -> str:
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "user", "content": f"前情：\n{context}"},
        ],
        "options": {"temperature": temperature},
    }
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Ollama 返回 {exc.code}。确认 {model} 还在 ollama list 里。") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("连不上本机 Ollama。先打开 Ollama，再试一次。") from exc
    return str((data.get("message") or {}).get("content") or "").strip()
