"""Load project .env before any Hugging Face import."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _dir_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def writable_runtime() -> None:
    """Python on this machine cannot create files in the user temp dir or ~/.unsloth."""
    cache = ROOT / ".cache"
    tmp = cache / "tmp"
    if not _dir_writable(Path(os.environ.get("TEMP") or os.environ.get("TMP") or tmp)):
        tmp.mkdir(parents=True, exist_ok=True)
        os.environ["TEMP"] = str(tmp)
        os.environ["TMP"] = str(tmp)
    os.environ.setdefault("UNSLOTH_LLAMA_CPP_PATH", str(cache / "llama.cpp"))
    os.environ.setdefault("UNSLOTH_LLAMA_CPP_CONVERTER_CACHE", str(cache / "llama.cpp-converter"))
    # Unsloth 扫描这份官方 llama.cpp 转换脚本时，ast.walk 会抛 too many values to unpack。
    os.environ.setdefault("UNSLOTH_DISABLE_CONVERTER_SCAN", "1")


def load_project_env() -> None:
    writable_runtime()
    triton_cache = ROOT / ".cache" / "triton"
    triton_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRITON_CACHE_DIR", str(triton_cache))
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def configure_stdio() -> None:
    import sys

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except Exception:
            pass
