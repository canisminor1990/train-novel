"""Small checks shared by the training entry points."""

from __future__ import annotations

import json
from pathlib import Path


def _is_sample_name(name: str) -> bool:
    return name.startswith("示例") or name.lower().startswith("example")


def sample_only_reason(path: Path) -> str | None:
    if not path.exists():
        return None
    books: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        books.add(str(json.loads(line).get("book", "")))
    if not books:
        return "训练集是空的。先导入小说并切分。"
    if all(_is_sample_name(name) for name in books):
        return "训练集里只有示例小说。把你自己的 txt 或 epub 导入这个项目，然后重新切分。"
    return None


def reject_sample_only(path: Path) -> None:
    reason = sample_only_reason(path)
    if reason:
        raise SystemExit(reason)
