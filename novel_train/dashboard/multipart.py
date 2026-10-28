"""Minimal multipart/form-data parser for the novel upload form."""

from __future__ import annotations

import re


def parse_multipart(content_type: str, body: bytes) -> tuple[dict[str, str], list[tuple[str, bytes]]]:
    match = re.search(r"boundary=(?P<boundary>[^;]+)", content_type)
    if not match:
        raise ValueError("缺少上传边界")
    boundary = match.group("boundary").strip().strip('"').encode("utf-8")
    fields: dict[str, str] = {}
    files: list[tuple[str, bytes]] = []
    for part in body.split(b"--" + boundary):
        if part in (b"", b"--", b"--\r\n") or part.startswith(b"--"):
            continue
        if part.startswith(b"\r\n"):
            part = part[2:]
        if part.endswith(b"\r\n"):
            part = part[:-2]
        header_blob, separator, content = part.partition(b"\r\n\r\n")
        if not separator:
            continue
        headers = header_blob.decode("utf-8", errors="replace")
        disposition = ""
        for line in headers.split("\r\n"):
            if line.lower().startswith("content-disposition:"):
                disposition = line.split(":", 1)[1]
        name_match = re.search(r'name="([^"]*)"', disposition)
        file_match = re.search(r'filename="([^"]*)"', disposition)
        name = name_match.group(1) if name_match else ""
        if file_match and file_match.group(1):
            files.append((file_match.group(1), content))
        else:
            fields[name] = content.decode("utf-8", errors="replace")
    return fields, files
