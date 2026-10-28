"""HTTP server for the local training dashboard. Binds to 127.0.0.1 only."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import socket
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable
from urllib.parse import unquote, urlparse

from novel_train import jobs, projects
from novel_train.dashboard import api
from novel_train.dashboard.multipart import parse_multipart
from novel_train.envload import ROOT, configure_stdio
from novel_train.jsonio import read_json
from novel_train.procs import pid_alive

DIST = ROOT / "web" / "dist"
HOST = "127.0.0.1"
PORT = 8765
MAX_BODY = 512 * 1024 * 1024
MISSING_BUILD = "看板还没有构建。在 web 目录运行 pnpm install 和 pnpm build。".encode("utf-8")

Route = tuple[str, re.Pattern, Callable]
P = r"(?P<project>[a-z0-9-]+)"
R = r"(?P<run>r-[0-9-]+)"


def _project(match: re.Match) -> projects.Project:
    return projects.get(match.group("project"))


ROUTES: list[Route] = [
    ("GET", re.compile(r"^/api/state$"), lambda m, req: api.state()),
    ("POST", re.compile(r"^/api/check$"), lambda m, req: api.check_env()),
    ("POST", re.compile(r"^/api/projects$"), lambda m, req: api.create_project(req.json())),
    ("POST", re.compile(r"^/api/models/check$"), lambda m, req: api.check_model(req.json())),
    ("GET", re.compile(rf"^/api/projects/{P}$"), lambda m, req: api.project_detail(_project(m))),
    ("POST", re.compile(rf"^/api/projects/{P}/settings$"), lambda m, req: api.update_project(_project(m), req.json())),
    ("POST", re.compile(rf"^/api/projects/{P}/delete$"), lambda m, req: api.delete_project(_project(m), req.json())),
    ("POST", re.compile(rf"^/api/projects/{P}/preset$"), lambda m, req: {"ok": True, "project": projects.set_preset(_project(m), str(req.json().get("preset") or ""))}),
    ("POST", re.compile(rf"^/api/projects/{P}/import$"), lambda m, req: api.import_books(_project(m), req.files())),
    ("POST", re.compile(rf"^/api/projects/{P}/files/delete$"), lambda m, req: api.remove_book(_project(m), req.json())),
    ("POST", re.compile(rf"^/api/projects/{P}/train$"), lambda m, req: jobs.start_training(_project(m))),
    ("POST", re.compile(rf"^/api/projects/{P}/chat$"), lambda m, req: api.continue_text(_project(m), req.json())),
    ("GET", re.compile(rf"^/api/projects/{P}/runs/{R}$"), lambda m, req: api.run_detail(_project(m), m.group("run"))),
    ("POST", re.compile(rf"^/api/projects/{P}/runs/{R}/stop$"), lambda m, req: jobs.stop(_project(m), _project(m).run_dir(m.group("run")))),
    ("POST", re.compile(rf"^/api/projects/{P}/runs/{R}/resume$"), lambda m, req: jobs.resume_training(_project(m), _project(m).run_dir(m.group("run")))),
    ("POST", re.compile(rf"^/api/projects/{P}/runs/{R}/export$"), lambda m, req: jobs.start_export(_project(m), _project(m).run_dir(m.group("run")), force=bool(req.json().get("force")))),
    ("POST", re.compile(rf"^/api/projects/{P}/runs/{R}/restart$"), lambda m, req: jobs.restart_training(_project(m), _project(m).run_dir(m.group("run")))),
    ("POST", re.compile(rf"^/api/projects/{P}/runs/{R}/delete$"), lambda m, req: api.delete_run(_project(m), m.group("run"))),
]


class Request:
    def __init__(self, content_type: str, body: bytes):
        self.content_type = content_type
        self.body = body

    def json(self) -> dict:
        payload = json.loads(self.body.decode("utf-8") or "{}")
        if not isinstance(payload, dict):
            raise ValueError("请求格式不对。")
        return payload

    def files(self) -> list[tuple[str, bytes]]:
        if not self.content_type.startswith("multipart/form-data"):
            raise ValueError("导入需要用表单上传文件。")
        _, files = parse_multipart(self.content_type, self.body)
        return files


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path.startswith("/api/"):
            self._dispatch("GET", path, Request("", b""))
            return
        self._static(path)

    def do_POST(self) -> None:
        path = unquote(urlparse(self.path).path)
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length < 0 or length > MAX_BODY:
            self._json(413, {"ok": False, "error": "上传超过 512MB。"})
            return
        body = self.rfile.read(length) if length else b""
        self._dispatch("POST", path, Request(self.headers.get("Content-Type", ""), body))

    def _dispatch(self, method: str, path: str, request: Request) -> None:
        allowed = False
        for route_method, pattern, handler in ROUTES:
            match = pattern.match(path)
            if not match:
                continue
            if route_method != method:
                allowed = True
                continue
            try:
                payload = handler(match, request)
                code = 200 if payload.get("ok", True) else 422
                self._json(code, payload)
            except projects.NotFound as exc:
                self._json(404, {"ok": False, "error": str(exc)})
            except jobs.Conflict as exc:
                self._json(409, {"ok": False, "error": str(exc)})
            except subprocess.TimeoutExpired:
                self._json(400, {"ok": False, "error": "环境检查超过 3 分钟，已中止。"})
            except (ValueError, json.JSONDecodeError, OSError, RuntimeError, TimeoutError) as exc:
                self._json(400, {"ok": False, "error": str(exc)})
            return
        self._json(405 if allowed else 404, {"ok": False, "error": "未找到"})

    def _static(self, url_path: str) -> None:
        dist = DIST.resolve()
        index = dist / "index.html"
        if not index.exists():
            self._send(503, "text/plain; charset=utf-8", MISSING_BUILD)
            return
        candidate = (dist / url_path.lstrip("/")).resolve()
        if candidate == dist or dist not in candidate.parents or not candidate.is_file():
            candidate = index
        mime = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        if candidate.suffix == ".js":
            mime = "text/javascript"
        if mime.startswith("text/"):
            mime += "; charset=utf-8"
        self._send(200, mime, candidate.read_bytes())

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def _send(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        return


def _legacy_busy() -> bool:
    status = read_json(ROOT / "outputs" / "sft" / "run_status.json", {}) or {}
    return status.get("state") == "running" and pid_alive(status.get("pid"))


def _remount() -> None:
    """Re-register each project's mounted GGUF if Ollama lost it, without touching the GPU."""
    from novel_train.export_ollama import mount
    from novel_train.ollama_bin import model_ready, ollama_executable, think_model_name

    if ollama_executable() is None or jobs.active() is not None:
        return
    for project in projects.list_projects():
        run_id = project.meta().get("mounted_run")
        if not run_id:
            continue
        try:
            run_dir = project.run_dir(run_id)
            model = project.ollama_model
            if not model_ready(model) or not model_ready(think_model_name(model)):
                mount(run_dir, model, force=True)
        except Exception as exc:
            print(f"重新挂载 {project.ollama_model} 失败：{exc}", flush=True)


class ExclusiveServer(ThreadingHTTPServer):
    """On Windows SO_REUSEADDR lets a second dashboard bind the same port silently."""

    allow_reuse_address = os.name != "nt"

    def server_bind(self) -> None:
        if os.name == "nt":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    parser = argparse.ArgumentParser(description="打开本机训练可视化后台。")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args(argv)
    migrated = projects.migrate_legacy(active=_legacy_busy())
    if migrated is not None:
        print(f"旧目录已迁移到 projects/{migrated.id}", flush=True)
    try:
        server = ExclusiveServer((HOST, args.port), Handler)
    except OSError:
        print(f"端口 {args.port} 已被占用。训练台可能已经开着：http://{HOST}:{args.port}", flush=True)
        return 1
    threading.Thread(target=_remount, daemon=True).start()
    print(f"训练后台 http://{HOST}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    return 0
