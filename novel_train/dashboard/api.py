"""JSON handlers behind the dashboard routes."""

from __future__ import annotations

from novel_train import jobs, projects, runs
from novel_train.dashboard import system
from novel_train.ollama_bin import (
    chat as ollama_chat,
    copy_model,
    model_ready,
    ollama_executable,
    remove_model,
    think_model_name,
)
from novel_train.presets import public_presets
from novel_train.projects import Project

CONTEXT_MIN = 20
# 约 4.6 万 token，给 64K 上下文的正文模型留出续写的余量。
CONTEXT_MAX = 60000


def _lock() -> dict | None:
    return jobs.active()


def _active_run(project: Project, lock: dict | None) -> str | None:
    if lock and lock.get("project") == project.id:
        return str(lock.get("run"))
    return None


def _ollama(project: Project) -> dict:
    installed = ollama_executable() is not None
    model = project.ollama_model
    think = think_model_name(model)
    return {
        "installed": installed,
        "model": model,
        "think_model": think,
        "ready": installed and model_ready(model, max_age=5),
        "think_ready": installed and model_ready(think, max_age=5),
    }


def project_summary(project: Project, lock: dict | None) -> dict:
    meta = project.meta()
    ready, reason = projects.readiness(project)
    stats = projects.dataset(project) or {}
    run_dirs = runs.list_run_dirs(project)
    latest = runs.summary(project, run_dirs[0], _active_run(project, lock), meta.get("mounted_run")) if run_dirs else None
    return {
        "id": project.id,
        "name": meta.get("name") or project.id,
        "created": meta.get("created"),
        "updated": meta.get("updated"),
        "preset": meta.get("preset"),
        "ready": ready,
        "ready_detail": reason,
        "books": len(stats.get("books") or []),
        "book_names": [row.get("book") for row in stats.get("books") or []],
        "train_samples": stats.get("train_samples") or 0,
        "run_count": len(run_dirs),
        "latest_run": latest,
        "mounted_run": meta.get("mounted_run"),
        "ollama": _ollama(project),
    }


def state() -> dict:
    lock = _lock()
    return {
        "gpu": system.gpu_status(),
        "active": lock,
        "busy": jobs.busy_reason(lock) if lock else None,
        "check": system.last_check(),
        "projects": [project_summary(project, lock) for project in projects.list_projects()],
    }


def project_detail(project: Project) -> dict:
    lock = _lock()
    meta = project.meta()
    summary = project_summary(project, lock)
    active_run = _active_run(project, lock)
    return {
        **summary,
        "dataset": projects.dataset(project),
        "raw_files": projects.raw_files(project),
        "samples": projects.sample_preview(project),
        "presets": public_presets(meta.get("preset"), summary["train_samples"]),
        "runs": [runs.summary(project, run_dir, active_run, meta.get("mounted_run")) for run_dir in runs.list_run_dirs(project)],
        "busy": jobs.busy_reason(lock) if lock else None,
        "active": lock,
    }


def run_detail(project: Project, run_id: str) -> dict:
    lock = _lock()
    run_dir = project.run_dir(run_id)
    meta = project.meta()
    return {
        "project": {"id": project.id, "name": meta.get("name"), "ollama": _ollama(project)},
        "run": runs.detail(project, run_dir, _active_run(project, lock), meta.get("mounted_run")),
        "busy": jobs.busy_reason(lock) if lock else None,
    }


def create_project(payload: dict) -> dict:
    model = str(payload.get("ollama_model") or "").strip() or None
    project = projects.create(str(payload.get("name") or ""), ollama_model=model)
    warning = None
    think = think_model_name(project.ollama_model)
    if model_ready(project.ollama_model) or model_ready(think):
        warning = f"Ollama 里已经有 {project.ollama_model} 或 {think}，这个项目训完挂载时会覆盖它们。"
    return {"ok": True, "project": project_summary(project, _lock()), "warning": warning}


def check_model(payload: dict) -> dict:
    """Live validation for the model-name field."""
    exclude = str(payload.get("project") or "") or None
    try:
        model = projects.check_model(str(payload.get("ollama_model") or ""), exclude)
    except ValueError as exc:
        return {"ok": True, "valid": False, "error": str(exc)}
    think = think_model_name(model)
    exists = model_ready(model, max_age=5) or model_ready(think, max_age=5)
    return {"ok": True, "valid": True, "model": model, "think_model": think, "exists": exists}


def update_project(project: Project, payload: dict) -> dict:
    fields: dict = {}
    if "name" in payload:
        projects.rename(project, str(payload.get("name") or ""))
    note = None
    if payload.get("ollama_model") is not None:
        old = project.ollama_model
        new = projects.check_model(str(payload["ollama_model"]), exclude=project.id)
        if new != old:
            _require_idle(project, "改模型名")
            pairs = [
                (source, target)
                for source, target in ((old, new), (think_model_name(old), think_model_name(new)))
                if model_ready(source)
            ]
            for source, target in pairs:
                ok, message = copy_model(source, target)
                if not ok:
                    raise ValueError(f"没能在 Ollama 里改名：{message}")
            for source, _target in pairs:
                remove_model(source)
            if pairs:
                renamed = "、".join(source for source, _target in pairs)
                created = "、".join(target for _source, target in pairs)
                note = f"Ollama 里的 {renamed} 已改名为 {created}。"
            fields["ollama_model"] = new
    if fields:
        project.update(**fields)
    return {"ok": True, "project": project_summary(project, _lock()), "note": note}


def _require_idle(project: Project, action: str) -> None:
    lock = _lock()
    if lock and lock.get("project") == project.id:
        raise jobs.Conflict(f"这个项目正在{'训练' if lock.get('kind') == 'train' else '导出'}。等它结束再{action}。")


def delete_project(project: Project, payload: dict) -> dict:
    _require_idle(project, "删除")
    remove_files = bool(payload.get("remove_files", True))
    remove_models = bool(payload.get("remove_models", True))
    if not remove_files and not remove_models:
        raise ValueError("至少选择一项。")
    note = None
    if remove_models:
        model = project.ollama_model
        present = [name for name in (model, think_model_name(model)) if model_ready(name)]
        removed: list[str] = []
        failed: list[str] = []
        for name in present:
            ok, message = remove_model(name)
            if ok:
                removed.append(name)
            else:
                failed.append(f"{name}（{message}）")
        if removed:
            note = f"已从 Ollama 删除 {'、'.join(removed)}。"
        if failed:
            missed = f"Ollama 里的 {'、'.join(failed)} 没删掉。"
            note = f"{note}{missed}" if note else missed
    if remove_files:
        projects.delete(project)
    return {"ok": True, "deleted": remove_files, "note": note}


def import_books(project: Project, files: list[tuple[str, bytes]]) -> dict:
    saved: list[str] = []
    errors: list[str] = []
    for filename, data in files:
        try:
            saved.append(projects.save_upload(project, filename, data))
        except ValueError as exc:
            errors.append(str(exc))
    if not saved:
        raise ValueError("；".join(errors) or "没有可导入的小说。")
    result = projects.prepare(project)
    return {**result, "saved": saved, "errors": errors}


def remove_book(project: Project, payload: dict) -> dict:
    projects.remove_file(project, str(payload.get("name") or ""))
    return projects.prepare(project)


def delete_run(project: Project, run_id: str) -> dict:
    run_dir = project.run_dir(run_id)
    if _active_run(project, _lock()) == run_id:
        raise jobs.Conflict("这次运行还在跑。先停止，再删除。")
    runs.delete_run(run_dir)
    if project.meta().get("mounted_run") == run_id:
        project.update(mounted_run=None)
    return {"ok": True}


def continue_text(project: Project, payload: dict) -> dict:
    body = str(payload.get("text") or "").strip()
    if len(body) < CONTEXT_MIN:
        raise ValueError(f"前情至少 {CONTEXT_MIN} 个字。")
    if len(body) > CONTEXT_MAX:
        raise ValueError(f"前情超过 {CONTEXT_MAX} 字。先截到最近的一段。")
    reason = jobs.busy_reason()
    if reason:
        raise jobs.Conflict(reason + "等它结束再续写，避免抢显存。")
    model = project.ollama_model
    if ollama_executable() is None:
        raise ValueError("没有找到 ollama.exe。")
    if not model_ready(model):
        raise ValueError(f"{model} 还没挂上。训练结束会自动挂载，或在运行详情里点「继续」。")
    return {"ok": True, "text": ollama_chat(model, body), "model": model}


def check_env() -> dict:
    reason = jobs.busy_reason()
    if reason:
        raise jobs.Conflict(reason + "等它结束再检查环境。")
    return system.run_check()
