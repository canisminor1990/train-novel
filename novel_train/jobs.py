"""One GPU job at a time: train a run, then export it to Ollama.

The dashboard starts `python -m novel_train.jobs --project P --run R` as a detached process.
That process owns the whole chain, so restarting the dashboard does not drop the export step.
"""

from __future__ import annotations

import argparse
import os
import secrets
import shutil
import sys
import time
from pathlib import Path

from novel_train.envload import ROOT, configure_stdio
from novel_train.ollama_bin import think_model_name
from novel_train.jsonio import read_json, write_json
from novel_train.export_ollama import build_dir, export_plan
from novel_train.pipeline import end_current, reset_export, reset_train, set_stage, view as stage_view
from novel_train.procs import kill_tree, pid_alive, run_module, spawn_module
from novel_train.projects import PROJECTS, Project, get as get_project
from novel_train.runs import create_run, latest_checkpoint, run_meta, update_run

LOCK = PROJECTS / ".active.json"


class Conflict(ValueError):
    pass


def active() -> dict | None:
    lock = read_json(LOCK, None)
    if not isinstance(lock, dict):
        return None
    if pid_alive(lock.get("pid")):
        return lock
    # The dashboard writes the lock before the job process publishes its own pid.
    pending = lock.get("pending")
    if lock.get("pid") is None and isinstance(pending, (int, float)) and time.time() - pending < 20:
        return lock
    return None


def _claim(pid: int, project: str, run: str, kind: str, token: str | None = None) -> None:
    payload = {"pid": pid, "project": project, "run": run, "kind": kind, "started": time.time()}
    if token:
        payload["token"] = token
    write_json(LOCK, payload)


def _release(pid: int) -> None:
    lock = read_json(LOCK, None)
    if isinstance(lock, dict) and lock.get("pid") == pid:
        LOCK.unlink(missing_ok=True)


def busy_reason(lock: dict | None = None) -> str | None:
    lock = lock or active()
    if lock is None:
        return None
    try:
        name = get_project(lock["project"]).meta().get("name")
    except LookupError:
        name = lock.get("project")
    what = KIND_LABELS.get(str(lock.get("kind")), "导出")
    return f"「{name}」正在{what}，显卡同一时间只跑一个任务。"


KIND_LABELS = {"train": "训练", "export": "导出", "premise": "写命题"}


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _spawn(project: Project, run_dir: Path, kind: str, extra: list[str]) -> int:
    return _spawn_job(
        project,
        run_dir.name,
        kind,
        "novel_train.jobs",
        ["--project", project.id, "--run", run_dir.name, *extra],
        run_dir / "job.log",
    )


def _spawn_job(project: Project, run: str, kind: str, module: str, args: list[str], log_path: Path) -> int:
    # venv's python.exe on Windows is a launcher. Popen.pid is that launcher,
    # while the job's os.getpid() is its child. A token lets the child replace
    # the lock with its own pid; recording the launcher pid makes the child
    # think another job already holds the GPU, so it exits before export starts.
    token = secrets.token_hex(8)
    write_json(
        LOCK,
        {
            "pid": None,
            "token": token,
            "project": project.id,
            "run": run,
            "kind": kind,
            "pending": time.time(),
            "started": time.time(),
        },
    )
    try:
        process = spawn_module(
            module,
            args,
            log_path,
            f"任务 {kind} {time.strftime('%Y-%m-%d %H:%M:%S')}",
            {"NOVEL_JOB_TOKEN": token},
        )
    except Exception:
        current = read_json(LOCK, None)
        if isinstance(current, dict) and current.get("token") == token:
            LOCK.unlink(missing_ok=True)
        raise
    return process.pid


def _adopt() -> bool:
    token = os.environ.get("NOVEL_JOB_TOKEN")
    if not token:
        return False
    lock = read_json(LOCK, None)
    if not isinstance(lock, dict) or lock.get("token") != token:
        return False
    _claim(os.getpid(), str(lock.get("project")), str(lock.get("run")), str(lock.get("kind") or "train"), token)
    return True


def hold_gpu(project: str, run: str, kind: str) -> int | None:
    """For job processes that are not novel_train.jobs. Returns this pid, or None when busy."""
    pid = os.getpid()
    if not _adopt():
        lock = active()
        if lock is not None and lock.get("pid") != pid:
            return None
        _claim(pid, project, run, kind)
    return pid


def release_gpu(pid: int) -> None:
    _release(pid)


# ---------- dashboard side ----------

def start_training(project: Project) -> dict:
    reason = busy_reason()
    if reason:
        raise Conflict(reason)
    run_dir = create_run(project)
    pid = _spawn(project, run_dir, "train", [])
    update_run(run_dir, pid=pid, phase="train")
    return {"ok": True, "run": run_dir.name, "pid": pid}


def resume_training(project: Project, run_dir: Path) -> dict:
    reason = busy_reason()
    if reason:
        raise Conflict(reason)
    if latest_checkpoint(run_dir) is None:
        raise ValueError("这次运行没有可继续的断点。")
    meta = run_meta(run_dir)
    if meta.get("state") in {"done", "trained"}:
        raise ValueError("这次运行已经训练完成。")
    update_run(run_dir, state="queued", phase="train", detail="正在从断点继续", ended=None, queued_at=time.time())
    pid = _spawn(project, run_dir, "train", ["--resume"])
    update_run(run_dir, pid=pid)
    return {"ok": True, "run": run_dir.name, "pid": pid}


def start_export(project: Project, run_dir: Path, force: bool = False) -> dict:
    reason = busy_reason()
    if reason:
        raise Conflict(reason)
    if not (run_dir / "lora" / "adapter_model.safetensors").exists():
        raise ValueError("这次运行还没有 LoRA。")
    if force and export_plan(run_dir) != "mount":
        shutil.rmtree(build_dir(run_dir), ignore_errors=True)
    plan = export_plan(run_dir)
    if plan == "mount":
        stage_id, detail = "ollama", "正在挂到 Ollama"
    elif plan == "gguf":
        set_stage(run_dir, "merge", "done", 1, "沿用已合并的 16 位权重")
        stage_id, detail = "gguf", "正在从已合并的权重继续导出"
    else:
        stage_id, detail = "merge", "正在从保存的 LoRA 继续导出"
    update_run(run_dir, state="queued", phase="export", detail=detail, queued_at=time.time())
    set_stage(run_dir, stage_id, "running", None, detail)
    pid = _spawn(project, run_dir, "export", ["--export-only"])
    update_run(run_dir, pid=pid)
    return {"ok": True, "run": run_dir.name, "pid": pid}


def restart_training(project: Project, run_dir: Path) -> dict:
    reason = busy_reason()
    if reason:
        raise Conflict(reason)
    if (run_dir / "lora" / "adapter_model.safetensors").exists():
        raise ValueError("LoRA 已经保存。点「继续」会接着导出，不会重新训练。")
    if latest_checkpoint(run_dir) is not None:
        raise ValueError("已有断点。点「继续」会从断点接着训。")
    reset_train(run_dir)
    reset_export(run_dir)
    update_run(
        run_dir,
        state="queued",
        phase="train",
        detail="正在重新开始",
        ended=None,
        started=None,
        queued_at=time.time(),
    )
    pid = _spawn(project, run_dir, "train", [])
    update_run(run_dir, pid=pid)
    return {"ok": True, "run": run_dir.name, "pid": pid}


def stop(project: Project, run_dir: Path) -> dict:
    lock = active()
    if lock is None or lock.get("run") != run_dir.name or lock.get("project") != project.id:
        meta = run_meta(run_dir)
        if meta.get("state") in {"queued", "running"}:
            update_run(run_dir, state="interrupted", detail="进程已经不在", ended=time.time())
        raise ValueError("这次运行没有在跑。")
    ok, message = kill_tree(int(lock["pid"]))
    if not ok and pid_alive(lock["pid"]):
        raise ValueError(message or "停止失败。")
    LOCK.unlink(missing_ok=True)
    if any(stage["state"] == "running" for stage in stage_view(run_dir, alive=True)):
        end_current(run_dir, "stopped", "已从看板暂停")
    has_ckpt = latest_checkpoint(run_dir) is not None
    exporting = run_meta(run_dir).get("phase") == "export"
    if exporting:
        detail = "已暂停导出。LoRA 还在，可以继续。"
    elif has_ckpt:
        detail = "已暂停。断点还在，可以继续。"
    else:
        detail = "已暂停。还没有存下断点。"
    update_run(run_dir, state="stopped", detail=detail, ended=time.time())
    return {"ok": True}


def start_premises(project: Project) -> dict:
    from novel_train.premise import pending_passages

    reason = busy_reason()
    if reason:
        raise Conflict(reason)
    todo, total = pending_passages(project.processed)
    if not total:
        raise ValueError("还没有切出样本。先导入小说。")
    write_json(
        project.root / "premise_status.json",
        {"state": "queued", "done": total - len(todo), "total": total, "detail": "正在启动", "updated": time.time()},
    )
    pid = _spawn_job(project, "", "premise", "novel_train.premise", ["--project", project.id], project.root / "premise.log")
    return {"ok": True, "pid": pid}


def stop_premises(project: Project) -> dict:
    lock = active()
    if lock is None or lock.get("kind") != "premise" or lock.get("project") != project.id:
        raise ValueError("这个项目没有在写命题。")
    ok, message = kill_tree(int(lock["pid"]))
    if not ok and pid_alive(lock["pid"]):
        raise ValueError(message or "停止失败。")
    LOCK.unlink(missing_ok=True)
    from novel_train.projects import prepare

    result = prepare(project)
    stats = result.get("dataset") or {}
    path = project.root / "premise_status.json"
    status = read_json(path, {}) or {}
    status.update(
        state="stopped",
        detail=f"已暂停。{stats.get('with_premise', 0)} 段有命题，已经用进样本，下次接着写剩下的。",
        ended=time.time(),
        updated=time.time(),
    )
    write_json(path, status)
    return {"ok": True}


# ---------- job process ----------

def _stage_detail(run_dir: Path, stage_ids: tuple[str, ...]) -> str:
    for stage in stage_view(run_dir, alive=True):
        if stage["id"] in stage_ids and stage["state"] == "failed" and stage["detail"]:
            return stage["detail"]
    return ""


def _train(project: Project, run_dir: Path, resume: bool) -> bool:
    meta = run_meta(run_dir)
    update_run(
        run_dir,
        state="running",
        phase="train",
        pid=os.getpid(),
        started=meta.get("started") or time.time(),
        detail="正在从断点继续" if resume else "正在训练",
    )
    args = ["--config", _rel(run_dir / "config.json")]
    if resume:
        checkpoint = latest_checkpoint(run_dir)
        if checkpoint is None:
            update_run(run_dir, state="failed", detail="没有可继续的断点。", ended=time.time())
            return False
        args += ["--resume", _rel(checkpoint)]
    code = run_module("novel_train.train_sft", args, run_dir / "train.log", "继续训练" if resume else "开始训练")
    if code != 0 or not (run_dir / "lora" / "adapter_model.safetensors").exists():
        reason = _stage_detail(run_dir, ("load", "train", "save")) or f"训练进程退出，代码 {code}。看 train.log。"
        update_run(run_dir, state="failed", detail=reason, ended=time.time())
        return False
    return True


def _export(project: Project, run_dir: Path) -> None:
    plan = export_plan(run_dir)
    if plan == "mount":
        detail = "正在挂到 Ollama"
        set_stage(run_dir, "ollama", "running", None, detail)
    elif plan == "gguf":
        set_stage(run_dir, "merge", "done", 1, "沿用已合并的 16 位权重")
        detail = "正在从已合并的权重继续导出 GGUF"
        set_stage(run_dir, "gguf", "running", None, detail)
    else:
        reset_export(run_dir)
        detail = "正在合并并导出 GGUF"
        set_stage(run_dir, "merge", "running", None, "正在加载并合并 LoRA")
    update_run(run_dir, state="running", phase="export", pid=os.getpid(), detail=detail)
    code = run_module(
        "novel_train.export_ollama",
        ["--run", _rel(run_dir), "--model", project.ollama_model],
        run_dir / "export.log",
        "导出 Ollama",
    )
    if code == 0:
        meta = project.meta()
        fields = {"mounted_run": run_dir.name}
        detail = f"已挂载为 {project.ollama_model} 和 {think_model_name(project.ollama_model)}"
        if run_meta(run_dir).get("preset") == "quick" and meta.get("preset") == "quick":
            fields["preset"] = "standard"
            detail += "。流程走通了，下一次已切到「标准」"
        project.update(**fields)
        update_run(run_dir, state="done", detail=detail, ended=time.time())
        return
    reason = _stage_detail(run_dir, ("merge", "gguf", "ollama")) or f"导出进程退出，代码 {code}。看 export.log。"
    update_run(run_dir, state="trained", detail=f"LoRA 已保存，导出没有完成：{reason}", ended=time.time())


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    parser = argparse.ArgumentParser(description="跑一次训练，并把结果挂到 Ollama。")
    parser.add_argument("--project", required=True)
    parser.add_argument("--run", help="已有的运行。不给就按项目当前预设新建一次。")
    parser.add_argument("--resume", action="store_true", help="从这次运行最新的断点继续")
    parser.add_argument("--export-only", action="store_true", help="只导出并挂载已有的 LoRA")
    args = parser.parse_args(argv)

    project = get_project(args.project)
    pid = os.getpid()
    adopted = _adopt()
    lock = active()
    if not adopted and lock is not None and lock.get("pid") != pid:
        message = busy_reason() or "显卡同一时间只跑一个任务。"
        print(message)
        return 2
    run_dir = project.run_dir(args.run) if args.run else create_run(project)
    _claim(pid, project.id, run_dir.name, "export" if args.export_only else "train")
    print(f"项目 {project.id}  运行 {run_dir.name}", flush=True)
    try:
        if not args.export_only and not _train(project, run_dir, args.resume):
            return 1
        _claim(pid, project.id, run_dir.name, "export")
        _export(project, run_dir)
        return 0
    except Exception as exc:
        update_run(run_dir, state="failed", detail=str(exc), ended=time.time())
        raise
    finally:
        _release(pid)


if __name__ == "__main__":
    sys.exit(main())
