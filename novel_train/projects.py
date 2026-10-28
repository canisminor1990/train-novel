"""Projects: one book (or one author's books) with its own raw text, samples, runs and Ollama model."""

from __future__ import annotations

import re
import shutil
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

from novel_train.envload import ROOT
from novel_train.jsonio import read_json, write_json
from novel_train.models import DEFAULT_MODEL, MODELS
from novel_train.presets import DEFAULT_PRESET, PRESETS

PROJECTS = ROOT / "projects"
MULTI_BOOK_CAP = 2000
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{2,40}$")


class NotFound(LookupError):
    pass


@dataclass(frozen=True)
class Project:
    id: str

    @property
    def root(self) -> Path:
        return PROJECTS / self.id

    @property
    def meta_path(self) -> Path:
        return self.root / "project.json"

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    @property
    def stats_path(self) -> Path:
        return self.processed / "stats.json"

    @property
    def train_file(self) -> Path:
        return self.processed / "sft" / "train.jsonl"

    @property
    def eval_file(self) -> Path:
        return self.processed / "sft" / "eval.jsonl"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    def run_dir(self, run_id: str) -> Path:
        if not re.fullmatch(r"r-[0-9]{8}-[0-9]{6}(-[0-9]+)?", run_id or ""):
            raise NotFound("没有这次运行。")
        path = self.runs / run_id
        if not path.is_dir():
            raise NotFound("没有这次运行。")
        return path

    def meta(self) -> dict:
        payload = read_json(self.meta_path, None)
        if not isinstance(payload, dict):
            raise NotFound("没有这个项目。")
        return payload

    def update(self, **fields) -> dict:
        payload = self.meta()
        payload.update(fields)
        payload["updated"] = time.time()
        write_json(self.meta_path, payload)
        return payload

    @property
    def ollama_model(self) -> str:
        return str(self.meta().get("ollama_model") or f"novel-{self.id}")

    @property
    def base_model(self) -> str:
        key = self.meta().get("base_model")
        return key if key in MODELS else DEFAULT_MODEL


def get(project_id: str) -> Project:
    if not ID_PATTERN.fullmatch(project_id or ""):
        raise NotFound("没有这个项目。")
    project = Project(project_id)
    project.meta()
    return project


def list_projects() -> list[Project]:
    if not PROJECTS.exists():
        return []
    found = []
    for path in PROJECTS.iterdir():
        if path.is_dir() and ID_PATTERN.fullmatch(path.name) and (path / "project.json").exists():
            found.append(Project(path.name))
    found.sort(key=lambda project: float(project.meta().get("created") or 0), reverse=True)
    return found


def _clean_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(name or "")).strip()
    if not cleaned:
        raise ValueError("项目名不能为空。")
    if len(cleaned) > 40:
        raise ValueError("项目名最多 40 个字。")
    return cleaned


def _new_id() -> str:
    base = time.strftime("%Y%m%d-%H%M%S")
    candidate = base
    suffix = 1
    while (PROJECTS / candidate).exists():
        suffix += 1
        candidate = f"{base}-{suffix}"
    return candidate


def model_owner(model: str, exclude: str | None = None) -> Project | None:
    from novel_train.ollama_bin import think_model_name

    wanted = {model, think_model_name(model)}
    for project in list_projects():
        if project.id == exclude:
            continue
        taken = {project.ollama_model, think_model_name(project.ollama_model)}
        if wanted & taken:
            return project
    return None


def _check_model(model: str, exclude: str | None = None) -> str:
    from novel_train.ollama_bin import clean_model_name, think_model_name

    model = clean_model_name(model)
    think_model_name(model)
    owner = model_owner(model, exclude)
    if owner is not None:
        raise ValueError(
            f"「{owner.meta().get('name')}」已经在用 {owner.ollama_model}。"
            "每个项目的正文模型名和 -think 模型名都要和其他项目错开。"
        )
    return model


def _id_from_model(model: str) -> str | None:
    slug = re.sub(r"[^a-z0-9-]+", "-", model.removeprefix("novel-")).strip("-")
    if ID_PATTERN.fullmatch(slug) and not (PROJECTS / slug).exists():
        return slug
    return None


def create(name: str, project_id: str | None = None, ollama_model: str | None = None) -> Project:
    clean = _clean_name(name)
    model = _check_model(ollama_model) if ollama_model else None
    project = Project(project_id or (model and _id_from_model(model)) or _new_id())
    if project.root.exists():
        raise ValueError("这个项目已经存在。")
    if model is None:
        model = _check_model(f"novel-{project.id}")
    project.raw.mkdir(parents=True)
    project.runs.mkdir(parents=True)
    now = time.time()
    write_json(
        project.meta_path,
        {
            "id": project.id,
            "name": clean,
            "created": now,
            "updated": now,
            "preset": DEFAULT_PRESET,
            "base_model": DEFAULT_MODEL,
            "ollama_model": model,
            "mounted_run": None,
        },
    )
    return project


def rename(project: Project, name: str) -> dict:
    return project.update(name=_clean_name(name))


def check_model(model: str, exclude: str | None = None) -> str:
    return _check_model(model, exclude)


def set_preset(project: Project, preset: str) -> dict:
    if preset not in PRESETS:
        raise ValueError("预设只能是试跑、标准或细读。")
    return project.update(preset=preset)


def set_base_model(project: Project, key: str) -> dict:
    if key not in MODELS:
        raise ValueError("模型只能是 Qwen3.5-4B 或 Qwen3.5-9B。")
    return project.update(base_model=key)


def delete(project: Project) -> None:
    shutil.rmtree(project.root)


# ---------- books ----------

def _suffixes() -> set[str]:
    from novel_train.prepare import SUFFIXES

    return SUFFIXES


def safe_filename(name: str) -> str:
    cleaned = Path(name or "").name.replace("\x00", "").strip()
    suffix = Path(cleaned).suffix.lower()
    if suffix not in _suffixes():
        raise ValueError(f"{cleaned or '文件'} 不是 txt、md 或 epub。")
    stem = re.sub(r'[<>:"/\\|?*]', "_", Path(cleaned).stem).strip(" .")
    if not stem or stem in {".", ".."}:
        raise ValueError("文件名无效。")
    return stem + suffix


def raw_files(project: Project) -> list[dict]:
    if not project.raw.exists():
        return []
    suffixes = _suffixes()
    rows = []
    for path in sorted(project.raw.iterdir()):
        if path.is_file() and path.suffix.lower() in suffixes and not path.name.startswith("."):
            stat = path.stat()
            rows.append({"name": path.name, "bytes": stat.st_size, "modified": stat.st_mtime})
    return rows


def save_upload(project: Project, filename: str, data: bytes) -> str:
    name = safe_filename(filename)
    if not data:
        raise ValueError(f"{name} 是空的。")
    project.raw.mkdir(parents=True, exist_ok=True)
    (project.raw / name).write_bytes(data)
    return name


def remove_file(project: Project, filename: str) -> None:
    name = safe_filename(filename)
    path = project.raw / name
    if not path.is_file():
        raise NotFound(f"项目里没有 {name}。")
    path.unlink()


def prepare(project: Project) -> dict:
    from novel_train.prepare import build, is_sample, list_books, parse_args

    real = [path for path in list_books(project.raw) if not is_sample(path)] if project.raw.exists() else []
    args = parse_args(["--input", str(project.raw), "--output", str(project.processed)])
    args.max_samples_per_book = MULTI_BOOK_CAP if len(real) > 1 else 0
    args.preview = 0
    log = StringIO()
    if not raw_files(project):
        if project.processed.exists():
            shutil.rmtree(project.processed)
        return {"ok": True, "log": "项目里没有书了，样本已清空。", "dataset": None}
    with redirect_stdout(log):
        code = build(args)
    if code != 0 and project.processed.exists():
        shutil.rmtree(project.processed)
    return {"ok": code == 0, "log": log.getvalue().strip(), "dataset": dataset(project)}


def dataset(project: Project) -> dict | None:
    stats = read_json(project.stats_path, None)
    return stats if isinstance(stats, dict) else None


def is_sample_name(name: str) -> bool:
    return name.startswith("示例") or name.lower().startswith("example")


def readiness(project: Project) -> tuple[bool, str]:
    stats = dataset(project) or {}
    books = [row for row in stats.get("books") or [] if isinstance(row, dict)]
    names = [str(row.get("book", "")) for row in books]
    if not raw_files(project):
        return False, "还没有导入小说。"
    if not names:
        return False, "导入的书没有切出续写样本。看一下文本是否太短。"
    if all(is_sample_name(name) for name in names):
        return False, "现在只有示例小说。导入你自己的书后才能开训。"
    return True, f"{len(names)} 本，{stats.get('train_samples') or 0} 条训练样本"


def sample_preview(project: Project, limit: int = 3) -> list[dict]:
    rows = []
    path = project.train_file
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = read_jsonl_line(line)
            if row is None:
                continue
            rows.append(
                {
                    "book": row.get("book", ""),
                    "chapter": row.get("chapter", ""),
                    "input": str(row.get("input", ""))[:900],
                    "output": str(row.get("output", ""))[:600],
                    "output_chars": len(str(row.get("output", ""))),
                }
            )
            if len(rows) >= limit:
                break
    return rows


def read_jsonl_line(line: str) -> dict | None:
    import json

    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return None
    return row if isinstance(row, dict) else None


# ---------- legacy layout ----------

def migrate_legacy(active: bool) -> Project | None:
    """Move the pre-project layout (data/raw, data/processed, outputs/sft) into one project."""
    if list_projects():
        return None
    legacy_raw = ROOT / "data" / "raw"
    legacy_processed = ROOT / "data" / "processed"
    legacy_run = ROOT / "outputs" / "sft"
    has_books = legacy_raw.exists() and any(path.is_file() for path in legacy_raw.iterdir())
    has_run = (legacy_run / "lora").exists() or (legacy_run / "metrics.jsonl").exists()
    if not has_books and not has_run:
        return None
    if active:
        return None
    project = create("第一本书", project_id="legacy-sft", ollama_model="novel-sft")
    if has_books:
        for path in legacy_raw.iterdir():
            if path.is_file():
                shutil.copy2(path, project.raw / path.name)
    if legacy_processed.exists():
        shutil.copytree(legacy_processed, project.processed, dirs_exist_ok=True)
    if has_run:
        marker = legacy_run / "train_config.json"
        began = (marker if marker.exists() else legacy_run).stat().st_mtime
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(began))
        run_dir = project.runs / f"r-{stamp}"
        shutil.move(str(legacy_run), str(run_dir))
        status = read_json(run_dir / "run_status.json", {}) or {}
        cfg = read_json(run_dir / "train_config.json", {}) or {}
        state = {"finished": "trained", "failed": "failed", "stopped": "stopped"}.get(status.get("state"), "interrupted")
        if (run_dir / "lora" / "adapter_model.safetensors").exists() and state != "failed":
            state = "trained"
        preset = next(
            (key for key, spec in PRESETS.items() if all(cfg.get(k) == v for k, v in spec["config"].items() if k in cfg)),
            None,
        )
        write_json(
            run_dir / "run.json",
            {
                "id": run_dir.name,
                "project": project.id,
                "preset": preset,
                "created": began,
                "started": began,
                "ended": max(began, float(status.get("updated") or began)),
                "state": state,
                "detail": "从旧目录 outputs/sft 迁移。LoRA 已保存，还没挂到 Ollama。" if state == "trained" else "从旧目录 outputs/sft 迁移",
                "pid": None,
                "dataset": None,
            },
        )
        if cfg:
            write_json(run_dir / "config.json", cfg)
        if state == "trained":
            from novel_train.pipeline import set_stage

            set_stage(run_dir, "save", "done", 1.0, "LoRA 已保存")
        for stale in ("run_status.json", "export_status.json"):
            (run_dir / stale).unlink(missing_ok=True)
    return project
