"""Merge a run's LoRA, write a GGUF and register it in Ollama under the project's model name."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import traceback
from pathlib import Path

from novel_train.envload import ROOT, configure_stdio, load_project_env
from novel_train.ollama_bin import forget_cache, model_ready, ollama_executable, think_model_name
from novel_train.pipeline import end_current, estimate_remaining, log_progress, set_stage
from novel_train.prompts import INSTRUCTIONS

load_project_env()

# Unsloth 从 GitHub Release 拉 llama.cpp。直连这里很慢，而且会停在半开的连接上。
_GITHUB_MIRROR = "https://gh-proxy.com/"
_MIRRORED_PREFIXES = (
    "https://github.com/",
    "https://release-assets.githubusercontent.com/",
    "https://objects.githubusercontent.com/",
    "https://codeload.github.com/",
)


def route_github_downloads() -> None:
    """把 llama.cpp 的 Release 下载改走镜像。api.github.com 只返回清单，保持直连。"""
    import os

    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    import unsloth_zoo.llama_cpp as llama_cpp

    if getattr(llama_cpp, "_novel_github_mirror", False):
        return

    def mirrored(url: str) -> str:
        if not isinstance(url, str) or url.startswith(_GITHUB_MIRROR):
            return url
        if url.startswith(_MIRRORED_PREFIXES):
            return _GITHUB_MIRROR + url
        return url

    download = llama_cpp._download_archive

    def _download_archive(url, dest_path):
        return download(mirrored(url), dest_path)

    request = llama_cpp._requests_get_with_retries

    def _requests_get_with_retries(url, timeout=(10, 120), headers=None, stream=False, max_attempts=3):
        return request(mirrored(url), timeout=timeout, headers=headers, stream=stream, max_attempts=max_attempts)

    llama_cpp._download_archive = _download_archive
    llama_cpp._requests_get_with_retries = _requests_get_with_retries
    llama_cpp._novel_github_mirror = True


# Qwen3.5 看到空的思考段就会接着写正文。不写这段时，它会先抛出思考。
_THINK_OFF = "<think>\n\n</think>\n\n"
# 64K 在 Ollama 里约占 5GB 显存。模型用完会在显存里停留 5 分钟，再往上开会和随后开始的训练抢显存。
# -think 版沿用基座的思考，常写几千 token，所以同样需要这个长度。
_NUM_CTX = 65536


# 跟 Ollama 的 qwen3.5 渲染器同一套 XML。解析用 qwen3-coder：它只认 <tool_call>，
# 不会把思考默认打开。qwen3.5 解析器在请求没带 think=false 时会把工具调用吞进思考。
_TOOL_SPEC = (
    "# Tools\n\nYou have access to the following functions:\n\n<tools>\n"
    "{{ range .Tools }}{{ . | json }}\n{{ end }}"
    "</tools>\n\n"
    "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n"
    "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\n"
    "This is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n"
    "<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>"
)
_TOOL_CALLS = (
    "{{ if .ToolCalls }}{{ range .ToolCalls }}"
    "<tool_call>\n<function={{ .Function.Name }}>\n"
    "{{ range $key, $value := .Function.Arguments }}"
    "<parameter={{ $key }}>\n{{ $value }}\n</parameter>\n"
    "{{ end }}</function>\n</tool_call>\n"
    "{{ end }}{{ end }}"
)


def render_modelfile(filename: str, *, thinking: bool) -> str:
    if thinking:
        past_thinking = "{{ if .Thinking }}<think>\n{{ .Thinking }}\n</think>\n\n{{ end }}"
        assistant = (
            "<|im_start|>assistant\n"
            "{{ if and $.IsThinkSet (not $.Think) }}" + _THINK_OFF + "{{ else }}<think>\n{{ end }}"
        )
    else:
        past_thinking = ""
        assistant = "<|im_start|>assistant\n" + _THINK_OFF
    template = (
        "{{ if .Tools }}<|im_start|>system\n"
        + _TOOL_SPEC
        + "{{ if .System }}\n\n{{ .System }}{{ end }}<|im_end|>\n"
        "{{ else if .System }}<|im_start|>system\n"
        "{{ .System }}<|im_end|>\n"
        "{{ end }}{{ range .Messages }}{{ if eq .Role \"user\" }}<|im_start|>user\n"
        "{{ .Content }}<|im_end|>\n"
        "{{ else if eq .Role \"assistant\" }}<|im_start|>assistant\n"
        + past_thinking
        + "{{ .Content }}"
        + _TOOL_CALLS
        + "<|im_end|>\n"
        "{{ else if eq .Role \"tool\" }}<|im_start|>user\n"
        "<tool_response>\n{{ .Content }}\n</tool_response><|im_end|>\n"
        "{{ end }}{{ end }}"
        + assistant
    )
    return (
        f"FROM ./{filename}\n"
        f'TEMPLATE """{template}"""\n'
        # 能力位看模板里有没有 .Tools。真正拆 <tool_call> 靠这个解析器。
        "PARSER qwen3-coder\n"
        f'SYSTEM """{INSTRUCTIONS[0]}"""\n'
        "PARAMETER stop <|im_end|>\n"
        "PARAMETER temperature 0.8\n"
        # min-p 随置信度收放候选，长文里比只用 top-p 少跑题，也少重复。
        "PARAMETER min_p 0.05\n"
        "PARAMETER top_p 0.95\n"
        "PARAMETER top_k 64\n"
        "PARAMETER repeat_penalty 1.05\n"
        "PARAMETER repeat_last_n 256\n"
        f"PARAMETER num_ctx {_NUM_CTX}\n"
    )


def _gguf_py() -> Path:
    import os

    return Path(os.environ.get("UNSLOTH_LLAMA_CPP_PATH") or ROOT / ".cache" / "llama.cpp") / "gguf-py"


def drop_chat_template(gguf: Path) -> None:
    """Remove the GGUF's embedded Jinja template so Ollama runs the Modelfile TEMPLATE.

    Ollama prefers the embedded Qwen3.5 template because it declares tools and thinking.
    That template thinks unless the request sends think=false, which bypasses the prose template.
    """
    import os

    package = _gguf_py()
    env = {**os.environ, "PYTHONPATH": str(package)}
    probe = (
        "import sys; from gguf import GGUFReader; "
        "sys.exit(0 if 'tokenizer.chat_template' in GGUFReader(sys.argv[1]).fields else 3)"
    )
    found = subprocess.run([sys.executable, "-c", probe, str(gguf)], env=env, capture_output=True, text=True)
    if found.returncode == 3:
        return
    if found.returncode != 0:
        raise RuntimeError(f"读不了 {gguf.name} 的元数据，退出代码 {found.returncode}：{found.stderr.strip()[-300:]}")
    partial = gguf.with_name(gguf.name + ".part")
    script = package / "gguf" / "scripts" / "gguf_new_metadata.py"
    try:
        subprocess.run(
            [sys.executable, str(script), "--remove-metadata", "tokenizer.chat_template", "--force", str(gguf), str(partial)],
            env=env,
            check=True,
        )
        partial.replace(gguf)
    finally:
        partial.unlink(missing_ok=True)


def build_dir(run_dir: Path) -> Path:
    return run_dir / "ollama_build"


def gguf_dir(run_dir: Path) -> Path:
    return run_dir / "ollama_build_gguf"


def merged_model_ready(directory: Path) -> bool:
    """True when the 16-bit merge finished. A later export can convert it without reloading."""
    config = directory / "config.json"
    index_path = directory / "model.safetensors.index.json"
    if not config.is_file():
        return False
    if not index_path.is_file():
        single = directory / "model.safetensors"
        return single.is_file() and single.stat().st_size > 1_000_000 and (directory / "tokenizer_config.json").is_file()
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        return False
    shards = {name for name in weight_map.values() if isinstance(name, str) and name}
    if not shards or not (directory / "tokenizer_config.json").is_file():
        return False
    actual = 0
    for name in shards:
        path = directory / name
        if not path.is_file():
            return False
        actual += path.stat().st_size
    total = (index.get("metadata") or {}).get("total_size") if isinstance(index.get("metadata"), dict) else None
    if isinstance(total, int) and total > 0 and actual < total:
        return False
    return actual > 1_000_000


def export_plan(run_dir: Path) -> str:
    """Next export step that does not repeat a finished one: mount, gguf, or merge."""
    if text_ggufs(gguf_dir(run_dir)):
        return "mount"
    if merged_model_ready(build_dir(run_dir)):
        return "gguf"
    return "merge"


def text_ggufs(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    files = [path for path in directory.glob("*.gguf") if "mmproj" not in path.name.lower() and path.stat().st_size > 50_000_000]
    quantized = [path for path in files if any(tag in path.name.lower() for tag in ("q4", "q5", "q8"))]
    return quantized or files


def write_modelfile(directory: Path, gguf: Path, *, thinking: bool) -> Path:
    path = directory / ("Modelfile.think" if thinking else "Modelfile")
    path.write_text(render_modelfile(gguf.name, thinking=thinking), encoding="utf-8")
    return path


def register(run_dir: Path, directory: Path, model: str, modelfile: Path) -> None:
    exe = ollama_executable()
    if exe is None:
        raise RuntimeError(f"GGUF 已写到 {directory}。没有找到 ollama.exe，装好 Ollama 后点「挂到 Ollama」。")
    set_stage(run_dir, "ollama", "running", None, f"正在执行 ollama create {model}")
    subprocess.run([exe, "create", model, "-f", modelfile.name], cwd=directory, check=True)
    forget_cache()


def _mount_detail(model: str, think: str) -> str:
    return f"已挂载到 Ollama。正文：ollama run {model} ；带思考：ollama run {think}"


def mount(run_dir: Path, model: str, force: bool = False) -> str | None:
    """Register one GGUF as two Ollama models: prose, and a -think sibling."""
    found = text_ggufs(gguf_dir(run_dir))
    if not found:
        return None
    think = think_model_name(model)
    directory = gguf_dir(run_dir)
    pending = [
        (name, thinking)
        for name, thinking in ((model, False), (think, True))
        if force or not model_ready(name)
    ]
    if pending and ollama_executable() is None:
        raise RuntimeError(f"GGUF 已写到 {directory}。没有找到 ollama.exe，装好 Ollama 后点「挂到 Ollama」。")
    if pending:
        set_stage(run_dir, "ollama", "running", None, "正在去掉 GGUF 里自带的对话模板")
        drop_chat_template(found[0])
    for name, thinking in pending:
        modelfile = write_modelfile(directory, found[0], thinking=thinking)
        register(run_dir, directory, name, modelfile)
    detail = _mount_detail(model, think)
    set_stage(run_dir, "ollama", "done", 1, detail)
    return detail


def _note(run_dir: Path, stage_id: str, detail: str) -> None:
    progress = log_progress(run_dir, stage_id)
    extra = {}
    remaining = estimate_remaining(run_dir, stage_id, progress)
    if remaining is not None:
        extra["eta"] = remaining
    set_stage(run_dir, stage_id, "running", progress, detail, **extra)


def _watch_log(run_dir: Path, stop: threading.Event, only: str | None = None) -> None:
    log_path = run_dir / "export.log"
    # Only the bytes written after this export starts. The file also holds earlier attempts.
    offset = log_path.stat().st_size if log_path.exists() else 0
    while not stop.wait(2):
        if not log_path.exists():
            continue
        text = log_path.read_bytes()[offset:].decode("utf-8", errors="replace")
        converting = "Converting to GGUF" in text or "Installing llama" in text or "quantiz" in text.lower()
        if only == "gguf":
            if converting:
                _note(run_dir, "gguf", "正在把已合并的模型转成 GGUF")
        elif converting:
            _note(run_dir, "gguf", "正在把合并后的模型转成 GGUF")
        elif "Merging" in text or "Loading weights" in text:
            _note(run_dir, "merge", "正在把 LoRA 合并进基座")


def _convert_merged(run_dir: Path, model: str) -> None:
    """Turn a finished 16-bit merge into GGUF. Does not load the LoRA again."""
    directory = build_dir(run_dir)
    for path in directory.glob("*.gguf"):
        path.unlink(missing_ok=True)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    dtype = config.get("torch_dtype") or text.get("torch_dtype") or "bfloat16"
    if dtype not in {"float16", "bfloat16"}:
        dtype = "bfloat16"
    architectures = config.get("architectures") or []
    is_vlm = "vision_config" in config or any(
        isinstance(name, str) and name.endswith(("ForConditionalGeneration", "ForVisionText2Text"))
        for name in architectures
    )
    model_type = str(config.get("model_type") or "")
    if not model_type:
        raise RuntimeError("合并后的 config.json 里没有 model_type，没法继续转 GGUF。")
    print("沿用已合并的 16 位权重，跳过重新加载和合并。", flush=True)
    route_github_downloads()
    from unsloth.save import save_to_gguf

    set_stage(run_dir, "gguf", "running", None, "正在把已合并的模型转成 GGUF")
    save_to_gguf(
        model_name=model.replace(":", "-"),
        model_type=model_type,
        model_dtype=dtype,
        is_sentencepiece=False,
        model_directory=str(directory),
        quantization_method="q4_k_m",
        is_vlm=is_vlm,
        gguf_directory=str(gguf_dir(run_dir)),
        merge_is_disposable=False,
    )


def _drop_merged(run_dir: Path) -> None:
    shutil.rmtree(build_dir(run_dir), ignore_errors=True)


def export(run_dir: Path, model: str) -> str:
    adapter = run_dir / "lora"
    plan = export_plan(run_dir)
    if plan == "mount":
        found = text_ggufs(gguf_dir(run_dir))
        set_stage(run_dir, "gguf", "done", 1, f"沿用已有的 {found[0].name}")
        return mount(run_dir, model, force=True) or ""
    if plan == "gguf":
        set_stage(run_dir, "merge", "done", 1, "沿用已合并的 16 位权重")
        stop = threading.Event()
        threading.Thread(target=_watch_log, args=(run_dir, stop, "gguf"), daemon=True).start()
        try:
            _convert_merged(run_dir, model)
        finally:
            stop.set()
        if not text_ggufs(gguf_dir(run_dir)):
            raise RuntimeError("没有找到文本 GGUF。")
        _drop_merged(run_dir)
        set_stage(run_dir, "gguf", "done", 1, "GGUF 已写好，合并用的 16 位权重已删除")
        return mount(run_dir, model, force=True) or ""
    set_stage(run_dir, "merge", "running", None, "正在加载并合并 LoRA")
    if not (adapter / "adapter_model.safetensors").exists():
        raise FileNotFoundError("这次运行还没有 LoRA。")
    route_github_downloads()
    from novel_train.ollama_bin import unload_all

    unload_all()
    from unsloth import FastLanguageModel

    stop = threading.Event()
    threading.Thread(target=_watch_log, args=(run_dir, stop), daemon=True).start()
    try:
        model_obj, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter),
            max_seq_length=2048,
            load_in_4bit=False,
            load_in_16bit=True,
            dtype=None,
        )
        result = model_obj.save_pretrained_gguf(str(build_dir(run_dir)), tokenizer, quantization_method="q4_k_m")
    finally:
        stop.set()
    directory = Path((result or {}).get("gguf_directory") or gguf_dir(run_dir))
    found = text_ggufs(directory)
    if not found:
        raise RuntimeError("没有找到文本 GGUF。")
    if directory != gguf_dir(run_dir):
        gguf_dir(run_dir).mkdir(parents=True, exist_ok=True)
        for path in found:
            shutil.move(str(path), str(gguf_dir(run_dir) / path.name))
    del model_obj
    shutil.rmtree(build_dir(run_dir), ignore_errors=True)
    set_stage(run_dir, "gguf", "done", 1, "GGUF 已写好，合并用的 16 位权重已删除")
    return mount(run_dir, model, force=True) or ""


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    parser = argparse.ArgumentParser(description="把一次运行的 LoRA 导出成 GGUF 并挂到 Ollama。")
    parser.add_argument("--run", required=True, help="运行目录，例如 projects/<项目>/runs/<运行>")
    parser.add_argument("--model", required=True, help="Ollama 里的模型名")
    args = parser.parse_args(argv)
    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    try:
        detail = export(run_dir, args.model)
        print(detail)
        return 0
    except Exception as exc:
        traceback.print_exc()
        text = str(exc) or type(exc).__name__
        if isinstance(exc, subprocess.CalledProcessError):
            text = f"ollama create 失败，退出代码 {exc.returncode}。"
        end_current(run_dir, "failed", text)
        return 1


if __name__ == "__main__":
    sys.exit(main())
