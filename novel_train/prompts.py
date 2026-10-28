"""Chat samples shared by data prep, training, and inference."""

from __future__ import annotations

INSTRUCTIONS = [
    "你是这部小说的作者。按命题写小说正文，保持原作的叙述视角、句式节奏和用词习惯。写够要求的字数，只写正文。",
    "按下面的命题写一章小说正文。人称、口吻和叙事节奏和原作一致，场景要写足，不要总结，不要点评。",
    "根据命题写小说正文，写到要求的字数为止。有前情时接着前情往下写，不要重复前情。",
]
DEFAULT_LENGTH = 5000


def instruction_for(index: int) -> str:
    return INSTRUCTIONS[index % len(INSTRUCTIONS)]


def round_length(chars: int) -> int:
    return max(100, int(round(chars / 100.0)) * 100)


def build_user_input(premise: str = "", chapter: str = "", length: int = 0, context: str = "") -> str:
    lines = []
    if premise.strip():
        lines.append(f"命题：{premise.strip()}")
    if chapter.strip():
        lines.append(f"章节：{chapter.strip()}")
    if length:
        lines.append(f"字数：约 {round_length(length)} 字")
    if context.strip():
        lines.append(f"前情：\n{context.strip()}")
    if not premise.strip() and not context.strip():
        lines.append("接着写这部小说的下一段。")
    return "\n".join(lines)


def render_chat(tokenizer, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    """Qwen3.5 会插入思考段。小说正文只学正文，所以关掉 thinking。"""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def to_messages(instruction: str, user: str, output: str | None = None) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user},
    ]
    if output is not None:
        messages.append({"role": "assistant", "content": output})
    return messages
