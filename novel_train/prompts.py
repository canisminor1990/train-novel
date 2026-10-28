"""Chat samples shared by data prep, training, and inference."""

from __future__ import annotations

INSTRUCTIONS = [
    "请用以下小说的文风续写，保持叙述视角、句式节奏和用词习惯一致。",
    "根据前情续写小说正文，保持人称、口吻和叙事节奏不变。只写正文。",
    "顺着下面的情节继续写。不要总结，不要点评，不要重复前情。",
]


def instruction_for(index: int) -> str:
    return INSTRUCTIONS[index % len(INSTRUCTIONS)]


def build_user_input(chapter: str, context: str) -> str:
    context = context.strip()
    if chapter:
        return f"章节：{chapter}\n前情：\n{context}"
    return f"前情：\n{context}"


def render_chat(tokenizer, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    """Qwen3.5 会插入思考段。小说续写只学正文，所以关掉 thinking。"""
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
