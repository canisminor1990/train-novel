"""Text-only continuation. Qwen3.5's processor treats a raw string as an image."""

from __future__ import annotations

import re

from novel_train.prompts import build_user_input, render_chat, to_messages

_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def text_tokenizer(tokenizer):
    inner = getattr(tokenizer, "tokenizer", None)
    if inner is not None and hasattr(inner, "apply_chat_template"):
        return inner
    return tokenizer


def continue_prose(
    model,
    tokenizer,
    user_text: str,
    *,
    instruction: str,
    max_seq_length: int = 2048,
    max_new_tokens: int = 400,
    temperature: float = 0.85,
    top_p: float = 0.9,
    repetition_penalty: float = 1.08,
) -> str:
    import torch

    tok = text_tokenizer(tokenizer)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompt = render_chat(tok, to_messages(instruction, user_text), add_generation_prompt=True)
    encoded = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_length,
        add_special_tokens=False,
    )
    device = next(model.parameters()).device
    inputs = {
        key: value.to(device)
        for key, value in encoded.items()
        if key in {"input_ids", "attention_mask"}
    }
    do_sample = temperature > 0
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            repetition_penalty=repetition_penalty,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    new_tokens = generated[0][inputs["input_ids"].shape[1] :]
    text = tok.decode(new_tokens, skip_special_tokens=True)
    text = _THINK.sub("", text)
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text.strip()


def prose_from_premise(model, tokenizer, premise: str, length: int, context: str = "", **kwargs) -> str:
    return continue_prose(model, tokenizer, build_user_input(premise, "", length, context), **kwargs)
