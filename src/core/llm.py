# -*- coding: utf-8 -*-
"""OpenAI client — singleton client and LLM call wrapper."""

from __future__ import annotations

from openai import OpenAI

from src.config import OPENAI_API_KEY, CHAT_MODEL, TEMPERATURE
from src.core.usage import record_llm_usage

_client = None


def get_client() -> OpenAI:
    """Return a singleton OpenAI client."""
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not found.\n"
                "Create a .env file in the project root with:\n"
                "OPENAI_API_KEY=sk-your_key_here"
            )
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client



def _extract_chat_usage(resp, model: str) -> dict:
    """Extract usage info from a Chat Completions response."""
    usage = getattr(resp, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or (prompt_tokens + completion_tokens))

    prompt_details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens = int(getattr(prompt_details, "cached_tokens", 0) or 0) if prompt_details else 0

    return record_llm_usage(
        model=model,
        prompt_tokens=prompt_tokens,
        cached_prompt_tokens=cached_tokens,
        completion_tokens=completion_tokens,
        total_llm_tokens=total_tokens,
        llm_calls=1,
    )



def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str = CHAT_MODEL,
    temperature: float = TEMPERATURE,
    return_usage: bool = False,
) -> str | tuple[str, dict]:
    """Single LLM call wrapper."""
    client = get_client()
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = resp.choices[0].message.content
    usage = _extract_chat_usage(resp, model)
    if return_usage:
        return content, usage
    return content



def call_llm_chat(
    messages: list[dict],
    model: str = CHAT_MODEL,
    temperature: float = TEMPERATURE,
    return_usage: bool = False,
) -> str | tuple[str, dict]:
    """Multi-turn chat completion. Accepts a full message history."""
    client = get_client()
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
    )
    content = resp.choices[0].message.content
    usage = _extract_chat_usage(resp, model)
    if return_usage:
        return content, usage
    return content
