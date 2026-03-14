# -*- coding: utf-8 -*-
"""OpenAI client — singleton client and LLM call wrapper."""

from openai import OpenAI
from src.config import OPENAI_API_KEY, CHAT_MODEL, TEMPERATURE

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


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str = CHAT_MODEL,
    temperature: float = TEMPERATURE,
) -> str:
    """Single LLM call wrapper. Returns the assistant message content."""
    client = get_client()
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content


def call_llm_chat(
    messages: list[dict],
    model: str = CHAT_MODEL,
    temperature: float = TEMPERATURE,
) -> str:
    """Multi-turn chat completion. Accepts a full message history."""
    client = get_client()
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
    )
    return resp.choices[0].message.content
