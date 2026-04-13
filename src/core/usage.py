# -*- coding: utf-8 -*-
"""Usage tracking and estimated cost helpers for OpenAI-powered pipelines."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, Iterator

# Current model pricing used by this repo.
# OpenAI's GPT-4.1 mini model page lists $0.40 / 1M input tokens,
# $0.10 / 1M cached input tokens, and $1.60 / 1M output tokens.
# The text-embedding-3-small model page lists $0.02 / 1M tokens.
LLM_PRICING_PER_1M: dict[str, dict[str, float]] = {
    "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
}
EMBEDDING_PRICING_PER_1M: dict[str, float] = {
    "text-embedding-3-small": 0.02,
}

_ACTIVE_USAGE_TRACKERS: list[dict[str, Any]] = []


def canonical_model_name(model_name: str | None) -> str | None:
    """Normalise snapshot aliases to a stable pricing key when possible."""
    if not model_name:
        return model_name
    known_prefixes = (
        "gpt-4.1-mini",
        "text-embedding-3-small",
    )
    for prefix in known_prefixes:
        if model_name == prefix or model_name.startswith(prefix + "-"):
            return prefix
    return model_name



def empty_usage(
    llm_model: str | None = None,
    embedding_model: str | None = None,
) -> Dict[str, Any]:
    """Create an empty usage payload with a stable schema."""
    return {
        "llm_model": canonical_model_name(llm_model),
        "embedding_model": canonical_model_name(embedding_model),
        "llm_calls": 0,
        "embedding_calls": 0,
        "prompt_tokens": 0,
        "cached_prompt_tokens": 0,
        "completion_tokens": 0,
        "total_llm_tokens": 0,
        "embedding_tokens": 0,
        "total_api_tokens": 0,
        "estimated_cost_usd": 0.0,
        "cost_breakdown": {
            "llm_input_usd": 0.0,
            "llm_cached_input_usd": 0.0,
            "llm_output_usd": 0.0,
            "embedding_usd": 0.0,
        },
    }



def _as_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0



def _increment_usage(
    target: Dict[str, Any],
    *,
    llm_model: str | None = None,
    embedding_model: str | None = None,
    llm_calls: int = 0,
    embedding_calls: int = 0,
    prompt_tokens: int = 0,
    cached_prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_llm_tokens: int | None = None,
    embedding_tokens: int = 0,
) -> None:
    """Mutate a usage dict by adding new counts."""
    llm_model = canonical_model_name(llm_model)
    embedding_model = canonical_model_name(embedding_model)

    if llm_model and not target.get("llm_model"):
        target["llm_model"] = llm_model
    if embedding_model and not target.get("embedding_model"):
        target["embedding_model"] = embedding_model

    prompt_tokens = _as_int(prompt_tokens)
    cached_prompt_tokens = min(_as_int(cached_prompt_tokens), prompt_tokens)
    completion_tokens = _as_int(completion_tokens)
    embedding_tokens = _as_int(embedding_tokens)
    total_llm_tokens = _as_int(total_llm_tokens) if total_llm_tokens is not None else prompt_tokens + completion_tokens

    target["llm_calls"] = _as_int(target.get("llm_calls")) + _as_int(llm_calls)
    target["embedding_calls"] = _as_int(target.get("embedding_calls")) + _as_int(embedding_calls)
    target["prompt_tokens"] = _as_int(target.get("prompt_tokens")) + prompt_tokens
    target["cached_prompt_tokens"] = _as_int(target.get("cached_prompt_tokens")) + cached_prompt_tokens
    target["completion_tokens"] = _as_int(target.get("completion_tokens")) + completion_tokens
    target["total_llm_tokens"] = _as_int(target.get("total_llm_tokens")) + total_llm_tokens
    target["embedding_tokens"] = _as_int(target.get("embedding_tokens")) + embedding_tokens



def finalise_usage(usage: Dict[str, Any] | None) -> Dict[str, Any]:
    """Return a normalised usage payload with derived totals and estimated cost."""
    payload = deepcopy(usage) if usage else empty_usage()

    prompt_tokens = _as_int(payload.get("prompt_tokens"))
    cached_prompt_tokens = min(_as_int(payload.get("cached_prompt_tokens")), prompt_tokens)
    completion_tokens = _as_int(payload.get("completion_tokens"))
    total_llm_tokens = _as_int(payload.get("total_llm_tokens")) or (prompt_tokens + completion_tokens)
    embedding_tokens = _as_int(payload.get("embedding_tokens"))

    llm_model = canonical_model_name(payload.get("llm_model"))
    embedding_model = canonical_model_name(payload.get("embedding_model"))
    llm_pricing = LLM_PRICING_PER_1M.get(llm_model, {})
    embedding_price = EMBEDDING_PRICING_PER_1M.get(embedding_model, 0.0)

    uncached_prompt_tokens = max(prompt_tokens - cached_prompt_tokens, 0)
    llm_input_usd = uncached_prompt_tokens * float(llm_pricing.get("input", 0.0)) / 1_000_000
    llm_cached_input_usd = cached_prompt_tokens * float(
        llm_pricing.get("cached_input", llm_pricing.get("input", 0.0))
    ) / 1_000_000
    llm_output_usd = completion_tokens * float(llm_pricing.get("output", 0.0)) / 1_000_000
    embedding_usd = embedding_tokens * float(embedding_price) / 1_000_000

    payload["llm_model"] = llm_model
    payload["embedding_model"] = embedding_model
    payload["llm_calls"] = _as_int(payload.get("llm_calls"))
    payload["embedding_calls"] = _as_int(payload.get("embedding_calls"))
    payload["prompt_tokens"] = prompt_tokens
    payload["cached_prompt_tokens"] = cached_prompt_tokens
    payload["completion_tokens"] = completion_tokens
    payload["total_llm_tokens"] = total_llm_tokens
    payload["embedding_tokens"] = embedding_tokens
    payload["total_api_tokens"] = total_llm_tokens + embedding_tokens
    payload["estimated_cost_usd"] = llm_input_usd + llm_cached_input_usd + llm_output_usd + embedding_usd
    payload["cost_breakdown"] = {
        "llm_input_usd": llm_input_usd,
        "llm_cached_input_usd": llm_cached_input_usd,
        "llm_output_usd": llm_output_usd,
        "embedding_usd": embedding_usd,
    }
    return payload



def merge_usage(*usages: Dict[str, Any] | None) -> Dict[str, Any]:
    """Merge multiple usage payloads into one normalised summary."""
    merged = empty_usage()
    for usage in usages:
        if not usage:
            continue
        _increment_usage(
            merged,
            llm_model=usage.get("llm_model"),
            embedding_model=usage.get("embedding_model"),
            llm_calls=usage.get("llm_calls", 0),
            embedding_calls=usage.get("embedding_calls", 0),
            prompt_tokens=usage.get("prompt_tokens", 0),
            cached_prompt_tokens=usage.get("cached_prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_llm_tokens=usage.get("total_llm_tokens"),
            embedding_tokens=usage.get("embedding_tokens", 0),
        )
    return finalise_usage(merged)



def record_llm_usage(
    *,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_llm_tokens: int | None = None,
    cached_prompt_tokens: int = 0,
    llm_calls: int = 1,
) -> Dict[str, Any]:
    """Record a single chat-completion usage payload and update active trackers."""
    single = empty_usage(llm_model=model)
    _increment_usage(
        single,
        llm_model=model,
        llm_calls=llm_calls,
        prompt_tokens=prompt_tokens,
        cached_prompt_tokens=cached_prompt_tokens,
        completion_tokens=completion_tokens,
        total_llm_tokens=total_llm_tokens,
    )
    for tracker in _ACTIVE_USAGE_TRACKERS:
        _increment_usage(
            tracker,
            llm_model=model,
            llm_calls=llm_calls,
            prompt_tokens=prompt_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
            completion_tokens=completion_tokens,
            total_llm_tokens=total_llm_tokens,
        )
    return finalise_usage(single)



def record_embedding_usage(
    *,
    model: str,
    embedding_tokens: int = 0,
    embedding_calls: int = 1,
) -> Dict[str, Any]:
    """Record a single embedding usage payload and update active trackers."""
    single = empty_usage(embedding_model=model)
    _increment_usage(
        single,
        embedding_model=model,
        embedding_calls=embedding_calls,
        embedding_tokens=embedding_tokens,
    )
    for tracker in _ACTIVE_USAGE_TRACKERS:
        _increment_usage(
            tracker,
            embedding_model=model,
            embedding_calls=embedding_calls,
            embedding_tokens=embedding_tokens,
        )
    return finalise_usage(single)


@contextmanager
def usage_tracking_session(
    llm_model: str | None = None,
    embedding_model: str | None = None,
) -> Iterator[Dict[str, Any]]:
    """Context manager that accumulates nested OpenAI usage in a shared dict."""
    tracker = empty_usage(llm_model=llm_model, embedding_model=embedding_model)
    _ACTIVE_USAGE_TRACKERS.append(tracker)
    try:
        yield tracker
    finally:
        try:
            _ACTIVE_USAGE_TRACKERS.remove(tracker)
        except ValueError:
            pass
        final_payload = finalise_usage(tracker)
        tracker.clear()
        tracker.update(final_payload)
