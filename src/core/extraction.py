# -*- coding: utf-8 -*-
"""
Structured extraction helpers.

Shared schema, validation, normalisation, and one-shot repair utilities for
structured extraction tasks across Manual, LangGraph, and LlamaIndex pipelines.
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.core.llm import call_llm

EXTRACTION_SCHEMA_NAME = "technical_doc_summary_v1"
EXTRACTION_FIELDS = [
    "title",
    "objective",
    "methodology",
    "experimental_setup",
    "main_findings",
    "limitations",
]
EXTRACTION_MISSING_VALUE = "Not found in retrieved context"
MAX_RAW_ERROR_EXCERPT = 500

_KEY_ALIASES = {
    "title": "title",
    "document_title": "title",
    "paper_title": "title",
    "objective": "objective",
    "aim": "objective",
    "purpose": "objective",
    "goal": "objective",
    "methodology": "methodology",
    "method": "methodology",
    "methods": "methodology",
    "approach": "methodology",
    "experimental_setup": "experimental_setup",
    "experiment_setup": "experimental_setup",
    "setup": "experimental_setup",
    "experimental_design": "experimental_setup",
    "main_findings": "main_findings",
    "findings": "main_findings",
    "results": "main_findings",
    "key_findings": "main_findings",
    "limitations": "limitations",
    "limitation": "limitations",
    "caveats": "limitations",
    "constraints": "limitations",
}


REPAIR_SYSTEM_PROMPT = (
    "You repair malformed model output for a technical-document extraction task.\n\n"
    f"Return ONE valid JSON object with exactly these keys: {', '.join(EXTRACTION_FIELDS)}.\n"
    f"If a field is missing or unsupported, set it to \"{EXTRACTION_MISSING_VALUE}\".\n"
    "Do not include markdown fences, notes, or any text outside the JSON object."
)


def extraction_schema_instruction() -> str:
    """Prompt-ready instruction describing the canonical extraction schema."""
    return (
        "Return exactly one JSON object with these keys: "
        f"{', '.join(EXTRACTION_FIELDS)}. "
        f"If a field is unsupported, set it to \"{EXTRACTION_MISSING_VALUE}\". "
        "Do not wrap the JSON in markdown fences and do not add extra commentary."
    )


def extraction_field_bullets() -> str:
    """Bullet list of canonical extraction fields for prompts or docs."""
    return "\n".join(f"- {field}" for field in EXTRACTION_FIELDS)


def empty_extraction_payload() -> dict[str, str]:
    """Canonical fallback payload with every field populated."""
    return {field: EXTRACTION_MISSING_VALUE for field in EXTRACTION_FIELDS}


def _truncate_excerpt(text: str, limit: int = MAX_RAW_ERROR_EXCERPT) -> str:
    clean = re.sub(r"\s+", " ", (text or "")).strip()
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "…"


def _strip_code_fences(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return text


def _extract_json_block(raw_text: str) -> str:
    text = _strip_code_fences(raw_text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return text[start : end + 1]
    return text


def _normalise_key(key: Any) -> str:
    text = str(key).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _stringify_value(value: Any) -> str:
    if value is None:
        return EXTRACTION_MISSING_VALUE
    if isinstance(value, str):
        cleaned = re.sub(r"\s+", " ", value).strip()
        return cleaned or EXTRACTION_MISSING_VALUE
    if isinstance(value, list):
        items = []
        for item in value:
            normalised = _stringify_value(item)
            if normalised and normalised != EXTRACTION_MISSING_VALUE:
                items.append(normalised)
        return "; ".join(items) if items else EXTRACTION_MISSING_VALUE
    if isinstance(value, dict):
        if not value:
            return EXTRACTION_MISSING_VALUE
        return json.dumps(value, ensure_ascii=False)
    cleaned = re.sub(r"\s+", " ", str(value)).strip()
    return cleaned or EXTRACTION_MISSING_VALUE


def _canonicalise_candidate(candidate: dict[str, Any]) -> tuple[dict[str, str], list[str], list[str]]:
    payload = empty_extraction_payload()
    missing_fields: list[str] = []
    extra_fields: list[str] = []
    canonical_values: dict[str, Any] = {}

    for raw_key, value in candidate.items():
        canonical_key = _KEY_ALIASES.get(_normalise_key(raw_key))
        if canonical_key is None:
            extra_fields.append(str(raw_key))
            continue
        if canonical_key not in canonical_values:
            canonical_values[canonical_key] = value

    for field in EXTRACTION_FIELDS:
        if field in canonical_values:
            payload[field] = _stringify_value(canonical_values[field])
        else:
            missing_fields.append(field)

    return payload, missing_fields, extra_fields


def _parse_candidate(raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
    candidate_text = _extract_json_block(raw_text)
    try:
        parsed = json.loads(candidate_text)
    except Exception:  # noqa: BLE001
        return None, "json_parse_error"
    if not isinstance(parsed, dict):
        return None, "json_not_object"
    return parsed, None


def _build_error_payload(error_type: str, raw_text: str) -> dict[str, Any]:
    payload: dict[str, Any] = empty_extraction_payload()
    payload["_error"] = {
        "type": error_type,
        "message": "Model output could not be converted into the extraction schema.",
    }
    excerpt = _truncate_excerpt(raw_text)
    if excerpt:
        payload["_error"]["raw_output_excerpt"] = excerpt
    return payload


def _format_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def repair_extraction_json(raw_text: str) -> str:
    """Use the LLM once to repair malformed extraction output into valid JSON."""
    user_prompt = (
        "Malformed extraction output:\n"
        f"{raw_text}\n\n"
        "Return repaired JSON only."
    )
    return call_llm(REPAIR_SYSTEM_PROMPT, user_prompt, temperature=0.0)


def validate_and_format_extraction(
    raw_text: str,
    *,
    attempt_repair: bool = True,
) -> tuple[str, dict[str, Any]]:
    """
    Validate extraction output against the shared schema.

    Returns:
        tuple[str, dict]:
            - formatted JSON string to store in PipelineResult.answer
            - metadata describing schema validity, repair status, and field coverage
    """
    metadata: dict[str, Any] = {
        "task_kind": "structured_extraction",
        "schema_name": EXTRACTION_SCHEMA_NAME,
        "schema_fields": list(EXTRACTION_FIELDS),
        "schema_valid": False,
        "schema_error_type": None,
        "schema_source": None,
        "repair_attempted": False,
        "repair_succeeded": False,
        "schema_missing_fields": [],
        "schema_extra_fields": [],
    }

    parsed, parse_error = _parse_candidate(raw_text)
    if parsed is not None:
        payload, missing_fields, extra_fields = _canonicalise_candidate(parsed)
        metadata.update(
            {
                "schema_valid": True,
                "schema_source": "raw",
                "schema_missing_fields": missing_fields,
                "schema_extra_fields": extra_fields,
            }
        )
        return _format_payload(payload), metadata

    if attempt_repair:
        metadata["repair_attempted"] = True
        metadata["schema_initial_error_type"] = parse_error
        try:
            repaired_raw = repair_extraction_json(raw_text)
            repaired, repair_error = _parse_candidate(repaired_raw)
        except Exception as exc:  # noqa: BLE001
            repaired = None
            repair_error = f"repair_call_failed: {exc}"
            repaired_raw = ""

        if repaired is not None:
            payload, missing_fields, extra_fields = _canonicalise_candidate(repaired)
            metadata.update(
                {
                    "schema_valid": True,
                    "schema_source": "repaired",
                    "repair_succeeded": True,
                    "schema_missing_fields": missing_fields,
                    "schema_extra_fields": extra_fields,
                }
            )
            return _format_payload(payload), metadata

        error_type = repair_error or parse_error or "unknown_schema_error"
        metadata.update(
            {
                "schema_valid": False,
                "schema_error_type": error_type,
                "schema_source": "error_fallback",
                "schema_missing_fields": list(EXTRACTION_FIELDS),
                "schema_extra_fields": [],
            }
        )
        return _format_payload(_build_error_payload(error_type, raw_text or repaired_raw)), metadata

    error_type = parse_error or "unknown_schema_error"
    metadata.update(
        {
            "schema_valid": False,
            "schema_error_type": error_type,
            "schema_source": "error_fallback",
            "schema_missing_fields": list(EXTRACTION_FIELDS),
            "schema_extra_fields": [],
        }
    )
    return _format_payload(_build_error_payload(error_type, raw_text)), metadata
