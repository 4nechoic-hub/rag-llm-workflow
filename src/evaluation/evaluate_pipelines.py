# -*- coding: utf-8 -*-
"""
RAG Pipeline Evaluation — reusable benchmark suite

Compares Manual Pipeline vs LangGraph Agent vs LlamaIndex across:
  - Answer quality (LLM-as-judge: completeness, grounding, clarity)
  - Citation accuracy (do inline citations match retrieved sources?)
  - Hallucination detection (are there unsupported claims?)
  - Retrieval metadata (retrieval mode, reranking, chunking)
  - Query latency, plus one-time pipeline preparation time
  - Cross-pipeline agreement
  - Refusal behavior on unanswerable questions

Run:
    python -m src.evaluation.evaluate_pipelines

Outputs:
  - eval_results/evaluation_results.csv
  - eval_results/evaluation_results.jsonl
  - eval_results/*.png

Author: Tingyi Zhang
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import EVAL_OUTPUT_DIR, TOP_K
from src.core.llm import call_llm
from src.core.types import PipelineResult
from src.core.usage import empty_usage, finalise_usage, usage_tracking_session


# ── Constants ───────────────────────────────────────────────────

EVAL_CASES_PATH = Path(__file__).with_name("eval_cases.jsonl")
EVAL_OUTPUT_PATH = Path(EVAL_OUTPUT_DIR)
CSV_OUTPUT_PATH = EVAL_OUTPUT_PATH / "evaluation_results.csv"
JSONL_OUTPUT_PATH = EVAL_OUTPUT_PATH / "evaluation_results.jsonl"

COLOURS = {"Manual": "#4C78A8", "LangGraph": "#F58518", "LlamaIndex": "#54A24B"}
PIPE_ORDER = ["Manual", "LangGraph", "LlamaIndex"]
DEFAULT_QUERY_TYPE_ORDER = ["factual", "summary", "technical", "comparison", "critical", "unanswerable"]

REFUSAL_HINTS = [
    "insufficient evidence",
    "insufficient information",
    "not enough information",
    "not enough evidence",
    "cannot determine",
    "can not determine",
    "cannot be determined",
    "can not be determined",
    "not stated",
    "not specified",
    "not mentioned",
    "not provided",
    "not described",
    "not available",
    "not clear from the documents",
    "not clear from the retrieved",
    "the documents do not specify",
    "the retrieved documents do not specify",
    "the sources do not specify",
    "i cannot verify",
    "i can't verify",
    "i cannot confirm",
    "i can't confirm",
    "no evidence",
    "not found in the retrieved",
]


# ── Case loading ────────────────────────────────────────────────


def load_eval_cases(case_path: Path = EVAL_CASES_PATH) -> List[Dict[str, Any]]:
    """Load evaluation cases from JSONL."""
    if not case_path.exists():
        raise FileNotFoundError(f"Evaluation case file not found: {case_path}")

    cases: List[Dict[str, Any]] = []
    seen_ids = set()

    with case_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {case_path}: {exc}") from exc

            query_id = case.get("id")
            query = case.get("query")
            query_type = case.get("type")

            if not query_id or not query or not query_type:
                raise ValueError(
                    f"Each eval case must define 'id', 'query', and 'type'. Problem on line {line_no}: {case}"
                )

            if query_id in seen_ids:
                raise ValueError(f"Duplicate eval case id detected: {query_id}")
            seen_ids.add(query_id)

            case["answerable"] = bool(case.get("answerable", True))
            case["notes"] = case.get("notes", "")
            cases.append(case)

    if not cases:
        raise ValueError(f"No evaluation cases loaded from {case_path}")

    return cases


# ── Utility helpers ─────────────────────────────────────────────


def to_jsonable(value: Any) -> Any:
    """Convert nested objects into JSON-serialisable Python primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    return str(value)



def normalise_sources(sources: Iterable[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    """Return a JSON-safe list of source dicts."""
    if not sources:
        return []
    return [to_jsonable(source) for source in sources]



def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default



def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default



def extract_json_block(raw_text: str) -> str:
    """Best-effort extraction of a JSON object from an LLM response."""
    text = (raw_text or "").strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return text[start:end + 1]
    return text



def parse_judge_output(raw_text: str, fallback: Dict[str, Any]) -> Tuple[Dict[str, Any], str | None]:
    """Parse judge JSON, returning parsed content plus an optional parse error."""
    try:
        parsed = json.loads(extract_json_block(raw_text))
        if not isinstance(parsed, dict):
            raise ValueError("Judge response was not a JSON object")
        return parsed, None
    except Exception as exc:  # noqa: BLE001
        parsed = dict(fallback)
        parsed["feedback"] = f"Parse error: {exc}"
        return parsed, str(exc)



def build_source_summary(sources: List[Dict[str, Any]]) -> str:
    if not sources:
        return "  (no sources retrieved)"
    return "\n".join(f"  - {s.get('file')} | Page {s.get('page')}" for s in sources)



def detect_refusal(case: Dict[str, Any], answer: str) -> bool | None:
    """Heuristic refusal check for unanswerable cases."""
    if case.get("answerable", True):
        return None

    text = (answer or "").strip().lower()
    if not text:
        return False
    return any(hint in text for hint in REFUSAL_HINTS)



def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")



def pipeline_metadata(result: PipelineResult) -> Dict[str, Any]:
    metadata = getattr(result, "metadata", None) or {}
    if not isinstance(metadata, dict):
        return {"raw_metadata": to_jsonable(metadata)}
    return {str(k): to_jsonable(v) for k, v in metadata.items()}



def normalise_usage(usage: Dict[str, Any] | None) -> Dict[str, Any]:
    """Return a stable, cost-aware usage payload."""
    if not isinstance(usage, dict):
        usage = empty_usage()
    return finalise_usage(usage)



def build_error_result(message: str) -> PipelineResult:
    result = PipelineResult(answer=message)
    result.latency = 0.0
    return result



def blank_quality_scores(error_message: str) -> Dict[str, Any]:
    return {
        "completeness": 0,
        "grounding": 0,
        "clarity": 0,
        "overall": 0,
        "feedback": error_message,
    }



def blank_citation_scores(error_message: str) -> Dict[str, Any]:
    return {
        "citation_precision": 0,
        "citation_coverage": 0,
        "citation_accuracy": 0,
        "num_citations_found": 0,
        "num_valid_citations": 0,
        "feedback": error_message,
    }



def blank_hallucination_scores(error_message: str) -> Dict[str, Any]:
    return {
        "hallucination_score": 0,
        "num_claims_checked": 0,
        "num_unsupported_claims": 0,
        "flagged_claims": [],
        "feedback": error_message,
    }


# ── LLM-as-Judge ────────────────────────────────────────────────


def judge_answer(case: Dict[str, Any], answer: str) -> Dict[str, Any]:
    """Score an answer on completeness, grounding, and clarity."""
    fallback = blank_quality_scores("Parse error")
    system = (
        "You are a strict evaluator for technical document Q&A systems.\n\n"
        "Score on three dimensions (each 0-10):\n"
        "  - Completeness: Does the answer address the full question?\n"
        "  - Grounding: Does the answer stay tied to retrieved evidence rather than speculation?\n"
        "  - Clarity: Is the answer well-structured and easy to follow?\n\n"
        "Important: some evaluation cases are intentionally unanswerable from the documents. "
        "If answerable=false, a strong answer should clearly say the evidence is insufficient and avoid fabricating details. "
        "Do not penalise a correct refusal for not inventing nonexistent facts.\n\n"
        "Return ONLY valid JSON:\n"
        '{"completeness": <int>, "grounding": <int>, "clarity": <int>, '
        '"overall": <average>, "feedback": "<one sentence>"}'
    )
    user = (
        f"Question type: {case.get('type')}\n"
        f"Answerable from documents: {case.get('answerable', True)}\n"
        f"Question: {case.get('query')}\n\n"
        f"Answer to evaluate:\n{answer}"
    )

    raw = ""
    try:
        raw = call_llm(system, user)
        parsed, parse_error = parse_judge_output(raw, fallback)
        return {"result": parsed, "raw": raw, "parse_error": parse_error}
    except Exception as exc:  # noqa: BLE001
        parsed = blank_quality_scores(f"Judge call failed: {exc}")
        return {"result": parsed, "raw": raw, "parse_error": str(exc)}



def judge_citation_accuracy(answer: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Score whether inline citations in the answer match retrieved sources."""
    fallback = blank_citation_scores("Parse error")
    source_str = build_source_summary(sources)

    system = (
        "You are a citation accuracy evaluator for a RAG system.\n\n"
        "You will receive an answer and a list of actually retrieved sources.\n"
        "Check whether the inline citations in the answer "
        "(for example [Source: filename.pdf | Page X]) match the retrieved source list.\n\n"
        "Score on two dimensions (each 0-10):\n"
        "  - Citation precision: what fraction of answer citations reference a source that was actually retrieved?\n"
        "  - Citation coverage: what fraction of retrieved sources are cited at least once?\n\n"
        "If the answer contains no inline citations at all, set both to 0.\n\n"
        "Return ONLY valid JSON:\n"
        '{"citation_precision": <int>, "citation_coverage": <int>, '
        '"citation_accuracy": <average>, "num_citations_found": <int>, '
        '"num_valid_citations": <int>, "feedback": "<one sentence>"}'
    )
    user = (
        f"Answer to evaluate:\n{answer}\n\n"
        f"Actually retrieved sources:\n{source_str}"
    )

    raw = ""
    try:
        raw = call_llm(system, user)
        parsed, parse_error = parse_judge_output(raw, fallback)
        return {"result": parsed, "raw": raw, "parse_error": parse_error}
    except Exception as exc:  # noqa: BLE001
        parsed = blank_citation_scores(f"Judge call failed: {exc}")
        return {"result": parsed, "raw": raw, "parse_error": str(exc)}



def judge_hallucination(case: Dict[str, Any], answer: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Detect unsupported claims in the answer."""
    fallback = blank_hallucination_scores("Parse error")
    source_str = build_source_summary(sources)

    system = (
        "You are a hallucination detector for a RAG system.\n\n"
        "A hallucination is a specific factual claim (number, name, method, finding, or conclusion) "
        "that appears in the answer but could not reasonably be derived from the listed retrieved sources.\n\n"
        "Do NOT flag:\n"
        "  - general transitional language\n"
        "  - hedging phrases such as 'the documents suggest'\n"
        "  - explicit statements that evidence is missing or insufficient\n\n"
        "Important: some questions are intentionally unanswerable. For answerable=false cases, "
        "a concise refusal grounded in insufficient evidence should score highly, while invented details should score poorly.\n\n"
        "Score (0-10):\n"
        "  10 = every factual claim is plausibly grounded in the sources\n"
        "   5 = some claims are unsupported but the core answer is grounded\n"
        "   0 = the answer is mostly fabricated\n\n"
        "Return ONLY valid JSON:\n"
        '{"hallucination_score": <int>, "num_claims_checked": <int>, '
        '"num_unsupported_claims": <int>, "flagged_claims": ["<claim 1>", "<claim 2>"], '
        '"feedback": "<one sentence>"}'
    )
    user = (
        f"Question type: {case.get('type')}\n"
        f"Answerable from documents: {case.get('answerable', True)}\n"
        f"Question: {case.get('query')}\n\n"
        f"Answer to evaluate:\n{answer}\n\n"
        f"Retrieved sources available to the system:\n{source_str}"
    )

    raw = ""
    try:
        raw = call_llm(system, user)
        parsed, parse_error = parse_judge_output(raw, fallback)
        return {"result": parsed, "raw": raw, "parse_error": parse_error}
    except Exception as exc:  # noqa: BLE001
        parsed = blank_hallucination_scores(f"Judge call failed: {exc}")
        return {"result": parsed, "raw": raw, "parse_error": str(exc)}



def judge_agreement(answer_a: str, answer_b: str) -> Dict[str, Any]:
    """Score how much two answers agree (0-10)."""
    fallback = {"agreement": 0, "note": "Parse error"}
    system = (
        "You compare two answers to the same question.\n"
        "Score agreement 0-10 (0 = contradictory, 10 = essentially the same content).\n"
        'Return ONLY JSON: {"agreement": <int>, "note": "<brief note>"}'
    )
    user = f"Answer A:\n{answer_a}\n\nAnswer B:\n{answer_b}"

    raw = ""
    try:
        raw = call_llm(system, user)
        parsed, parse_error = parse_judge_output(raw, fallback)
        return {"result": parsed, "raw": raw, "parse_error": parse_error}
    except Exception as exc:  # noqa: BLE001
        parsed = {"agreement": 0, "note": f"Judge call failed: {exc}"}
        return {"result": parsed, "raw": raw, "parse_error": str(exc)}


# ── Prepared pipeline runners ───────────────────────────────────


def prepare_manual() -> Dict[str, Any]:
    from src.pipelines.manual_pipeline import answer_question, build_manual_pipeline

    t0 = time.time()
    with usage_tracking_session() as prepare_usage:
        df_chunks, embeddings = build_manual_pipeline()
    prepare_time_s = time.time() - t0

    def run(query: str) -> PipelineResult:
        q0 = time.time()
        result = answer_question(query, df_chunks, embeddings, top_k=TOP_K)
        result.latency = time.time() - q0
        return result

    return {
        "run": run,
        "prepare_time_s": prepare_time_s,
        "prepare_usage": finalise_usage(prepare_usage),
    }



def prepare_langgraph() -> Dict[str, Any]:
    from src.pipelines.langgraph_agent import build_langgraph_pipeline, run_research_agent

    t0 = time.time()
    with usage_tracking_session() as prepare_usage:
        agent = build_langgraph_pipeline()
    prepare_time_s = time.time() - t0

    def run(query: str) -> PipelineResult:
        q0 = time.time()
        result = run_research_agent(agent, query, task_type="Question Answering", top_k=TOP_K)
        result.latency = time.time() - q0
        return result

    return {
        "run": run,
        "prepare_time_s": prepare_time_s,
        "prepare_usage": finalise_usage(prepare_usage),
    }



def prepare_llamaindex() -> Dict[str, Any]:
    from src.pipelines.llamaindex_pipeline import answer_question, build_index

    t0 = time.time()
    index = build_index(force_rebuild=False)
    prepare_time_s = time.time() - t0
    prepare_usage = normalise_usage(getattr(index, "_build_usage", empty_usage()))

    def run(query: str) -> PipelineResult:
        q0 = time.time()
        result = answer_question(query, index, top_k=TOP_K)
        result.latency = time.time() - q0
        return result

    return {
        "run": run,
        "prepare_time_s": prepare_time_s,
        "prepare_usage": prepare_usage,
    }


PIPELINE_PREPARERS: Dict[str, Callable[[], Dict[str, Any]]] = {
    "Manual": prepare_manual,
    "LangGraph": prepare_langgraph,
    "LlamaIndex": prepare_llamaindex,
}


# ── Full evaluation ─────────────────────────────────────────────


def run_full_evaluation(eval_cases: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Run all evaluation cases through all pipelines and judge each answer."""
    summary_rows: List[Dict[str, Any]] = []
    artifact_rows: List[Dict[str, Any]] = []

    prepared_runners: Dict[str, Dict[str, Any]] = {}
    first_query_seen = {pipe_name: False for pipe_name in PIPE_ORDER}

    print(f"\nPreparing {len(PIPELINE_PREPARERS)} pipelines...\n")
    for pipe_name in PIPE_ORDER:
        prepare_fn = PIPELINE_PREPARERS[pipe_name]
        print(f"  Preparing {pipe_name}...")
        t0 = time.time()
        prepared = prepare_fn()
        if "prepare_time_s" not in prepared:
            prepared["prepare_time_s"] = time.time() - t0
        prepared_runners[pipe_name] = prepared
        print(f"    Prepare time: {prepared['prepare_time_s']:.2f}s")
        prepare_usage = normalise_usage(prepared.get("prepare_usage"))
        if prepare_usage.get("total_api_tokens", 0) > 0 or prepare_usage.get("estimated_cost_usd", 0.0) > 0:
            print(
                f"    Prepare usage: {prepare_usage.get('total_api_tokens', 0):,} tokens | "
                f"${prepare_usage.get('estimated_cost_usd', 0.0):.5f}"
            )

    for case in eval_cases:
        qid = case["id"]
        query = case["query"]
        qtype = case["type"]
        answerable = case.get("answerable", True)

        print(f"\n{'=' * 72}\nEVALUATING {qid} ({qtype} | answerable={answerable}): {query[:70]}\n{'=' * 72}")

        answers: Dict[str, str] = {}

        for pipe_name in PIPE_ORDER:
            prepared = prepared_runners[pipe_name]
            pipe_fn = prepared["run"]
            prepare_time_s = safe_float(prepared.get("prepare_time_s"), 0.0)
            is_first_query = not first_query_seen[pipe_name]
            print(f"\n  Running {pipe_name}...")

            error_message = None
            try:
                result = pipe_fn(query)
            except Exception as exc:  # noqa: BLE001
                error_message = f"Pipeline error: {exc}"
                print(f"    [ERROR] {pipe_name}: {exc}")
                result = build_error_result(f"ERROR: {exc}")

            first_query_seen[pipe_name] = True
            answer = getattr(result, "answer", "") or ""
            query_time_s = safe_float(getattr(result, "latency", 0.0), 0.0)
            latency_s = query_time_s  # compatibility with existing charts / README
            source_dicts = normalise_sources(getattr(result, "source_dicts", []))
            metadata = pipeline_metadata(result)
            query_usage = normalise_usage(metadata.get("usage"))
            prepare_usage = normalise_usage(prepared.get("prepare_usage"))
            answers[pipe_name] = answer

            if error_message:
                quality_judge = {"result": blank_quality_scores(error_message), "raw": "", "parse_error": error_message}
                citation_judge = {"result": blank_citation_scores(error_message), "raw": "", "parse_error": error_message}
                halluc_judge = {"result": blank_hallucination_scores(error_message), "raw": "", "parse_error": error_message}
            else:
                quality_judge = judge_answer(case, answer)
                citation_judge = judge_citation_accuracy(answer, source_dicts)
                halluc_judge = judge_hallucination(case, answer, source_dicts)

            quality = quality_judge["result"]
            citation = citation_judge["result"]
            halluc = halluc_judge["result"]
            refusal_ok = detect_refusal(case, answer)

            print(f"    Quality: {quality.get('overall', '?')}/10 | Query latency: {latency_s:.2f}s")
            print(f"    Citation accuracy: {citation.get('citation_accuracy', '?')}/10")
            print(f"    Hallucination score: {halluc.get('hallucination_score', '?')}/10")
            if refusal_ok is not None:
                print(f"    Refusal OK: {refusal_ok}")
            if metadata:
                print(
                    f"    Retrieval: {metadata.get('retrieval_mode', 'unknown')} | "
                    f"Rerank: {metadata.get('rerank_enabled', 'unknown')} | "
                    f"Chunking: {metadata.get('chunking_style', 'unknown')}"
                )
                if metadata.get("task_kind") == "structured_extraction":
                    print(
                        f"    Schema valid: {metadata.get('schema_valid')} | "
                        f"Source: {metadata.get('schema_source')} | "
                        f"Error: {metadata.get('schema_error_type')}"
                    )
            if query_usage.get("total_api_tokens", 0) > 0 or query_usage.get("estimated_cost_usd", 0.0) > 0:
                print(
                    f"    Usage: prompt {query_usage.get('prompt_tokens', 0):,} | "
                    f"completion {query_usage.get('completion_tokens', 0):,} | "
                    f"embedding {query_usage.get('embedding_tokens', 0):,} | "
                    f"cost ${query_usage.get('estimated_cost_usd', 0.0):.5f}"
                )

            summary_rows.append({
                "row_type": "pipeline",
                "query_id": qid,
                "query": query,
                "query_type": qtype,
                "answerable": answerable,
                "pipeline": pipe_name,
                "answer": answer,
                "error": error_message,
                "prepare_time_s": round(prepare_time_s, 2),
                "query_time_s": round(query_time_s, 2),
                "latency_s": round(latency_s, 2),
                "is_first_query": is_first_query,
                "avg_similarity": round(safe_float(getattr(result, "avg_similarity", 0.0), 0.0), 4),
                "max_similarity": round(safe_float(getattr(result, "max_similarity", 0.0), 0.0), 4),
                "num_chunks": safe_int(getattr(result, "num_chunks", 0), 0),
                "source_count": len(source_dicts),
                "backend": metadata.get("backend"),
                "retrieval_mode": metadata.get("retrieval_mode"),
                "rerank_enabled": metadata.get("rerank_enabled"),
                "chunking_style": metadata.get("chunking_style"),
                "top_k": metadata.get("top_k"),
                "task_kind": metadata.get("task_kind"),
                "schema_name": metadata.get("schema_name"),
                "schema_valid": metadata.get("schema_valid"),
                "schema_source": metadata.get("schema_source"),
                "schema_error_type": metadata.get("schema_error_type"),
                "repair_attempted": metadata.get("repair_attempted"),
                "repair_succeeded": metadata.get("repair_succeeded"),
                "llm_calls": safe_int(query_usage.get("llm_calls", 0), 0),
                "embedding_calls": safe_int(query_usage.get("embedding_calls", 0), 0),
                "prompt_tokens": safe_int(query_usage.get("prompt_tokens", 0), 0),
                "cached_prompt_tokens": safe_int(query_usage.get("cached_prompt_tokens", 0), 0),
                "completion_tokens": safe_int(query_usage.get("completion_tokens", 0), 0),
                "llm_total_tokens": safe_int(query_usage.get("total_llm_tokens", 0), 0),
                "embedding_tokens": safe_int(query_usage.get("embedding_tokens", 0), 0),
                "total_api_tokens": safe_int(query_usage.get("total_api_tokens", 0), 0),
                "estimated_cost_usd": safe_float(query_usage.get("estimated_cost_usd", 0.0), 0.0),
                "prepare_llm_calls": safe_int(prepare_usage.get("llm_calls", 0), 0),
                "prepare_embedding_calls": safe_int(prepare_usage.get("embedding_calls", 0), 0),
                "prepare_prompt_tokens": safe_int(prepare_usage.get("prompt_tokens", 0), 0),
                "prepare_cached_prompt_tokens": safe_int(prepare_usage.get("cached_prompt_tokens", 0), 0),
                "prepare_completion_tokens": safe_int(prepare_usage.get("completion_tokens", 0), 0),
                "prepare_llm_total_tokens": safe_int(prepare_usage.get("total_llm_tokens", 0), 0),
                "prepare_embedding_tokens": safe_int(prepare_usage.get("embedding_tokens", 0), 0),
                "prepare_total_api_tokens": safe_int(prepare_usage.get("total_api_tokens", 0), 0),
                "prepare_estimated_cost_usd": safe_float(prepare_usage.get("estimated_cost_usd", 0.0), 0.0),
                "quality_score": metadata.get("quality_score"),
                "iterations": metadata.get("iterations"),
                "completeness": safe_float(quality.get("completeness", 0), 0.0),
                "grounding": safe_float(quality.get("grounding", 0), 0.0),
                "clarity": safe_float(quality.get("clarity", 0), 0.0),
                "overall": safe_float(quality.get("overall", 0), 0.0),
                "feedback": quality.get("feedback", ""),
                "citation_precision": safe_float(citation.get("citation_precision", 0), 0.0),
                "citation_coverage": safe_float(citation.get("citation_coverage", 0), 0.0),
                "citation_accuracy": safe_float(citation.get("citation_accuracy", 0), 0.0),
                "num_citations_found": safe_int(citation.get("num_citations_found", 0), 0),
                "num_valid_citations": safe_int(citation.get("num_valid_citations", 0), 0),
                "citation_feedback": citation.get("feedback", ""),
                "hallucination_score": safe_float(halluc.get("hallucination_score", 0), 0.0),
                "num_claims_checked": safe_int(halluc.get("num_claims_checked", 0), 0),
                "num_unsupported_claims": safe_int(halluc.get("num_unsupported_claims", 0), 0),
                "flagged_claims": json.dumps(halluc.get("flagged_claims", []), ensure_ascii=False),
                "hallucination_feedback": halluc.get("feedback", ""),
                "refusal_ok": refusal_ok,
            })

            artifact_rows.append({
                "artifact_type": "pipeline_answer",
                "query_id": qid,
                "query": query,
                "query_type": qtype,
                "answerable": answerable,
                "pipeline": pipe_name,
                "error": error_message,
                "notes": case.get("notes", ""),
                "timing": {
                    "prepare_time_s": round(prepare_time_s, 4),
                    "query_time_s": round(query_time_s, 4),
                    "latency_s": round(latency_s, 4),
                    "is_first_query": is_first_query,
                },
                "usage": {
                    "query": query_usage,
                    "prepare": prepare_usage,
                },
                "retrieval": {
                    "avg_similarity": round(safe_float(getattr(result, "avg_similarity", 0.0), 0.0), 6),
                    "max_similarity": round(safe_float(getattr(result, "max_similarity", 0.0), 0.0), 6),
                    "num_chunks": safe_int(getattr(result, "num_chunks", 0), 0),
                    "source_count": len(source_dicts),
                    "metadata": metadata,
                },
                "answer": answer,
                "sources": source_dicts,
                "refusal_ok": refusal_ok,
                "judges": {
                    "quality": quality_judge,
                    "citation_accuracy": citation_judge,
                    "hallucination": halluc_judge,
                },
            })

        # Cross-pipeline agreement per query
        for i in range(len(PIPE_ORDER)):
            for j in range(i + 1, len(PIPE_ORDER)):
                pipe_a = PIPE_ORDER[i]
                pipe_b = PIPE_ORDER[j]
                agreement_judge = judge_agreement(answers.get(pipe_a, ""), answers.get(pipe_b, ""))
                agreement = agreement_judge["result"]
                pair_name = f"AGREEMENT: {pipe_a} vs {pipe_b}"

                summary_rows.append({
                    "row_type": "agreement",
                    "query_id": qid,
                    "query": query,
                    "query_type": qtype,
                    "answerable": answerable,
                    "pipeline": pair_name,
                    "answer": "",
                    "error": None,
                    "prepare_time_s": 0,
                    "query_time_s": 0,
                    "latency_s": 0,
                    "is_first_query": False,
                    "avg_similarity": 0,
                    "max_similarity": 0,
                    "num_chunks": 0,
                    "source_count": 0,
                    "backend": None,
                    "retrieval_mode": None,
                    "rerank_enabled": None,
                    "chunking_style": None,
                    "top_k": None,
                    "task_kind": None,
                    "schema_name": None,
                    "schema_valid": None,
                    "schema_source": None,
                    "schema_error_type": None,
                    "repair_attempted": None,
                    "repair_succeeded": None,
                    "llm_calls": 0,
                    "embedding_calls": 0,
                    "prompt_tokens": 0,
                    "cached_prompt_tokens": 0,
                    "completion_tokens": 0,
                    "llm_total_tokens": 0,
                    "embedding_tokens": 0,
                    "total_api_tokens": 0,
                    "estimated_cost_usd": 0.0,
                    "prepare_llm_calls": 0,
                    "prepare_embedding_calls": 0,
                    "prepare_prompt_tokens": 0,
                    "prepare_cached_prompt_tokens": 0,
                    "prepare_completion_tokens": 0,
                    "prepare_llm_total_tokens": 0,
                    "prepare_embedding_tokens": 0,
                    "prepare_total_api_tokens": 0,
                    "prepare_estimated_cost_usd": 0.0,
                    "quality_score": None,
                    "iterations": None,
                    "completeness": 0,
                    "grounding": 0,
                    "clarity": 0,
                    "overall": safe_float(agreement.get("agreement", 0), 0.0),
                    "feedback": agreement.get("note", ""),
                    "citation_precision": 0,
                    "citation_coverage": 0,
                    "citation_accuracy": 0,
                    "num_citations_found": 0,
                    "num_valid_citations": 0,
                    "citation_feedback": "",
                    "hallucination_score": 0,
                    "num_claims_checked": 0,
                    "num_unsupported_claims": 0,
                    "flagged_claims": "[]",
                    "hallucination_feedback": "",
                    "refusal_ok": None,
                })

                artifact_rows.append({
                    "artifact_type": "agreement",
                    "query_id": qid,
                    "query": query,
                    "query_type": qtype,
                    "answerable": answerable,
                    "pipeline_pair": [pipe_a, pipe_b],
                    "answer_a": answers.get(pipe_a, ""),
                    "answer_b": answers.get(pipe_b, ""),
                    "judge": agreement_judge,
                })

    return pd.DataFrame(summary_rows), artifact_rows


# ── Charts ──────────────────────────────────────────────────────


def present_query_type_order(df_pipes: pd.DataFrame) -> List[str]:
    present = set(df_pipes["query_type"].dropna().tolist())
    ordered = [qtype for qtype in DEFAULT_QUERY_TYPE_ORDER if qtype in present]
    extras = sorted(present - set(ordered))
    return ordered + extras



def generate_charts(df: pd.DataFrame) -> None:
    """Generate benchmark charts."""
    df_pipes = df[df["row_type"] == "pipeline"].copy()
    df_agree = df[df["row_type"] == "agreement"].copy()

    # Chart 1: Overall Quality
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = df_pipes.groupby("pipeline")["overall"].mean().reindex(PIPE_ORDER)
    bars = ax.bar(pivot.index, pivot.values, color=[COLOURS[p] for p in pivot.index], edgecolor="white", linewidth=1.2)
    ax.set_ylabel("Average Overall Score (0-10)")
    ax.set_title("Overall Answer Quality by Pipeline", fontweight="bold")
    ax.set_ylim(0, 10)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, pivot.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, f"{val:.1f}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(EVAL_OUTPUT_PATH / "01_overall_quality.png", dpi=150)
    plt.close()

    # Chart 2: Quality Breakdown
    fig, ax = plt.subplots(figsize=(14, 5))
    dims = ["completeness", "grounding", "clarity", "citation_accuracy", "hallucination_score"]
    dim_labels = ["Completeness", "Grounding", "Clarity", "Citation\nAccuracy", "Hallucination\nFreedom"]
    x = np.arange(len(dims))
    width = 0.25
    for i, pipe in enumerate(PIPE_ORDER):
        pipe_df = df_pipes[df_pipes["pipeline"] == pipe]
        vals = [pipe_df[d].mean() for d in dims]
        bars = ax.bar(x + i * width, vals, width, label=pipe, color=COLOURS[pipe], edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{val:.1f}", ha="center", fontsize=8)
    ax.set_ylabel("Average Score (0-10)")
    ax.set_title("Answer Quality Breakdown", fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(dim_labels)
    ax.set_ylim(0, 10)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(EVAL_OUTPUT_PATH / "02_quality_breakdown.png", dpi=150)
    plt.close()

    # Chart 3: Latency (query time only)
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot_lat = df_pipes.groupby("pipeline")["query_time_s"].mean().reindex(PIPE_ORDER)
    bars = ax.bar(pivot_lat.index, pivot_lat.values, color=[COLOURS[p] for p in pivot_lat.index], edgecolor="white")
    ax.set_ylabel("Average Query Latency (seconds)")
    ax.set_title("Query Latency by Pipeline", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, pivot_lat.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{val:.1f}s", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(EVAL_OUTPUT_PATH / "03_latency.png", dpi=150)
    plt.close()

    # Chart 4: Per-Query Scores
    fig, ax = plt.subplots(figsize=(14, 5))
    query_ids = df_pipes["query_id"].unique()
    x = np.arange(len(query_ids))
    width = 0.25
    for i, pipe in enumerate(PIPE_ORDER):
        pipe_df = df_pipes[df_pipes["pipeline"] == pipe]
        vals = [pipe_df[pipe_df["query_id"] == qid]["overall"].values[0] if len(pipe_df[pipe_df["query_id"] == qid]) > 0 else 0 for qid in query_ids]
        ax.bar(x + i * width, vals, width, label=pipe, color=COLOURS[pipe], edgecolor="white")
    ax.set_ylabel("Overall Score (0-10)")
    ax.set_title("Per-Query Answer Quality", fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(query_ids, rotation=45)
    ax.set_xlabel("Query ID")
    ax.set_ylim(0, 10)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(EVAL_OUTPUT_PATH / "04_per_query.png", dpi=150)
    plt.close()

    # Chart 5: Agreement
    if len(df_agree) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        agree_avg = df_agree.groupby("pipeline")["overall"].mean()
        pair_colours = ["#9D755D", "#B07AA1", "#76B7B2"]
        bars = ax.bar(range(len(agree_avg)), agree_avg.values, color=pair_colours[:len(agree_avg)], edgecolor="white")
        ax.set_ylabel("Average Agreement Score (0-10)")
        ax.set_title("Cross-Pipeline Answer Agreement", fontweight="bold")
        labels = [s.replace("AGREEMENT: ", "") for s in agree_avg.index]
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 10)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, agree_avg.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, f"{val:.1f}", ha="center", fontweight="bold")
        plt.tight_layout()
        plt.savefig(EVAL_OUTPUT_PATH / "05_agreement.png", dpi=150)
        plt.close()

    # Chart 6: Radar
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    categories = ["Completeness", "Grounding", "Clarity", "Citation\nAccuracy", "Hallucination\nFreedom", "Speed"]
    num_categories = len(categories)
    angles = [n / float(num_categories) * 2 * np.pi for n in range(num_categories)]
    angles += angles[:1]
    for pipe in PIPE_ORDER:
        pipe_df = df_pipes[df_pipes["pipeline"] == pipe]
        vals = [
            pipe_df["completeness"].mean(),
            pipe_df["grounding"].mean(),
            pipe_df["clarity"].mean(),
            pipe_df["citation_accuracy"].mean(),
            pipe_df["hallucination_score"].mean(),
            10 - min(pipe_df["query_time_s"].mean(), 10),
        ]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=pipe, color=COLOURS[pipe])
        ax.fill(angles, vals, alpha=0.1, color=COLOURS[pipe])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 10)
    ax.set_title("Pipeline Profile Comparison", fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), frameon=False)
    plt.tight_layout()
    plt.savefig(EVAL_OUTPUT_PATH / "06_radar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Chart 7: Citation & Hallucination Detail
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    x = np.arange(len(PIPE_ORDER))
    width = 0.35
    prec = [df_pipes[df_pipes["pipeline"] == p]["citation_precision"].mean() for p in PIPE_ORDER]
    cov = [df_pipes[df_pipes["pipeline"] == p]["citation_coverage"].mean() for p in PIPE_ORDER]
    bars1 = ax.bar(x - width / 2, prec, width, label="Precision", color="#4C78A8", edgecolor="white")
    bars2 = ax.bar(x + width / 2, cov, width, label="Coverage", color="#72B7B2", edgecolor="white")
    for bar, val in zip(bars1, prec):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, f"{val:.1f}", ha="center", fontsize=9)
    for bar, val in zip(bars2, cov):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, f"{val:.1f}", ha="center", fontsize=9)
    ax.set_ylabel("Score (0-10)")
    ax.set_title("Citation Precision vs Coverage", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(PIPE_ORDER)
    ax.set_ylim(0, 10)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    halluc_scores = [df_pipes[df_pipes["pipeline"] == p]["hallucination_score"].mean() for p in PIPE_ORDER]
    unsup_counts = [df_pipes[df_pipes["pipeline"] == p]["num_unsupported_claims"].mean() for p in PIPE_ORDER]
    bars = ax.bar(PIPE_ORDER, halluc_scores, color=[COLOURS[p] for p in PIPE_ORDER], edgecolor="white")
    for bar, val, count in zip(bars, halluc_scores, unsup_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            f"{val:.1f}\n({count:.1f} flagged)",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_ylabel("Hallucination Freedom (0-10)")
    ax.set_title("Hallucination Detection", fontweight="bold")
    ax.set_ylim(0, 10)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(EVAL_OUTPUT_PATH / "07_citation_hallucination.png", dpi=150)
    plt.close()

    # Chart 8: Query-type quality breakdown
    query_type_order = present_query_type_order(df_pipes)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(query_type_order))
    width = 0.25
    for i, pipe in enumerate(PIPE_ORDER):
        pipe_df = df_pipes[df_pipes["pipeline"] == pipe]
        vals = [
            pipe_df[pipe_df["query_type"] == qtype]["overall"].mean() if len(pipe_df[pipe_df["query_type"] == qtype]) > 0 else 0
            for qtype in query_type_order
        ]
        bars = ax.bar(x + i * width, vals, width, label=pipe, color=COLOURS[pipe], edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{val:.1f}", ha="center", fontsize=8)
    ax.set_ylabel("Average Overall Score (0-10)")
    ax.set_title("Average Quality by Query Type", fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([q.title() for q in query_type_order])
    ax.set_ylim(0, 10)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(EVAL_OUTPUT_PATH / "08_query_type_quality.png", dpi=150)
    plt.close()

    # Chart 9: Average estimated query cost
    fig, ax = plt.subplots(figsize=(8, 5))
    cost_pivot = df_pipes.groupby("pipeline")["estimated_cost_usd"].mean().reindex(PIPE_ORDER)
    bars = ax.bar(cost_pivot.index, cost_pivot.values, color=[COLOURS[p] for p in cost_pivot.index], edgecolor="white", linewidth=1.2)
    ax.set_ylabel("Average Estimated Query Cost (USD)")
    ax.set_title("Average Query Cost by Pipeline", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ymax = max(float(cost_pivot.max() or 0.0) * 1.15, 0.00001)
    ax.set_ylim(0, ymax)
    for bar, val in zip(bars, cost_pivot.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.02, f"${val:.5f}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(EVAL_OUTPUT_PATH / "09_query_cost.png", dpi=150)
    plt.close()

    print(f"  Charts saved to {EVAL_OUTPUT_PATH}/")


# ── Console summary ─────────────────────────────────────────────


def print_summary(df: pd.DataFrame) -> None:
    """Print benchmark summary tables to console."""
    df_pipes = df[df["row_type"] == "pipeline"]
    df_agree = df[df["row_type"] == "agreement"]

    print(f"\n{'=' * 72}\nEVALUATION SUMMARY\n{'=' * 72}")
    summary = df_pipes.groupby("pipeline").agg({
        "overall": "mean",
        "completeness": "mean",
        "grounding": "mean",
        "clarity": "mean",
        "citation_accuracy": "mean",
        "hallucination_score": "mean",
        "query_time_s": "mean",
        "prepare_time_s": "mean",
        "estimated_cost_usd": "mean",
        "prepare_estimated_cost_usd": "mean",
        "total_api_tokens": "mean",
    }).reindex(PIPE_ORDER)
    summary.columns = [
        "Overall",
        "Complete",
        "Grounded",
        "Clarity",
        "Citation",
        "Halluc.Free",
        "Query(s)",
        "Prepare(s)",
        "QueryCost($)",
        "PrepCost($)",
        "Tokens",
    ]
    for col in ["Overall", "Complete", "Grounded", "Clarity", "Citation", "Halluc.Free"]:
        summary[col] = summary[col].round(1)
    summary["Query(s)"] = summary["Query(s)"].round(2)
    summary["Prepare(s)"] = summary["Prepare(s)"].round(2)
    summary["QueryCost($)"] = summary["QueryCost($)"].round(5)
    summary["PrepCost($)"] = summary["PrepCost($)"].round(5)
    summary["Tokens"] = summary["Tokens"].round(0).astype(int)
    print(f"\n{summary.to_string()}")

    print(f"\n{'-' * 72}\nBEST PIPELINE PER DIMENSION\n{'-' * 72}")
    for col in ["Overall", "Complete", "Grounded", "Clarity", "Citation", "Halluc.Free"]:
        best = summary[col].idxmax()
        print(f"  {col:12s}: {best} ({summary.loc[best, col]})")
    fastest = summary["Query(s)"].idxmin()
    cheapest = summary["QueryCost($)"].idxmin()
    print(f"  {'Speed':12s}: {fastest} ({summary.loc[fastest, 'Query(s)']}s)")
    print(f"  {'Cost':12s}: {cheapest} (${summary.loc[cheapest, 'QueryCost($)']})")

    print(f"\n{'-' * 72}\nQUALITY BY QUERY TYPE\n{'-' * 72}")
    query_type_summary = (
        df_pipes.groupby(["query_type", "pipeline"])["overall"]
        .mean()
        .unstack("pipeline")
        .reindex(present_query_type_order(df_pipes))
        .reindex(columns=PIPE_ORDER)
        .round(1)
    )
    print(query_type_summary.to_string())

    if df_pipes["answerable"].eq(False).any():
        refusal_df = df_pipes[df_pipes["answerable"] == False].copy()  # noqa: E712
        refusal_summary = refusal_df.groupby("pipeline")["refusal_ok"].mean().reindex(PIPE_ORDER).fillna(0).round(2)
        print(f"\n{'-' * 72}\nUNANSWERABLE CASES — REFUSAL RATE\n{'-' * 72}")
        for pipe, val in refusal_summary.items():
            print(f"  {pipe:12s}: {val:.2f}")

    if len(df_agree) > 0:
        print(f"\n{'-' * 72}\nAGREEMENT SUMMARY\n{'-' * 72}")
        agree_summary = df_agree.groupby("pipeline")["overall"].mean().round(1)
        for pair_name, score in agree_summary.items():
            print(f"  {pair_name.replace('AGREEMENT: ', ''):28s}: {score}")

    print(f"\n{'-' * 72}\nHALLUCINATION DETAIL\n{'-' * 72}")
    for pipe in PIPE_ORDER:
        pipe_df = df_pipes[df_pipes["pipeline"] == pipe]
        avg_unsup = pipe_df["num_unsupported_claims"].mean()
        avg_checked = pipe_df["num_claims_checked"].mean()
        print(f"  {pipe:12s}: {avg_unsup:.1f} unsupported claims / {avg_checked:.1f} checked (avg per query)")

    print(f"\n{'=' * 72}\n")


# ── Main ────────────────────────────────────────────────────────


def main() -> None:
    eval_cases = load_eval_cases()
    EVAL_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    print(f"\nSTARTING EVALUATION — {len(eval_cases)} cases × {len(PIPE_ORDER)} pipelines\n")
    print(f"Case file: {EVAL_CASES_PATH}")

    df_results, artifact_rows = run_full_evaluation(eval_cases)

    df_results.to_csv(CSV_OUTPUT_PATH, index=False)
    write_jsonl(JSONL_OUTPUT_PATH, artifact_rows)

    print(f"\n[OK] Summary CSV saved to {CSV_OUTPUT_PATH}")
    print(f"[OK] Rich artifacts saved to {JSONL_OUTPUT_PATH}")

    print_summary(df_results)

    print("Generating charts...")
    generate_charts(df_results)

    print("\nEVALUATION COMPLETE")
    print(f"  CSV:    {CSV_OUTPUT_PATH}")
    print(f"  JSONL:  {JSONL_OUTPUT_PATH}")
    print(f"  Charts: {EVAL_OUTPUT_PATH}/*.png")


if __name__ == "__main__":
    main()
