# -*- coding: utf-8 -*-
"""
RAG Pipeline Evaluation — Head-to-Head Comparison

Compares Manual Pipeline vs LangGraph Agent vs LlamaIndex across:
  - Answer quality (LLM-as-judge: completeness, grounding, clarity)
  - Citation accuracy (do inline citations match retrieved sources?)
  - Hallucination detection (are there unsupported claims?)
  - Retrieval metrics (similarity scores, chunk counts)
  - Latency per query
  - Cross-pipeline agreement

Run:  python -m src.evaluation.evaluate_pipelines

Author: Tingyi Zhang
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

from src.core.llm import call_llm
from src.core.types import PipelineResult
from src.config import EVAL_OUTPUT_DIR, TOP_K

# ── Test queries ────────────────────────────────────────────────

TEST_QUERIES = [
    {"id": "Q1", "query": "What experimental setup and measurement techniques were used?", "type": "factual"},
    {"id": "Q2", "query": "What are the main findings and conclusions of this research?", "type": "summary"},
    {"id": "Q3", "query": "Describe the data acquisition and signal processing methodology.", "type": "technical"},
    {"id": "Q4", "query": "What calibration procedures were applied to the instruments?", "type": "factual"},
    {"id": "Q5", "query": "What are the limitations or sources of uncertainty in this study?", "type": "critical"},
]

COLOURS = {"Manual": "#4C78A8", "LangGraph": "#F58518", "LlamaIndex": "#54A24B"}
PIPE_ORDER = ["Manual", "LangGraph", "LlamaIndex"]

# ── LLM-as-Judge ────────────────────────────────────────────────

def judge_answer(query, answer):
    """Score an answer on completeness, grounding, and clarity (0-10 each)."""
    system = (
        "You are a strict evaluator for technical document Q&A systems.\n\n"
        "Score on three dimensions (each 0-10):\n"
        "  - Completeness: Does it address all parts of the question?\n"
        "  - Grounding: Does it rely on retrieved evidence?\n"
        "  - Clarity: Is it well-structured and easy to follow?\n\n"
        "Return ONLY valid JSON:\n"
        '{"completeness": <int>, "grounding": <int>, "clarity": <int>, '
        '"overall": <average>, "feedback": "<one sentence>"}'
    )
    user = f"Question: {query}\n\nAnswer to evaluate:\n{answer}"

    try:
        raw = call_llm(system, user)
        cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(cleaned)
    except Exception as e:
        return {"completeness": 0, "grounding": 0, "clarity": 0, "overall": 0, "feedback": f"Parse error: {e}"}


def judge_citation_accuracy(answer, sources):
    """
    Score whether inline citations in the answer match actually retrieved sources.

    Checks two things:
      - Do the cited filenames and page numbers appear in the retrieved source list?
      - Are there claims with citations that don't match any retrieved source?

    Returns citation_accuracy (0-10) and details.
    """
    source_str = "\n".join(
        f"  - {s['file']} | Page {s['page']}"
        for s in sources
    ) if sources else "  (no sources retrieved)"

    system = (
        "You are a citation accuracy evaluator for a RAG system.\n\n"
        "You will receive an answer and a list of actually retrieved sources.\n"
        "Your job is to check whether the inline citations in the answer "
        "(e.g. [Source: filename.pdf | Page X]) match the retrieved source list.\n\n"
        "Score on two dimensions (each 0-10):\n"
        "  - Citation precision: What fraction of citations in the answer "
        "reference a source that was actually retrieved? "
        "(10 = all citations match, 0 = none match)\n"
        "  - Citation coverage: What fraction of retrieved sources are cited "
        "at least once? (10 = all sources cited, 0 = none cited)\n\n"
        "If the answer contains NO inline citations at all, "
        "citation_precision = 0 and citation_coverage = 0.\n\n"
        "Return ONLY valid JSON:\n"
        '{"citation_precision": <int>, "citation_coverage": <int>, '
        '"citation_accuracy": <average of the two>, '
        '"num_citations_found": <int>, "num_valid_citations": <int>, '
        '"feedback": "<one sentence>"}'
    )
    user = (
        f"Answer to evaluate:\n{answer}\n\n"
        f"Actually retrieved sources:\n{source_str}"
    )

    try:
        raw = call_llm(system, user)
        cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(cleaned)
    except Exception as e:
        return {
            "citation_precision": 0, "citation_coverage": 0,
            "citation_accuracy": 0, "num_citations_found": 0,
            "num_valid_citations": 0, "feedback": f"Parse error: {e}",
        }


def judge_hallucination(query, answer, sources):
    """
    Detect unsupported claims (hallucinations) in the answer.

    The judge checks whether the answer contains specific factual claims
    that could not reasonably be derived from the listed retrieved sources.

    Returns hallucination_score (0-10, where 10 = no hallucinations)
    and a list of flagged claims.
    """
    source_str = "\n".join(
        f"  - {s['file']} | Page {s['page']}"
        for s in sources
    ) if sources else "  (no sources retrieved)"

    system = (
        "You are a hallucination detector for a RAG system.\n\n"
        "A RAG system retrieves document chunks and generates an answer. "
        "Your job is to identify claims in the answer that are NOT supported "
        "by the retrieved sources.\n\n"
        "A hallucination is a specific factual claim (a number, name, method, "
        "finding, or conclusion) that appears in the answer but could not "
        "reasonably be derived from the listed source documents.\n\n"
        "Do NOT flag:\n"
        "  - General transitional or structural language\n"
        "  - Hedging phrases like 'the documents suggest'\n"
        "  - Explicit statements of insufficient evidence\n\n"
        "Score (0-10):\n"
        "  10 = Every factual claim is plausibly grounded in the sources\n"
        "   5 = Some claims are unsupported but the core answer is grounded\n"
        "   0 = The answer is mostly fabricated\n\n"
        "Return ONLY valid JSON:\n"
        '{"hallucination_score": <int>, '
        '"num_claims_checked": <int>, '
        '"num_unsupported_claims": <int>, '
        '"flagged_claims": ["<claim 1>", "<claim 2>"], '
        '"feedback": "<one sentence>"}'
    )
    user = (
        f"Question: {query}\n\n"
        f"Answer to evaluate:\n{answer}\n\n"
        f"Retrieved sources available to the system:\n{source_str}"
    )

    try:
        raw = call_llm(system, user)
        cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(cleaned)
    except Exception as e:
        return {
            "hallucination_score": 0, "num_claims_checked": 0,
            "num_unsupported_claims": 0, "flagged_claims": [],
            "feedback": f"Parse error: {e}",
        }


def judge_agreement(answer_a, answer_b):
    """Score how much two answers agree (0-10)."""
    system = (
        "You compare two answers to the same question.\n"
        "Score agreement 0-10 (0=contradictory, 10=same content).\n"
        'Return ONLY JSON: {"agreement": <int>, "note": "<brief note>"}'
    )
    user = f"Answer A:\n{answer_a}\n\nAnswer B:\n{answer_b}"

    try:
        raw = call_llm(system, user)
        cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(cleaned)
    except Exception:
        return {"agreement": 0, "note": "Parse error"}


# ── Pipeline runners — all return PipelineResult ────────────────

def run_manual(query) -> PipelineResult:
    from src.pipelines.manual_pipeline import build_manual_pipeline, answer_question
    df_chunks, embeddings = build_manual_pipeline()

    t0 = time.time()
    result = answer_question(query, df_chunks, embeddings, top_k=TOP_K)
    result.latency = time.time() - t0
    return result


def run_langgraph(query) -> PipelineResult:
    from src.pipelines.langgraph_agent import build_langgraph_pipeline, run_research_agent
    agent = build_langgraph_pipeline()

    t0 = time.time()
    result = run_research_agent(agent, query, task_type="Question Answering", top_k=TOP_K)
    result.latency = time.time() - t0
    return result


def run_llamaindex(query) -> PipelineResult:
    from src.pipelines.llamaindex_pipeline import build_index, answer_question
    index = build_index(force_rebuild=False)

    t0 = time.time()
    result = answer_question(query, index, top_k=TOP_K)
    result.latency = time.time() - t0
    return result


PIPELINES = {"Manual": run_manual, "LangGraph": run_langgraph, "LlamaIndex": run_llamaindex}


# ── Full evaluation ─────────────────────────────────────────────

def run_full_evaluation():
    """Run all test queries through all pipelines and judge each answer."""
    all_results = []

    for q_info in TEST_QUERIES:
        qid, query, qtype = q_info["id"], q_info["query"], q_info["type"]
        print(f"\n{'='*60}\nEVALUATING {qid}: {query[:60]}...\n{'='*60}")

        answers = {}
        for pipe_name, pipe_fn in PIPELINES.items():
            print(f"\n  Running {pipe_name}...")
            try:
                result = pipe_fn(query)
                answer = result.answer
                latency = result.latency
                source_dicts = result.source_dicts
            except Exception as e:
                print(f"    [ERROR] {pipe_name}: {e}")
                answer, latency = f"ERROR: {e}", 0
                source_dicts = []
                result = PipelineResult(answer=answer)

            answers[pipe_name] = answer

            # Quality scoring
            scores = judge_answer(query, answer)
            print(f"    Quality: {scores.get('overall', '?')}/10 | Latency: {latency:.2f}s")

            # Citation accuracy
            citation = judge_citation_accuracy(answer, source_dicts)
            print(f"    Citation accuracy: {citation.get('citation_accuracy', '?')}/10")

            # Hallucination detection
            halluc = judge_hallucination(query, answer, source_dicts)
            print(f"    Hallucination score: {halluc.get('hallucination_score', '?')}/10")

            row = {
                "query_id": qid, "query": query, "query_type": qtype,
                "pipeline": pipe_name, "answer": answer[:500],
                "latency_s": round(latency, 2),
                "avg_similarity": round(result.avg_similarity, 4),
                "max_similarity": round(result.max_similarity, 4),
                "num_chunks": result.num_chunks,
                "quality_score": result.metadata.get("quality_score") if result.metadata else None,
                "iterations": result.metadata.get("iterations") if result.metadata else None,
                # Quality dimensions
                "completeness": scores.get("completeness", 0),
                "grounding": scores.get("grounding", 0),
                "clarity": scores.get("clarity", 0),
                "overall": scores.get("overall", 0),
                "feedback": scores.get("feedback", ""),
                # Citation accuracy
                "citation_precision": citation.get("citation_precision", 0),
                "citation_coverage": citation.get("citation_coverage", 0),
                "citation_accuracy": citation.get("citation_accuracy", 0),
                "num_citations_found": citation.get("num_citations_found", 0),
                "num_valid_citations": citation.get("num_valid_citations", 0),
                "citation_feedback": citation.get("feedback", ""),
                # Hallucination
                "hallucination_score": halluc.get("hallucination_score", 0),
                "num_claims_checked": halluc.get("num_claims_checked", 0),
                "num_unsupported_claims": halluc.get("num_unsupported_claims", 0),
                "flagged_claims": json.dumps(halluc.get("flagged_claims", [])),
                "hallucination_feedback": halluc.get("feedback", ""),
            }
            all_results.append(row)

        # Cross-pipeline agreement
        pipe_names = list(answers.keys())
        for i in range(len(pipe_names)):
            for j in range(i + 1, len(pipe_names)):
                a, b = pipe_names[i], pipe_names[j]
                agreement = judge_agreement(answers[a], answers[b])
                all_results.append({
                    "query_id": qid, "query": query, "query_type": qtype,
                    "pipeline": f"AGREEMENT: {a} vs {b}",
                    "answer": "", "latency_s": 0,
                    "avg_similarity": 0, "max_similarity": 0, "num_chunks": 0,
                    "quality_score": None, "iterations": None,
                    "completeness": 0, "grounding": 0, "clarity": 0,
                    "overall": agreement.get("agreement", 0),
                    "feedback": agreement.get("note", ""),
                    "citation_precision": 0, "citation_coverage": 0,
                    "citation_accuracy": 0, "num_citations_found": 0,
                    "num_valid_citations": 0, "citation_feedback": "",
                    "hallucination_score": 0, "num_claims_checked": 0,
                    "num_unsupported_claims": 0, "flagged_claims": "[]",
                    "hallucination_feedback": "",
                })

    return pd.DataFrame(all_results)


# ── Charts ──────────────────────────────────────────────────────

def generate_charts(df):
    """Generate all comparison charts."""
    df_pipes = df[~df["pipeline"].str.startswith("AGREEMENT")].copy()
    df_agree = df[df["pipeline"].str.startswith("AGREEMENT")].copy()

    # Chart 1: Overall Quality
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = df_pipes.groupby("pipeline")["overall"].mean().reindex(PIPE_ORDER)
    bars = ax.bar(pivot.index, pivot.values, color=[COLOURS[p] for p in pivot.index], edgecolor="white", linewidth=1.2)
    ax.set_ylabel("Average Overall Score (0-10)")
    ax.set_title("Overall Answer Quality by Pipeline", fontweight="bold")
    ax.set_ylim(0, 10)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, pivot.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.15, f"{val:.1f}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{EVAL_OUTPUT_DIR}/01_overall_quality.png", dpi=150)
    plt.close()

    # Chart 2: Quality Breakdown (includes citation + hallucination)
    fig, ax = plt.subplots(figsize=(14, 5))
    dims = ["completeness", "grounding", "clarity", "citation_accuracy", "hallucination_score"]
    dim_labels = ["Completeness", "Grounding", "Clarity", "Citation\nAccuracy", "Hallucination\nFreedom"]
    x = np.arange(len(dims))
    width = 0.25
    for i, pipe in enumerate(PIPE_ORDER):
        pipe_df = df_pipes[df_pipes["pipeline"] == pipe]
        vals = [pipe_df[d].mean() for d in dims]
        bars = ax.bar(x + i*width, vals, width, label=pipe, color=COLOURS[pipe], edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.1, f"{val:.1f}", ha="center", fontsize=8)
    ax.set_ylabel("Average Score (0-10)")
    ax.set_title("Answer Quality Breakdown", fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(dim_labels)
    ax.set_ylim(0, 10)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{EVAL_OUTPUT_DIR}/02_quality_breakdown.png", dpi=150)
    plt.close()

    # Chart 3: Latency
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot_lat = df_pipes.groupby("pipeline")["latency_s"].mean().reindex(PIPE_ORDER)
    bars = ax.bar(pivot_lat.index, pivot_lat.values, color=[COLOURS[p] for p in pivot_lat.index], edgecolor="white")
    ax.set_ylabel("Average Latency (seconds)")
    ax.set_title("Query Latency by Pipeline", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, pivot_lat.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.1, f"{val:.1f}s", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{EVAL_OUTPUT_DIR}/03_latency.png", dpi=150)
    plt.close()

    # Chart 4: Per-Query Scores
    fig, ax = plt.subplots(figsize=(12, 5))
    query_ids = df_pipes["query_id"].unique()
    x = np.arange(len(query_ids))
    width = 0.25
    for i, pipe in enumerate(PIPE_ORDER):
        pipe_df = df_pipes[df_pipes["pipeline"] == pipe]
        vals = [pipe_df[pipe_df["query_id"]==qid]["overall"].values[0] if len(pipe_df[pipe_df["query_id"]==qid]) > 0 else 0 for qid in query_ids]
        ax.bar(x + i*width, vals, width, label=pipe, color=COLOURS[pipe], edgecolor="white")
    ax.set_ylabel("Overall Score (0-10)")
    ax.set_title("Per-Query Answer Quality", fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(query_ids)
    ax.set_xlabel("Query ID")
    ax.set_ylim(0, 10)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{EVAL_OUTPUT_DIR}/04_per_query.png", dpi=150)
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
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.15, f"{val:.1f}", ha="center", fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{EVAL_OUTPUT_DIR}/05_agreement.png", dpi=150)
        plt.close()

    # Chart 6: Radar (includes citation accuracy and hallucination)
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    categories = ["Completeness", "Grounding", "Clarity", "Citation\nAccuracy", "Hallucination\nFreedom", "Speed"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    for pipe in PIPE_ORDER:
        pipe_df = df_pipes[df_pipes["pipeline"] == pipe]
        vals = [
            pipe_df["completeness"].mean(),
            pipe_df["grounding"].mean(),
            pipe_df["clarity"].mean(),
            pipe_df["citation_accuracy"].mean(),
            pipe_df["hallucination_score"].mean(),
            10 - min(pipe_df["latency_s"].mean(), 10),
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
    plt.savefig(f"{EVAL_OUTPUT_DIR}/06_radar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Chart 7: Citation & Hallucination Detail
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 7a: Citation precision vs coverage
    ax = axes[0]
    x = np.arange(len(PIPE_ORDER))
    width = 0.35
    prec = [df_pipes[df_pipes["pipeline"] == p]["citation_precision"].mean() for p in PIPE_ORDER]
    cov = [df_pipes[df_pipes["pipeline"] == p]["citation_coverage"].mean() for p in PIPE_ORDER]
    bars1 = ax.bar(x - width/2, prec, width, label="Precision", color="#4C78A8", edgecolor="white")
    bars2 = ax.bar(x + width/2, cov, width, label="Coverage", color="#72B7B2", edgecolor="white")
    for bar, val in zip(bars1, prec):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.15, f"{val:.1f}", ha="center", fontsize=9)
    for bar, val in zip(bars2, cov):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.15, f"{val:.1f}", ha="center", fontsize=9)
    ax.set_ylabel("Score (0-10)")
    ax.set_title("Citation Precision vs Coverage", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(PIPE_ORDER)
    ax.set_ylim(0, 10)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    # 7b: Hallucination score + unsupported claim count
    ax = axes[1]
    halluc_scores = [df_pipes[df_pipes["pipeline"] == p]["hallucination_score"].mean() for p in PIPE_ORDER]
    unsup_counts = [df_pipes[df_pipes["pipeline"] == p]["num_unsupported_claims"].mean() for p in PIPE_ORDER]
    bars = ax.bar(PIPE_ORDER, halluc_scores, color=[COLOURS[p] for p in PIPE_ORDER], edgecolor="white")
    for bar, val, count in zip(bars, halluc_scores, unsup_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.15,
                f"{val:.1f}\n({count:.1f} flagged)", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Hallucination Freedom (0-10)")
    ax.set_title("Hallucination Detection", fontweight="bold")
    ax.set_ylim(0, 10)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{EVAL_OUTPUT_DIR}/07_citation_hallucination.png", dpi=150)
    plt.close()

    print(f"  Charts saved to {EVAL_OUTPUT_DIR}/")


def print_summary(df):
    """Print summary table to console."""
    df_pipes = df[~df["pipeline"].str.startswith("AGREEMENT")]

    print(f"\n{'='*70}\nEVALUATION SUMMARY\n{'='*70}")
    summary = df_pipes.groupby("pipeline").agg({
        "overall": "mean", "completeness": "mean",
        "grounding": "mean", "clarity": "mean",
        "citation_accuracy": "mean", "hallucination_score": "mean",
        "latency_s": "mean",
    }).reindex(PIPE_ORDER)
    summary.columns = ["Overall", "Complete", "Grounded", "Clarity",
                        "Citation", "Halluc.Free", "Latency(s)"]
    for col in ["Overall", "Complete", "Grounded", "Clarity", "Citation", "Halluc.Free"]:
        summary[col] = summary[col].round(1)
    summary["Latency(s)"] = summary["Latency(s)"].round(2)
    print(f"\n{summary.to_string()}")

    print(f"\n{'-'*70}\nBEST PIPELINE PER DIMENSION\n{'-'*70}")
    for col in ["Overall", "Complete", "Grounded", "Clarity", "Citation", "Halluc.Free"]:
        best = summary[col].idxmax()
        print(f"  {col:12s}: {best} ({summary.loc[best, col]})")
    fastest = summary["Latency(s)"].idxmin()
    print(f"  {'Speed':12s}: {fastest} ({summary.loc[fastest, 'Latency(s)']}s)")

    # Hallucination detail
    print(f"\n{'-'*70}\nHALLUCINATION DETAIL\n{'-'*70}")
    for pipe in PIPE_ORDER:
        pipe_df = df_pipes[df_pipes["pipeline"] == pipe]
        avg_unsup = pipe_df["num_unsupported_claims"].mean()
        avg_checked = pipe_df["num_claims_checked"].mean()
        print(f"  {pipe:12s}: {avg_unsup:.1f} unsupported claims / {avg_checked:.1f} checked (avg per query)")

    print(f"\n{'='*70}\n")


# ── Main ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nSTARTING EVALUATION — {len(TEST_QUERIES)} queries × {len(PIPELINES)} pipelines\n")

    df_results = run_full_evaluation()

    csv_path = f"{EVAL_OUTPUT_DIR}/evaluation_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n[OK] Results saved to {csv_path}")

    print_summary(df_results)

    print("Generating charts...")
    generate_charts(df_results)

    print(f"\nEVALUATION COMPLETE")
    print(f"  CSV:    {csv_path}")
    print(f"  Charts: {EVAL_OUTPUT_DIR}/*.png")
