# -*- coding: utf-8 -*-
"""
RAG Pipeline Evaluation — Head-to-Head Comparison

Compares Manual Pipeline vs LangGraph Agent vs LlamaIndex across:
  - Answer quality (LLM-as-judge: completeness, grounding, clarity)
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


# ── Pipeline runners ────────────────────────────────────────────

def run_manual(query):
    from src.pipelines.manual_pipeline import build_manual_pipeline, answer_question
    df_chunks, embeddings = build_manual_pipeline()

    t0 = time.time()
    answer, retrieved = answer_question(query, df_chunks, embeddings, top_k=TOP_K)
    latency = time.time() - t0

    return {
        "answer": answer, "latency": latency,
        "avg_similarity": float(retrieved["similarity"].mean()),
        "max_similarity": float(retrieved["similarity"].max()),
        "num_chunks": len(retrieved),
    }


def run_langgraph(query):
    from src.pipelines.langgraph_agent import build_langgraph_pipeline, run_research_agent
    agent = build_langgraph_pipeline()

    t0 = time.time()
    result = run_research_agent(
        agent,
        query,
        task_type="Question Answering",
        top_k=TOP_K,
    )
    latency = time.time() - t0

    answer = result.get("answer") or result.get("final_answer") or ""
    sources = result.get("sources", [])
    scores = [s["score"] for s in sources if s.get("score") is not None]

    return {
        "answer": answer,
        "latency": latency,
        "avg_similarity": float(np.mean(scores)) if scores else 0,
        "max_similarity": float(np.max(scores)) if scores else 0,
        "num_chunks": len(sources),
        "quality_score": result.get("quality_score", 0),
        "iterations": result.get("iteration", 1),
    }


def run_llamaindex(query):
    from src.pipelines.llamaindex_pipeline import build_index, answer_question
    index = build_index(force_rebuild=False)

    t0 = time.time()
    answer, sources = answer_question(query, index, top_k=TOP_K)
    latency = time.time() - t0

    scores = [s["score"] for s in sources if s.get("score")]
    return {
        "answer": answer, "latency": latency,
        "avg_similarity": float(np.mean(scores)) if scores else 0,
        "max_similarity": float(np.max(scores)) if scores else 0,
        "num_chunks": len(sources),
    }


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
                answer, latency = result["answer"], result["latency"]
            except Exception as e:
                print(f"    [ERROR] {pipe_name}: {e}")
                answer, latency = f"ERROR: {e}", 0
                result = {}

            answers[pipe_name] = answer
            scores = judge_answer(query, answer)

            row = {
                "query_id": qid, "query": query, "query_type": qtype,
                "pipeline": pipe_name, "answer": answer[:500],
                "latency_s": round(latency, 2),
                "avg_similarity": round(result.get("avg_similarity", 0), 4),
                "max_similarity": round(result.get("max_similarity", 0), 4),
                "num_chunks": result.get("num_chunks", 0),
                "quality_score": result.get("quality_score"),
                "iterations": result.get("iterations"),
                "completeness": scores.get("completeness", 0),
                "grounding": scores.get("grounding", 0),
                "clarity": scores.get("clarity", 0),
                "overall": scores.get("overall", 0),
                "feedback": scores.get("feedback", ""),
            }
            all_results.append(row)
            print(f"    Score: {scores.get('overall', '?')}/10 | Latency: {latency:.2f}s")

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

    # Chart 2: Quality Breakdown
    fig, ax = plt.subplots(figsize=(10, 5))
    dims = ["completeness", "grounding", "clarity"]
    x = np.arange(len(dims))
    width = 0.25
    for i, pipe in enumerate(PIPE_ORDER):
        pipe_df = df_pipes[df_pipes["pipeline"] == pipe]
        vals = [pipe_df[d].mean() for d in dims]
        bars = ax.bar(x + i*width, vals, width, label=pipe, color=COLOURS[pipe], edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.1, f"{val:.1f}", ha="center", fontsize=9)
    ax.set_ylabel("Average Score (0-10)")
    ax.set_title("Answer Quality Breakdown", fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(["Completeness", "Grounding", "Clarity"])
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

    # Chart 6: Radar
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    categories = ["Completeness", "Grounding", "Clarity", "Speed"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    for pipe in PIPE_ORDER:
        pipe_df = df_pipes[df_pipes["pipeline"] == pipe]
        vals = [pipe_df["completeness"].mean(), pipe_df["grounding"].mean(),
                pipe_df["clarity"].mean(), 10 - min(pipe_df["latency_s"].mean(), 10)]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=pipe, color=COLOURS[pipe])
        ax.fill(angles, vals, alpha=0.1, color=COLOURS[pipe])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 10)
    ax.set_title("Pipeline Profile Comparison", fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=False)
    plt.tight_layout()
    plt.savefig(f"{EVAL_OUTPUT_DIR}/06_radar.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Charts saved to {EVAL_OUTPUT_DIR}/")


def print_summary(df):
    """Print summary table to console."""
    df_pipes = df[~df["pipeline"].str.startswith("AGREEMENT")]

    print(f"\n{'='*70}\nEVALUATION SUMMARY\n{'='*70}")
    summary = df_pipes.groupby("pipeline").agg({
        "overall": "mean", "completeness": "mean",
        "grounding": "mean", "clarity": "mean", "latency_s": "mean",
    }).reindex(PIPE_ORDER)
    summary.columns = ["Overall", "Complete", "Grounded", "Clarity", "Latency(s)"]
    for col in ["Overall", "Complete", "Grounded", "Clarity"]:
        summary[col] = summary[col].round(1)
    summary["Latency(s)"] = summary["Latency(s)"].round(2)
    print(f"\n{summary.to_string()}")

    print(f"\n{'-'*70}\nBEST PIPELINE PER DIMENSION\n{'-'*70}")
    for col in ["Overall", "Complete", "Grounded", "Clarity"]:
        best = summary[col].idxmax()
        print(f"  {col:12s}: {best} ({summary.loc[best, col]})")
    fastest = summary["Latency(s)"].idxmin()
    print(f"  {'Speed':12s}: {fastest} ({summary.loc[fastest, 'Latency(s)']}s)")
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
