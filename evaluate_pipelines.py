# -*- coding: utf-8 -*-
"""
RAG Pipeline Evaluation — Head-to-Head Comparison
Compares Manual Pipeline vs LangGraph Agent vs LlamaIndex across:
  - Retrieval quality (similarity scores, source coverage)
  - Answer quality (LLM-as-judge: completeness, grounding, clarity)
  - Efficiency (latency per query, total LLM calls)
  - Cross-pipeline agreement (do they converge on similar answers?)

Outputs:
  - Console summary table
  - Detailed CSV (eval_results/evaluation_results.csv)
  - Comparison charts (eval_results/*.png)

Author: Tingyi Zhang
"""

# %% ========== IMPORTS ==========

import os
import sys
import json
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures

from dotenv import load_dotenv
from openai import OpenAI

# %% ========== CONFIGURATION ==========

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Create a .env file.")

client = OpenAI(api_key=api_key)

CHAT_MODEL = "gpt-4.1-mini"
OUTPUT_DIR = "eval_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# TEST QUERIES — edit these to match your documents
# A good evaluation set covers different question types.
# ------------------------------------------------------------------
TEST_QUERIES = [
    {
        "id": "Q1",
        "query": "What experimental setup and measurement techniques were used?",
        "type": "factual",
    },
    {
        "id": "Q2",
        "query": "What are the main findings and conclusions of this research?",
        "type": "summary",
    },
    {
        "id": "Q3",
        "query": "Describe the data acquisition and signal processing methodology.",
        "type": "technical",
    },
    {
        "id": "Q4",
        "query": "What calibration procedures were applied to the instruments?",
        "type": "factual",
    },
    {
        "id": "Q5",
        "query": "What are the limitations or sources of uncertainty in this study?",
        "type": "critical",
    },
]

print(f"[OK] Evaluation configured: {len(TEST_QUERIES)} test queries")
print(f"[OK] Output directory: {OUTPUT_DIR}/")


# %% ========== LLM-AS-JUDGE ==========

def judge_answer(query, answer, judge_model=CHAT_MODEL):
    """
    Use an LLM to score an answer on three dimensions (0-10 each).
    Returns dict with scores and feedback.
    """
    system_prompt = (
        "You are a strict evaluator for technical document Q&A systems.\n\n"
        "Score the given answer on three dimensions (each 0-10):\n"
        "  - Completeness: Does it address all parts of the question?\n"
        "  - Grounding: Does it appear to rely on retrieved evidence "
        "(citing sources, specific details) rather than generic knowledge?\n"
        "  - Clarity: Is it well-structured and easy to follow?\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "completeness": <int>,\n'
        '  "grounding": <int>,\n'
        '  "clarity": <int>,\n'
        '  "overall": <average as float>,\n'
        '  "feedback": "<one sentence of feedback>"\n'
        "}"
    )

    user_prompt = f"Question: {query}\n\nAnswer to evaluate:\n{answer}"

    try:
        resp = client.chat.completions.create(
            model=judge_model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = resp.choices[0].message.content
        cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
        scores = json.loads(cleaned)
        return scores
    except Exception as e:
        print(f"    [WARN] Judge parse error: {e}")
        return {
            "completeness": 0, "grounding": 0, "clarity": 0,
            "overall": 0, "feedback": f"Parse error: {e}"
        }


def judge_agreement(answer_a, answer_b, judge_model=CHAT_MODEL):
    """
    Use an LLM to score how much two answers agree (0-10).
    """
    system_prompt = (
        "You compare two answers to the same question.\n"
        "Score their agreement from 0-10:\n"
        "  0 = completely contradictory\n"
        "  5 = partially overlapping\n"
        "  10 = essentially the same content\n\n"
        "Return ONLY valid JSON: {\"agreement\": <int>, \"note\": \"<brief note>\"}"
    )

    user_prompt = f"Answer A:\n{answer_a}\n\nAnswer B:\n{answer_b}"

    try:
        resp = client.chat.completions.create(
            model=judge_model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = resp.choices[0].message.content
        cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(cleaned)
    except Exception:
        return {"agreement": 0, "note": "Parse error"}


# %% ========== PIPELINE RUNNERS ==========
# Each runner imports and calls its respective pipeline.
# They return a standardised dict for comparison.

def run_manual_pipeline(query):
    """Run query through the manual RAG pipeline."""
    from rag_pipeline_spyder import (
        load_all_pdfs, create_document_chunks, embed_chunks,
        answer_question, retrieve_top_k, format_context,
        PDF_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
    )

    # Build (uses cache if available — recompute if cache is incompatible)
    pdf_data = load_all_pdfs(PDF_FOLDER)
    df_chunks = create_document_chunks(pdf_data, CHUNK_SIZE, CHUNK_OVERLAP)
    try:
        df_chunks, embeddings = embed_chunks(df_chunks, force_recompute=False)
    except Exception:
        print("    [INFO] Cache incompatible, recomputing embeddings...")
        df_chunks, embeddings = embed_chunks(df_chunks, force_recompute=True)

    # Time the query
    t0 = time.time()
    answer, retrieved = answer_question(query, df_chunks, embeddings, top_k=TOP_K)
    latency = time.time() - t0

    # Extract retrieval metrics
    avg_sim = float(retrieved["similarity"].mean())
    max_sim = float(retrieved["similarity"].max())
    sources = retrieved["doc_name"].tolist()
    pages = retrieved["page_number"].tolist()

    return {
        "answer": answer,
        "latency": latency,
        "avg_similarity": avg_sim,
        "max_similarity": max_sim,
        "num_chunks": len(retrieved),
        "sources": sources,
        "pages": pages,
    }


def run_langgraph_agent(query):
    """Run query through the LangGraph research agent."""
    from langgraph_research_agent import (
        load_all_pdfs, create_document_chunks, embed_chunks,
        build_research_graph, PDF_FOLDER
    )

    # Build (uses cache — force recompute if cache is incompatible)
    pdf_data = load_all_pdfs(PDF_FOLDER)
    df_chunks = create_document_chunks(pdf_data)
    try:
        df_chunks_out, embeddings = embed_chunks(df_chunks, force_recompute=False)
    except Exception:
        print("    [INFO] Cache incompatible, recomputing embeddings...")
        df_chunks_out, embeddings = embed_chunks(df_chunks, force_recompute=True)

    # Need to make df_chunks and embeddings available to the graph nodes
    import langgraph_research_agent as lg_module
    lg_module.df_chunks = df_chunks_out
    lg_module.embeddings = embeddings

    research_agent = build_research_graph()

    initial_state = {
        "query": query,
        "sub_questions": [],
        "retrieved_context": "",
        "source_table": "",
        "draft_answer": "",
        "critique": "",
        "quality_score": 0.0,
        "iteration": 0,
        "final_answer": "",
    }

    t0 = time.time()
    result = research_agent.invoke(initial_state)
    latency = time.time() - t0

    return {
        "answer": result.get("draft_answer", ""),
        "latency": latency,
        "quality_score": result.get("quality_score", 0),
        "iterations": result.get("iteration", 1),
        "avg_similarity": 0,  # not directly accessible from graph output
        "max_similarity": 0,
        "num_chunks": 0,
        "sources": [],
        "pages": [],
    }


def run_llamaindex_pipeline(query):
    """Run query through the LlamaIndex pipeline."""
    from llamaindex_rag_pipeline import (
        build_index, create_query_engine, create_retriever, QA_PROMPT, TOP_K
    )

    index = build_index(force_rebuild=False)
    qa_engine = create_query_engine(index, prompt_template=QA_PROMPT, top_k=TOP_K)
    retriever = create_retriever(index, top_k=TOP_K)

    # Time the query
    t0 = time.time()
    response = qa_engine.query(query)
    latency = time.time() - t0

    answer = str(response)

    # Extract retrieval metrics
    scores = [n.score for n in response.source_nodes if n.score]
    sources = [n.metadata.get("file_name", "?") for n in response.source_nodes]
    pages = [n.metadata.get("page_label", "?") for n in response.source_nodes]

    return {
        "answer": answer,
        "latency": latency,
        "avg_similarity": float(np.mean(scores)) if scores else 0,
        "max_similarity": float(np.max(scores)) if scores else 0,
        "num_chunks": len(response.source_nodes),
        "sources": sources,
        "pages": pages,
    }


# %% ========== RUN EVALUATION ==========

PIPELINES = {
    "Manual": run_manual_pipeline,
    "LangGraph": run_langgraph_agent,
    "LlamaIndex": run_llamaindex_pipeline,
}


def run_full_evaluation():
    """
    Run all test queries through all pipelines.
    Score each answer with LLM-as-judge.
    Returns a DataFrame of results.
    """
    all_results = []

    for q_info in TEST_QUERIES:
        qid = q_info["id"]
        query = q_info["query"]
        qtype = q_info["type"]

        print(f"\n{'=' * 60}")
        print(f"EVALUATING {qid}: {query[:60]}...")
        print("=" * 60)

        answers = {}  # pipeline_name -> answer text

        for pipe_name, pipe_fn in PIPELINES.items():
            print(f"\n  Running {pipe_name}...")

            result = None
            try:
                result = pipe_fn(query)
                answer = result["answer"]
                latency = result["latency"]
                avg_sim = result.get("avg_similarity", 0)
                max_sim = result.get("max_similarity", 0)
                num_chunks = result.get("num_chunks", 0)
            except Exception as e:
                print(f"    [ERROR] {pipe_name} failed: {e}")
                answer = f"ERROR: {e}"
                latency = 0
                avg_sim = 0
                max_sim = 0
                num_chunks = 0

            answers[pipe_name] = answer

            # Judge the answer
            print(f"    Judging answer...")
            scores = judge_answer(query, answer)

            row = {
                "query_id": qid,
                "query": query,
                "query_type": qtype,
                "pipeline": pipe_name,
                "answer": answer[:500],  # truncate for CSV
                "latency_s": round(latency, 2),
                "avg_similarity": round(avg_sim, 4),
                "max_similarity": round(max_sim, 4),
                "num_chunks": num_chunks,
                "completeness": scores.get("completeness", 0),
                "grounding": scores.get("grounding", 0),
                "clarity": scores.get("clarity", 0),
                "overall": scores.get("overall", 0),
                "feedback": scores.get("feedback", ""),
            }

            # LangGraph-specific metrics
            if pipe_name == "LangGraph" and result is not None:
                row["lg_quality_score"] = result.get("quality_score", 0)
                row["lg_iterations"] = result.get("iterations", 1)

            all_results.append(row)
            print(f"    Score: {scores.get('overall', '?')}/10 "
                  f"| Latency: {latency:.2f}s")

        # Cross-pipeline agreement
        pipe_names = list(answers.keys())
        for i in range(len(pipe_names)):
            for j in range(i + 1, len(pipe_names)):
                name_a, name_b = pipe_names[i], pipe_names[j]
                print(f"  Judging agreement: {name_a} vs {name_b}...")
                agreement = judge_agreement(answers[name_a], answers[name_b])

                all_results.append({
                    "query_id": qid,
                    "query": query,
                    "query_type": qtype,
                    "pipeline": f"AGREEMENT: {name_a} vs {name_b}",
                    "answer": "",
                    "latency_s": 0,
                    "avg_similarity": 0,
                    "max_similarity": 0,
                    "num_chunks": 0,
                    "completeness": 0,
                    "grounding": 0,
                    "clarity": 0,
                    "overall": agreement.get("agreement", 0),
                    "feedback": agreement.get("note", ""),
                })

    df = pd.DataFrame(all_results)
    return df


# %% ========== CHARTS ==========

def generate_charts(df):
    """Generate comparison charts from evaluation results."""

    # Filter to pipeline results only (exclude agreement rows)
    df_pipes = df[~df["pipeline"].str.startswith("AGREEMENT")].copy()
    df_agree = df[df["pipeline"].str.startswith("AGREEMENT")].copy()

    # Colour palette
    colours = {"Manual": "#4C78A8", "LangGraph": "#F58518", "LlamaIndex": "#54A24B"}

    # ─── Chart 1: Overall Quality by Pipeline ───
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = df_pipes.groupby("pipeline")["overall"].mean().reindex(
        ["Manual", "LangGraph", "LlamaIndex"]
    )
    bars = ax.bar(pivot.index, pivot.values,
                  color=[colours[p] for p in pivot.index],
                  edgecolor="white", linewidth=1.2)
    ax.set_ylabel("Average Overall Score (0-10)", fontsize=11)
    ax.set_title("Overall Answer Quality by Pipeline", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 10)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, pivot.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{val:.1f}", ha="center", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_overall_quality.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/01_overall_quality.png")

    # ─── Chart 2: Quality Breakdown (Completeness, Grounding, Clarity) ───
    fig, ax = plt.subplots(figsize=(10, 5))
    dims = ["completeness", "grounding", "clarity"]
    pipe_order = ["Manual", "LangGraph", "LlamaIndex"]
    x = np.arange(len(dims))
    width = 0.25

    for i, pipe in enumerate(pipe_order):
        pipe_df = df_pipes[df_pipes["pipeline"] == pipe]
        vals = [pipe_df[d].mean() for d in dims]
        bars = ax.bar(x + i * width, vals, width,
                      label=pipe, color=colours[pipe],
                      edgecolor="white", linewidth=1)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{val:.1f}", ha="center", fontsize=9)

    ax.set_ylabel("Average Score (0-10)", fontsize=11)
    ax.set_title("Answer Quality Breakdown", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(["Completeness", "Grounding", "Clarity"], fontsize=11)
    ax.set_ylim(0, 10)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_quality_breakdown.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/02_quality_breakdown.png")

    # ─── Chart 3: Latency Comparison ───
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot_lat = df_pipes.groupby("pipeline")["latency_s"].mean().reindex(pipe_order)
    bars = ax.bar(pivot_lat.index, pivot_lat.values,
                  color=[colours[p] for p in pivot_lat.index],
                  edgecolor="white", linewidth=1.2)
    ax.set_ylabel("Average Latency (seconds)", fontsize=11)
    ax.set_title("Query Latency by Pipeline", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, pivot_lat.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}s", ha="center", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_latency.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/03_latency.png")

    # ─── Chart 4: Per-Query Scores ───
    fig, ax = plt.subplots(figsize=(12, 5))
    query_ids = df_pipes["query_id"].unique()
    x = np.arange(len(query_ids))
    width = 0.25

    for i, pipe in enumerate(pipe_order):
        pipe_df = df_pipes[df_pipes["pipeline"] == pipe]
        vals = []
        for qid in query_ids:
            row = pipe_df[pipe_df["query_id"] == qid]
            vals.append(row["overall"].values[0] if len(row) > 0 else 0)
        ax.bar(x + i * width, vals, width,
               label=pipe, color=colours[pipe],
               edgecolor="white", linewidth=1)

    ax.set_ylabel("Overall Score (0-10)", fontsize=11)
    ax.set_title("Per-Query Answer Quality", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(query_ids, fontsize=11)
    ax.set_xlabel("Query ID", fontsize=11)
    ax.set_ylim(0, 10)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_per_query.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/04_per_query.png")

    # ─── Chart 5: Cross-Pipeline Agreement ───
    if len(df_agree) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        agree_avg = df_agree.groupby("pipeline")["overall"].mean()
        pair_colours = ["#9D755D", "#B07AA1", "#76B7B2"]
        bars = ax.bar(range(len(agree_avg)), agree_avg.values,
                      color=pair_colours[:len(agree_avg)],
                      edgecolor="white", linewidth=1.2)
        ax.set_ylabel("Average Agreement Score (0-10)", fontsize=11)
        ax.set_title("Cross-Pipeline Answer Agreement", fontsize=13, fontweight="bold")
        labels = [s.replace("AGREEMENT: ", "") for s in agree_avg.index]
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 10)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, agree_avg.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                    f"{val:.1f}", ha="center", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/05_agreement.png", dpi=150)
        plt.close()
        print(f"  Saved: {OUTPUT_DIR}/05_agreement.png")

    # ─── Chart 6: Radar Chart — Pipeline Profiles ───
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    categories = ["Completeness", "Grounding", "Clarity", "Speed"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # close the loop

    for pipe in pipe_order:
        pipe_df = df_pipes[df_pipes["pipeline"] == pipe]
        vals = [
            pipe_df["completeness"].mean(),
            pipe_df["grounding"].mean(),
            pipe_df["clarity"].mean(),
            10 - min(pipe_df["latency_s"].mean(), 10),  # invert: lower latency = better
        ]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=pipe, color=colours[pipe])
        ax.fill(angles, vals, alpha=0.1, color=colours[pipe])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 10)
    ax.set_title("Pipeline Profile Comparison", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/06_radar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/06_radar.png")


# %% ========== SUMMARY TABLE ==========

def print_summary(df):
    """Print a clean summary table to console."""
    df_pipes = df[~df["pipeline"].str.startswith("AGREEMENT")]

    print(f"\n{'=' * 70}")
    print("EVALUATION SUMMARY")
    print("=" * 70)

    summary = df_pipes.groupby("pipeline").agg({
        "overall": "mean",
        "completeness": "mean",
        "grounding": "mean",
        "clarity": "mean",
        "latency_s": "mean",
    }).reindex(["Manual", "LangGraph", "LlamaIndex"])

    summary.columns = ["Overall", "Complete", "Grounded", "Clarity", "Latency(s)"]

    for col in ["Overall", "Complete", "Grounded", "Clarity"]:
        summary[col] = summary[col].round(1)
    summary["Latency(s)"] = summary["Latency(s)"].round(2)

    print(f"\n{summary.to_string()}")

    # Best pipeline per dimension
    print(f"\n{'-' * 70}")
    print("BEST PIPELINE PER DIMENSION")
    print("-" * 70)
    for col in ["Overall", "Complete", "Grounded", "Clarity"]:
        best = summary[col].idxmax()
        print(f"  {col:12s}: {best} ({summary.loc[best, col]})")
    fastest = summary["Latency(s)"].idxmin()
    print(f"  {'Speed':12s}: {fastest} ({summary.loc[fastest, 'Latency(s)']}s)")

    # Agreement summary
    df_agree = df[df["pipeline"].str.startswith("AGREEMENT")]
    if len(df_agree) > 0:
        print(f"\n{'-' * 70}")
        print("CROSS-PIPELINE AGREEMENT")
        print("-" * 70)
        for pair, group in df_agree.groupby("pipeline"):
            label = pair.replace("AGREEMENT: ", "")
            print(f"  {label}: {group['overall'].mean():.1f}/10")

    print(f"\n{'=' * 70}\n")


# %% ========== RUN EVERYTHING ==========

print("\n" + "=" * 60)
print("STARTING FULL EVALUATION")
print(f"Pipelines: {', '.join(PIPELINES.keys())}")
print(f"Queries:   {len(TEST_QUERIES)}")
print("=" * 60)

df_results = run_full_evaluation()

# Save raw results
csv_path = f"{OUTPUT_DIR}/evaluation_results.csv"
df_results.to_csv(csv_path, index=False)
print(f"\n[OK] Results saved to {csv_path}")

# Print summary
print_summary(df_results)

# Generate charts
print("Generating charts...")
generate_charts(df_results)

print(f"\n{'=' * 60}")
print("EVALUATION COMPLETE")
print(f"  CSV:    {OUTPUT_DIR}/evaluation_results.csv")
print(f"  Charts: {OUTPUT_DIR}/*.png")
print("=" * 60)