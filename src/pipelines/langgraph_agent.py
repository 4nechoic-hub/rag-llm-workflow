# -*- coding: utf-8 -*-
"""
LangGraph Multi-Step Research Agent

Agentic workflow with query decomposition, retrieval, synthesis,
self-critique, and conditional refinement loops.

Graph: START → PLAN → RETRIEVE → SYNTHESISE → CRITIQUE → [REFINE|FINALISE] → END

Author: Tingyi Zhang
"""

import json
from typing import TypedDict, List, Optional

from langgraph.graph import StateGraph, START, END

from src.core.pdf_loader import load_all_pdfs
from src.core.chunker import create_document_chunks
from src.core.embedder import embed_chunks
from src.core.retriever import retrieve_top_k, format_context
from src.core.llm import call_llm
from src.config import (
    PDF_FOLDER, TOP_K, QUALITY_THRESHOLD, MAX_ITERATIONS,
)

# ── Module-level state (populated by build function) ────────────
_df_chunks = None
_embeddings = None


# ── LangGraph state schema ──────────────────────────────────────

class ResearchState(TypedDict):
    query: str
    sub_questions: List[str]
    retrieved_context: str
    source_table: Optional[str]
    draft_answer: str
    critique: str
    quality_score: float
    iteration: int
    final_answer: str


# ── Graph nodes ─────────────────────────────────────────────────

def plan_node(state: ResearchState) -> dict:
    """Decompose the user query into 2-4 focused sub-questions."""
    print("\n>>> NODE: PLAN")

    system = (
        "You are a research planning assistant.\n\n"
        "Given a user question about technical/scientific documents, "
        "decompose it into 2-4 focused sub-questions that will help "
        "retrieve the most relevant information.\n\n"
        "Return ONLY a JSON array of strings, e.g.:\n"
        '["sub-question 1", "sub-question 2", "sub-question 3"]'
    )
    user = f"User question: {state['query']}"

    raw = call_llm(system, user)

    try:
        cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
        sub_qs = json.loads(cleaned)
        if not isinstance(sub_qs, list):
            sub_qs = [state["query"]]
    except (json.JSONDecodeError, TypeError):
        sub_qs = [state["query"]]

    print(f"    Sub-questions: {sub_qs}")
    return {"sub_questions": sub_qs}


def retrieve_node(state: ResearchState) -> dict:
    """Retrieve relevant chunks for each sub-question and deduplicate."""
    print("\n>>> NODE: RETRIEVE")

    all_chunks = {}
    for sq in state["sub_questions"]:
        results = retrieve_top_k(sq, _df_chunks, _embeddings, top_k=TOP_K)
        for _, row in results.iterrows():
            cid = row["chunk_id"]
            if cid not in all_chunks or row["similarity"] > all_chunks[cid]["similarity"]:
                all_chunks[cid] = row

    import pandas as pd
    combined = pd.DataFrame(all_chunks.values())
    combined = combined.sort_values("similarity", ascending=False).head(TOP_K * 2)

    context = format_context(combined)

    source_rows = []
    for _, row in combined.iterrows():
        source_rows.append(
            f"  {row['doc_name']:40s} Page {row['page_number']:3d}  "
            f"Score: {row['similarity']:.4f}"
        )
    source_table = "\n".join(source_rows)

    print(f"    Retrieved {len(combined)} unique chunks")
    return {"retrieved_context": context, "source_table": source_table}


def synthesise_node(state: ResearchState) -> dict:
    """Produce a grounded answer from all retrieved evidence."""
    print("\n>>> NODE: SYNTHESISE")

    system = (
        "You are a technical research synthesiser.\n\n"
        "Rules:\n"
        "1. Answer using ONLY the provided context.\n"
        "2. Be comprehensive yet concise.\n"
        "3. Structure your answer with clear sections where appropriate.\n"
        "4. Cite sources inline, e.g. [Source: filename.pdf | Page X].\n"
        "5. If evidence is insufficient, explicitly state that.\n"
        "6. End with a brief 'Sources' list."
    )

    user_parts = [
        f"Original question: {state['query']}",
        f"\nRetrieved evidence:\n{state['retrieved_context']}",
    ]

    if state.get("iteration", 0) > 0 and state.get("critique"):
        user_parts.append(
            f"\n--- PREVIOUS DRAFT ---\n{state['draft_answer']}"
            f"\n\n--- CRITIQUE FEEDBACK ---\n{state['critique']}"
            "\n\nPlease produce an improved answer addressing the critique."
        )

    answer = call_llm(system, "\n".join(user_parts))
    print(f"    Draft produced (iteration {state.get('iteration', 0)})")
    return {"draft_answer": answer}


def critique_node(state: ResearchState) -> dict:
    """Score the draft answer on completeness, grounding, and clarity."""
    print("\n>>> NODE: CRITIQUE")

    system = (
        "You are a strict quality evaluator for technical document Q&A.\n\n"
        "Score on three dimensions (each 0-10):\n"
        "  - Completeness: Does it address all parts of the question?\n"
        "  - Grounding: Is every claim supported by retrieved context?\n"
        "  - Clarity: Is it well-structured and easy to follow?\n\n"
        "Return ONLY valid JSON:\n"
        '{\n  "completeness": <score>,\n  "grounding": <score>,\n'
        '  "clarity": <score>,\n  "overall": <average>,\n'
        '  "feedback": "<specific suggestions>"\n}'
    )

    user = (
        f"Question: {state['query']}\n\n"
        f"Retrieved context:\n{state['retrieved_context']}\n\n"
        f"Answer to evaluate:\n{state['draft_answer']}"
    )

    raw = call_llm(system, user)

    try:
        cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
        data = json.loads(cleaned)
        score = float(data.get("overall", 0))
        feedback = data.get("feedback", "No specific feedback.")

        critique_text = (
            f"Completeness: {data.get('completeness', '?')}/10\n"
            f"Grounding:    {data.get('grounding', '?')}/10\n"
            f"Clarity:      {data.get('clarity', '?')}/10\n"
            f"Overall:      {score}/10\n"
            f"Feedback:     {feedback}"
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        score = QUALITY_THRESHOLD
        critique_text = f"Could not parse critique. Raw:\n{raw}"

    iteration = state.get("iteration", 0) + 1
    print(f"    Score: {score}/10 | Iteration: {iteration}/{MAX_ITERATIONS}")

    return {"critique": critique_text, "quality_score": score, "iteration": iteration}


def finalise_node(state: ResearchState) -> dict:
    """Format the final output with metadata."""
    print("\n>>> NODE: FINALISE")

    final = (
        f"{'=' * 60}\n"
        f"RESEARCH AGENT — FINAL ANSWER\n"
        f"{'=' * 60}\n"
        f"Query: {state['query']}\n"
        f"Iterations: {state.get('iteration', 1)}\n"
        f"Quality Score: {state.get('quality_score', 'N/A')}/10\n"
        f"{'=' * 60}\n\n"
        f"{state['draft_answer']}\n\n"
        f"{'-' * 60}\n"
        f"CRITIQUE SUMMARY\n{'-' * 60}\n"
        f"{state.get('critique', 'N/A')}\n\n"
        f"{'-' * 60}\n"
        f"RETRIEVED SOURCES\n{'-' * 60}\n"
        f"{state.get('source_table', 'N/A')}\n"
    )
    return {"final_answer": final}


# ── Routing logic ───────────────────────────────────────────────

def should_refine(state: ResearchState) -> str:
    score = state.get("quality_score", 10)
    iteration = state.get("iteration", 0)

    if score < QUALITY_THRESHOLD and iteration < MAX_ITERATIONS:
        print(f"    → ROUTING: Score {score} < {QUALITY_THRESHOLD}, refining")
        return "refine"
    else:
        print(f"    → ROUTING: Finalising")
        return "finalise"


# ── Build the graph ─────────────────────────────────────────────

def build_research_graph() -> object:
    """Construct and compile the LangGraph StateGraph."""
    graph = StateGraph(ResearchState)

    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("synthesise", synthesise_node)
    graph.add_node("critique", critique_node)
    graph.add_node("finalise", finalise_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "synthesise")
    graph.add_edge("synthesise", "critique")
    graph.add_conditional_edges(
        "critique", should_refine,
        {"refine": "retrieve", "finalise": "finalise"},
    )
    graph.add_edge("finalise", END)

    return graph.compile()


# ── Convenience: build the full agent ───────────────────────────

def build_langgraph_pipeline(pdf_folder=PDF_FOLDER, force_recompute=False):
    """Load PDFs, chunk, embed, build graph — returns the compiled agent."""
    global _df_chunks, _embeddings

    pdf_data = load_all_pdfs(pdf_folder)
    _df_chunks = create_document_chunks(pdf_data)
    _df_chunks, _embeddings = embed_chunks(_df_chunks, force_recompute=force_recompute)
    agent = build_research_graph()
    return agent


def run_research_agent(agent, query: str) -> dict:
    """Run a query through the research agent. Returns the final state."""
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
    return agent.invoke(initial_state)
