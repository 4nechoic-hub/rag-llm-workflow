# -*- coding: utf-8 -*-
"""
LangGraph Multi-Step Research Agent
Built on top of the RAG pipeline for technical document analysis.

Agent Graph:
    ┌─────────┐
    │  START   │
    └────┬─────┘
         │
    ┌────▼─────┐
    │  PLAN    │  ← LLM decomposes the query into sub-questions
    └────┬─────┘
         │
    ┌────▼──────┐
    │ RETRIEVE  │  ← Embedding search for each sub-question
    └────┬──────┘
         │
    ┌────▼───────┐
    │ SYNTHESISE │  ← LLM produces a grounded answer from all evidence
    └────┬───────┘
         │
    ┌────▼──────┐
    │ CRITIQUE  │  ← LLM scores the answer (completeness, grounding, clarity)
    └────┬──────┘
         │
     ┌───▼───┐    score < threshold
     │ ROUTE │ ──────────────────► back to RETRIEVE (with refined query)
     └───┬───┘
         │ score >= threshold
    ┌────▼─────┐
    │ FINALISE │  ← Format final output with sources
    └────┬─────┘
         │
    ┌────▼─────┐
    │   END    │
    └──────────┘

Author: Tingyi Zhang
"""

# %% ========== IMPORTS ==========

import os
import sys
import json
import pickle
from pathlib import Path
from typing import TypedDict, List, Optional, Annotated
import operator

# --- Dependency check ---
_required = {
    "PyMuPDF": "fitz",
    "numpy": "numpy",
    "pandas": "pandas",
    "python-dotenv": "dotenv",
    "scikit-learn": "sklearn",
    "openai": "openai",
    "langgraph": "langgraph",
}
_missing = []
for _pkg, _imp in _required.items():
    try:
        __import__(_imp)
    except ImportError:
        _missing.append(_pkg)

if _missing:
    print("=" * 60)
    print("Missing packages. Install with:")
    print(f"  pip install {' '.join(_missing)}")
    print("=" * 60)
    sys.exit(1)

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

from langgraph.graph import StateGraph, START, END


# %% ========== CONFIGURATION ==========

PDF_FOLDER = "pdfs"
CACHE_FOLDER = "cache"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 5

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.0

# Critique thresholds
QUALITY_THRESHOLD = 7          # out of 10; below this triggers a re-retrieval loop
MAX_ITERATIONS = 3             # max refinement loops to avoid runaway costs

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found.\n"
        "Create a .env file with: OPENAI_API_KEY=sk-your_key_here"
    )

client = OpenAI(api_key=api_key)
os.makedirs(CACHE_FOLDER, exist_ok=True)

print(f"[OK] API key loaded  | Model: {CHAT_MODEL}")
print(f"[OK] Quality threshold: {QUALITY_THRESHOLD}/10 | Max iterations: {MAX_ITERATIONS}")


# %% ========== RAG CORE (reused from pipeline) ==========

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text("text")
        if text and text.strip():
            pages.append({"page_number": i + 1, "text": text.strip()})
    doc.close()
    return pages


def load_all_pdfs(pdf_folder=PDF_FOLDER):
    pdf_folder = Path(pdf_folder)
    if not pdf_folder.exists():
        raise FileNotFoundError(f"PDF folder not found: {pdf_folder.resolve()}")
    pdf_files = sorted(pdf_folder.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs in {pdf_folder.resolve()}")
    pdf_data = {}
    for f in pdf_files:
        print(f"  Loading: {f.name}")
        pdf_data[f.name] = extract_text_from_pdf(f)
    print(f"  Loaded {len(pdf_data)} PDF(s)")
    return pdf_data


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        c = text[start:end].strip()
        if c:
            chunks.append(c)
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


def create_document_chunks(pdf_data):
    records = []
    for doc_name, pages in pdf_data.items():
        for page in pages:
            for i, c in enumerate(chunk_text(page["text"])):
                records.append({
                    "doc_name": doc_name,
                    "page_number": page["page_number"],
                    "chunk_id": f"{doc_name}_p{page['page_number']}_c{i+1}",
                    "text": c
                })
    return pd.DataFrame(records)


def get_embedding(text):
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


def embed_chunks(df_chunks, force_recompute=False):
    cache_path = Path(CACHE_FOLDER) / "chunk_embeddings.pkl"
    if cache_path.exists() and not force_recompute:
        print("  Loading cached embeddings...")
        with open(cache_path, "rb") as f:
            p = pickle.load(f)
        return p["df_chunks"], p["embeddings"]

    print(f"  Embedding {len(df_chunks)} chunks...")
    embs = []
    for idx, row in df_chunks.iterrows():
        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"    chunk {idx+1}/{len(df_chunks)}")
        embs.append(get_embedding(row["text"]))
    embs = np.array(embs, dtype=np.float32)

    with open(cache_path, "wb") as f:
        pickle.dump({"df_chunks": df_chunks, "embeddings": embs}, f)
    return df_chunks, embs


def retrieve_top_k(query, df_chunks, embeddings, top_k=TOP_K):
    q_emb = np.array(get_embedding(query), dtype=np.float32).reshape(1, -1)
    sims = cosine_similarity(q_emb, embeddings)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]
    results = df_chunks.iloc[top_idx].copy()
    results["similarity"] = sims[top_idx]
    return results.reset_index(drop=True)


def format_context(retrieved_df):
    blocks = []
    for _, row in retrieved_df.iterrows():
        blocks.append(
            f"[Source: {row['doc_name']} | Page {row['page_number']}]\n{row['text']}"
        )
    return "\n\n---\n\n".join(blocks)


# %% ========== LLM HELPER ==========

def call_llm(system_prompt, user_prompt, temperature=TEMPERATURE):
    """Single LLM call wrapper."""
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return resp.choices[0].message.content


# %% ========== LANGGRAPH STATE ==========

class ResearchState(TypedDict):
    """Shared state passed between all nodes in the graph."""
    # --- Input ---
    query: str                             # original user question

    # --- Planning ---
    sub_questions: List[str]               # decomposed sub-questions

    # --- Retrieval ---
    retrieved_context: str                 # formatted evidence text
    source_table: Optional[str]            # source summary for display

    # --- Synthesis ---
    draft_answer: str                      # current answer draft

    # --- Critique ---
    critique: str                          # critique feedback text
    quality_score: float                   # 0-10 score from critic
    iteration: int                         # current refinement loop count

    # --- Output ---
    final_answer: str                      # polished final answer


# %% ========== GRAPH NODES ==========

def plan_node(state: ResearchState) -> dict:
    """
    NODE 1 — PLAN
    Decompose the user query into 2-4 focused sub-questions
    for better retrieval coverage.
    """
    print("\n>>> NODE: PLAN")

    system = (
        "You are a research planning assistant.\n"
        "Given a user question about technical documents, decompose it into "
        "2-4 focused sub-questions that together would fully answer the original query.\n"
        "Return ONLY a JSON array of strings, e.g.:\n"
        '["sub-question 1", "sub-question 2", "sub-question 3"]'
    )
    user = f"User question: {state['query']}"

    raw = call_llm(system, user)

    # Parse the JSON array
    try:
        # Strip markdown code fences if present
        cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
        sub_qs = json.loads(cleaned)
        if not isinstance(sub_qs, list):
            sub_qs = [state["query"]]
    except (json.JSONDecodeError, TypeError):
        sub_qs = [state["query"]]

    print(f"    Sub-questions ({len(sub_qs)}):")
    for i, q in enumerate(sub_qs, 1):
        print(f"      {i}. {q}")

    return {"sub_questions": sub_qs}


def retrieve_node(state: ResearchState) -> dict:
    """
    NODE 2 — RETRIEVE
    Run embedding search for each sub-question, merge and deduplicate results.
    """
    print("\n>>> NODE: RETRIEVE")

    all_results = []
    seen_chunk_ids = set()

    for sq in state["sub_questions"]:
        results = retrieve_top_k(sq, df_chunks, embeddings, top_k=TOP_K)
        for _, row in results.iterrows():
            if row["chunk_id"] not in seen_chunk_ids:
                seen_chunk_ids.add(row["chunk_id"])
                all_results.append(row)

    if not all_results:
        return {
            "retrieved_context": "No relevant chunks found.",
            "source_table": "No sources."
        }

    merged_df = pd.DataFrame(all_results)
    merged_df = merged_df.sort_values("similarity", ascending=False).head(TOP_K * 2)

    context = format_context(merged_df)
    source_summary = merged_df[["doc_name", "page_number", "chunk_id", "similarity"]].to_string(index=False)

    print(f"    Retrieved {len(merged_df)} unique chunks")

    return {
        "retrieved_context": context,
        "source_table": source_summary
    }


def synthesise_node(state: ResearchState) -> dict:
    """
    NODE 3 — SYNTHESISE
    Produce a comprehensive, grounded answer from retrieved evidence.
    If this is a refinement iteration, incorporate the critique feedback.
    """
    print("\n>>> NODE: SYNTHESISE")

    system = (
        "You are a technical research synthesiser.\n\n"
        "Rules:\n"
        "1. Answer using ONLY the provided context — do not use outside knowledge.\n"
        "2. Be comprehensive yet concise.\n"
        "3. Structure your answer with clear sections where appropriate.\n"
        "4. Cite sources inline, e.g. [Source: filename.pdf | Page X].\n"
        "5. If evidence is insufficient for any part, explicitly state that.\n"
        "6. End with a brief 'Sources' list."
    )

    # Build user prompt — include critique if this is a refinement loop
    user_parts = [
        f"Original question: {state['query']}",
        f"\nRetrieved evidence:\n{state['retrieved_context']}"
    ]

    if state.get("iteration", 0) > 0 and state.get("critique"):
        user_parts.append(
            f"\n--- PREVIOUS DRAFT ---\n{state['draft_answer']}"
            f"\n\n--- CRITIQUE FEEDBACK ---\n{state['critique']}"
            "\n\nPlease produce an improved answer that addresses the critique."
        )

    user = "\n".join(user_parts)
    answer = call_llm(system, user)

    iteration = state.get("iteration", 0)
    print(f"    Draft produced (iteration {iteration})")

    return {"draft_answer": answer}


def critique_node(state: ResearchState) -> dict:
    """
    NODE 4 — CRITIQUE
    Evaluate the draft answer for completeness, grounding, and clarity.
    Returns a quality score (0-10) and detailed feedback.
    """
    print("\n>>> NODE: CRITIQUE")

    system = (
        "You are a strict quality evaluator for technical document Q&A.\n\n"
        "Evaluate the provided answer against the retrieved context.\n"
        "Score on three dimensions (each 0-10):\n"
        "  - Completeness: Does it address all parts of the question?\n"
        "  - Grounding: Is every claim supported by the retrieved context?\n"
        "  - Clarity: Is it well-structured and easy to follow?\n\n"
        "Return ONLY valid JSON in this exact format:\n"
        "{\n"
        '  "completeness": <score>,\n'
        '  "grounding": <score>,\n'
        '  "clarity": <score>,\n'
        '  "overall": <average of three scores>,\n'
        '  "feedback": "<specific suggestions for improvement>"\n'
        "}"
    )

    user = (
        f"Question: {state['query']}\n\n"
        f"Retrieved context:\n{state['retrieved_context']}\n\n"
        f"Answer to evaluate:\n{state['draft_answer']}"
    )

    raw = call_llm(system, user)

    # Parse critique JSON
    try:
        cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
        critique_data = json.loads(cleaned)
        score = float(critique_data.get("overall", 0))
        feedback = critique_data.get("feedback", "No specific feedback.")

        # Build readable critique
        critique_text = (
            f"Completeness: {critique_data.get('completeness', '?')}/10\n"
            f"Grounding:    {critique_data.get('grounding', '?')}/10\n"
            f"Clarity:      {critique_data.get('clarity', '?')}/10\n"
            f"Overall:      {score}/10\n"
            f"Feedback:     {feedback}"
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        score = QUALITY_THRESHOLD  # default to passing if parse fails
        critique_text = f"Could not parse critique. Raw output:\n{raw}"
        feedback = ""

    iteration = state.get("iteration", 0) + 1

    print(f"    Score: {score}/10 (threshold: {QUALITY_THRESHOLD})")
    print(f"    Iteration: {iteration}/{MAX_ITERATIONS}")

    return {
        "critique": critique_text,
        "quality_score": score,
        "iteration": iteration
    }


def finalise_node(state: ResearchState) -> dict:
    """
    NODE 5 — FINALISE
    Format the final output.
    """
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
        f"CRITIQUE SUMMARY\n"
        f"{'-' * 60}\n"
        f"{state.get('critique', 'N/A')}\n\n"
        f"{'-' * 60}\n"
        f"RETRIEVED SOURCES\n"
        f"{'-' * 60}\n"
        f"{state.get('source_table', 'N/A')}\n"
    )

    return {"final_answer": final}


# %% ========== ROUTING LOGIC ==========

def should_refine(state: ResearchState) -> str:
    """
    Conditional edge: decide whether to loop back for refinement
    or proceed to finalise.
    """
    score = state.get("quality_score", 10)
    iteration = state.get("iteration", 0)

    if score < QUALITY_THRESHOLD and iteration < MAX_ITERATIONS:
        print(f"\n    → ROUTING: Score {score} < {QUALITY_THRESHOLD}, refining (iteration {iteration})")
        return "refine"
    else:
        reason = f"Score {score} >= {QUALITY_THRESHOLD}" if score >= QUALITY_THRESHOLD else f"Max iterations ({MAX_ITERATIONS}) reached"
        print(f"\n    → ROUTING: {reason}, finalising")
        return "finalise"


# %% ========== BUILD THE GRAPH ==========

def build_research_graph():
    """
    Construct the LangGraph StateGraph with all nodes and edges.
    """
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("synthesise", synthesise_node)
    graph.add_node("critique", critique_node)
    graph.add_node("finalise", finalise_node)

    # Define edges
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "synthesise")
    graph.add_edge("synthesise", "critique")

    # Conditional edge after critique
    graph.add_conditional_edges(
        "critique",
        should_refine,
        {
            "refine": "retrieve",      # loop back
            "finalise": "finalise"     # proceed to output
        }
    )

    graph.add_edge("finalise", END)

    return graph.compile()


# %% ========== HELPER: Run a query through the agent ==========

def run_research_agent(query: str):
    """
    Convenience function: run a query through the full research agent graph.
    Returns the final state dict.
    """
    initial_state = {
        "query": query,
        "sub_questions": [],
        "retrieved_context": "",
        "source_table": "",
        "draft_answer": "",
        "critique": "",
        "quality_score": 0.0,
        "iteration": 0,
        "final_answer": ""
    }

    result = research_agent.invoke(initial_state)
    return result


# %% ========== BUILD RAG SYSTEM ==========

print("\n" + "=" * 60)
print("BUILDING RAG SYSTEM")
print("=" * 60)

print("\nStep 1: Loading PDFs...")
pdf_data = load_all_pdfs(PDF_FOLDER)

print("\nStep 2: Creating chunks...")
df_chunks = create_document_chunks(pdf_data)
print(f"  Total chunks: {len(df_chunks)}")

print("\nStep 3: Embedding chunks...")
df_chunks, embeddings = embed_chunks(df_chunks, force_recompute=False)

print("\nStep 4: Building LangGraph research agent...")
research_agent = build_research_graph()

print("\n" + "=" * 60)
print("RESEARCH AGENT READY")
print(f"  Documents : {df_chunks['doc_name'].nunique()}")
print(f"  Chunks    : {len(df_chunks)}")
print(f"  Graph     : plan → retrieve → synthesise → critique → [refine|finalise]")
print("=" * 60)


# %% ========== EXAMPLE: Run a query ==========
# Edit the query below and run this cell.

query = "What experimental setup and measurement techniques were used in this study?"

result = run_research_agent(query)
print(result["final_answer"])


# %% ========== EXAMPLE: Another query ==========

query = "What are the main findings and conclusions of this research?"

result = run_research_agent(query)
print(result["final_answer"])


# %% ========== EXAMPLE: Methodology comparison ==========

query = "Describe the data acquisition and signal processing methodology in detail."

result = run_research_agent(query)
print(result["final_answer"])


# %% ========== SAVE RESULTS ==========
# Optional: save the last result to file

# save_path = Path(CACHE_FOLDER) / "agent_output.txt"
# with open(save_path, "w", encoding="utf-8") as f:
#     f.write(result["final_answer"])
# print(f"Saved to {save_path}")
