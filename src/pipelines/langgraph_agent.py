# -*- coding: utf-8 -*-
"""
LangGraph Multi-Step Research Agent

Agentic workflow with query decomposition, retrieval, synthesis,
self-critique, and conditional refinement loops.

Graph: START -> PLAN -> RETRIEVE -> SYNTHESISE -> CRITIQUE -> [REFINE|FINALISE] -> END

Author: Tingyi Zhang
"""

import json
from typing import Any, List, Optional, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from src.core.chunker import create_document_chunks
from src.core.embedder import embed_chunks
from src.core.extraction import (
    EXTRACTION_MISSING_VALUE,
    extraction_field_bullets,
    extraction_schema_instruction,
    validate_and_format_extraction,
)
from src.core.llm import call_llm
from src.core.pdf_loader import load_all_pdfs
from src.core.retriever import format_context, retrieve_top_k
from src.core.types import PipelineResult, SourceChunk
from src.config import MAX_ITERATIONS, PDF_FOLDER, QUALITY_THRESHOLD, TOP_K, RERANK_ENABLED


class ResearchState(TypedDict):
    query: str
    task_type: str
    top_k: int
    sub_questions: List[str]
    retrieved_context: str
    sources: List[dict]
    source_table: Optional[str]
    draft_answer: str
    answer: str
    critique: str
    quality_score: float
    critique_parse_error: bool
    iteration: int
    final_answer: str



def _clean_json_response(raw: str) -> str:
    """Strip optional markdown fences from a model response."""
    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):]
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()



def _normalise_sub_questions(candidate: Any, fallback_query: str) -> List[str]:
    """Validate and deduplicate a proposed list of sub-questions."""
    if not isinstance(candidate, list):
        return [fallback_query]

    sub_questions: List[str] = []
    seen = set()
    for item in candidate:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        sub_questions.append(text)

    return sub_questions or [fallback_query]



def _task_planning_guidance(task_type: str) -> str:
    if task_type == "Structured Extraction":
        return (
            "Plan sub-questions that retrieve evidence for the requested fields: "
            "title, objective, methodology, experimental setup, main findings, and limitations."
        )
    if task_type == "Document Comparison":
        return (
            "Plan sub-questions that separate the documents or approaches being compared, "
            "surface shared themes, and isolate evidence-backed differences."
        )
    return (
        "Plan focused retrieval questions that directly answer the user's question with grounded evidence."
    )



def _task_output_instruction(task_type: str) -> str:
    if task_type == "Structured Extraction":
        return extraction_schema_instruction()
    if task_type == "Document Comparison":
        return (
            "Return a structured comparison with these sections: "
            "1. Documents involved 2. Similarities 3. Differences "
            "4. Key methodological differences 5. Key findings differences 6. Source list."
        )
    return "Return: 1. Answer 2. Key supporting points 3. Source list."



def _render_source_table(sources: List[dict]) -> str:
    if not sources:
        return "No sources retrieved."

    rows = []
    for source in sources:
        score = source.get("score")
        score_str = f"Score: {score:.4f}" if isinstance(score, (int, float)) else "Score: N/A"
        rows.append(
            f"  {source['file']:40s} Page {int(source['page']):3d}  {score_str}"
        )
    return "\n".join(rows)


# -- Graph construction -------------------------------------------------


def build_research_graph(
    df_chunks,
    embeddings,
    *,
    quality_threshold: float = QUALITY_THRESHOLD,
    max_iterations: int = MAX_ITERATIONS,
) -> object:
    """Construct and compile the LangGraph StateGraph."""

    def plan_node(state: ResearchState) -> dict:
        """Decompose the user query into focused sub-questions."""
        print("\n>>> NODE: PLAN")

        task_type = state.get("task_type", "Question Answering")
        system = (
            "You are a research planning assistant.\n\n"
            "Given a user question about technical or scientific documents, decompose it into 2-4 focused "
            "sub-questions that will improve document retrieval.\n\n"
            f"Task type: {task_type}.\n"
            f"Task-specific guidance: {_task_planning_guidance(task_type)}\n\n"
            "Return ONLY a JSON array of strings, for example:\n"
            '["sub-question 1", "sub-question 2", "sub-question 3"]'
        )
        user = f"User question: {state['query']}"

        raw = call_llm(system, user)
        try:
            sub_qs = json.loads(_clean_json_response(raw))
        except (json.JSONDecodeError, TypeError):
            sub_qs = [state["query"]]

        sub_qs = _normalise_sub_questions(sub_qs, state["query"])
        print(f"    Sub-questions: {sub_qs}")
        return {"sub_questions": sub_qs}

    def retrieve_node(state: ResearchState) -> dict:
        """Retrieve relevant chunks for each sub-question and deduplicate."""
        print("\n>>> NODE: RETRIEVE")

        top_k = max(1, int(state.get("top_k", TOP_K)))
        sub_questions = state.get("sub_questions") or [state["query"]]
        all_chunks = {}

        for sq in sub_questions:
            results = retrieve_top_k(sq, df_chunks, embeddings, top_k=top_k)
            for _, row in results.iterrows():
                chunk_id = row["chunk_id"]
                if chunk_id not in all_chunks or row["similarity"] > all_chunks[chunk_id]["similarity"]:
                    all_chunks[chunk_id] = row

        if all_chunks:
            combined = pd.DataFrame(all_chunks.values())
            combined = combined.sort_values("similarity", ascending=False).head(top_k)
            context = format_context(combined)
            sources = [
                {
                    "file": row["doc_name"],
                    "page": int(row["page_number"]),
                    "score": round(float(row["similarity"]), 4),
                    "chunk_id": row["chunk_id"],
                }
                for _, row in combined.iterrows()
            ]
        else:
            combined = pd.DataFrame()
            context = ""
            sources = []

        source_table = _render_source_table(sources)
        print(f"    Retrieved {len(sources)} unique chunks")
        return {
            "retrieved_context": context,
            "sources": sources,
            "source_table": source_table,
        }

    def synthesise_node(state: ResearchState) -> dict:
        """Produce a grounded answer from all retrieved evidence."""
        print("\n>>> NODE: SYNTHESISE")

        task_type = state.get("task_type", "Question Answering")
        system_parts = [
            "You are a technical research synthesiser.",
            f"Current task type: {task_type}.",
            "",
            "Rules:",
            "1. Answer using ONLY the provided context.",
            "2. Do not use outside knowledge.",
            "3. Every substantive claim must be supported by the retrieved context.",
        ]
        if task_type != "Structured Extraction":
            system_parts.append(
                "4. Cite sources inline where appropriate, e.g. [Source: filename.pdf | Page X]."
            )
            system_parts.append("5. If evidence is insufficient, explicitly say so.")
            system_parts.append(f"6. {_task_output_instruction(task_type)}")
        else:
            system_parts.append(
                f"4. If a field is unsupported, set it to \"{EXTRACTION_MISSING_VALUE}\"."
            )
            system_parts.append(f"5. {_task_output_instruction(task_type)}")

        system = "\n".join(system_parts)

        user_parts = [
            f"Original question: {state['query']}",
            f"Task type: {task_type}",
            f"Planned sub-questions: {json.dumps(state.get('sub_questions', []), ensure_ascii=False)}",
            f"\nRetrieved evidence:\n{state.get('retrieved_context', '')}",
        ]

        if task_type == "Structured Extraction":
            user_parts.append(f"\nRequired JSON fields:\n{extraction_field_bullets()}")

        if state.get("iteration", 0) > 0 and state.get("critique"):
            user_parts.append(
                f"\n--- PREVIOUS DRAFT ---\n{state.get('draft_answer', '')}"
                f"\n\n--- CRITIQUE FEEDBACK ---\n{state['critique']}"
                "\n\nPlease produce an improved answer that addresses the critique."
            )

        answer = call_llm(system, "\n".join(user_parts))
        print(f"    Draft produced (iteration {state.get('iteration', 0)})")
        return {"draft_answer": answer}

    def critique_node(state: ResearchState) -> dict:
        """Score the draft answer on completeness, grounding, and clarity."""
        print("\n>>> NODE: CRITIQUE")

        task_type = state.get("task_type", "Question Answering")
        system = (
            "You are a strict quality evaluator for technical document tasks.\n\n"
            f"Task type: {task_type}. Judge the answer against both the question and the requested output format.\n\n"
            "Score on three dimensions (each 0-10):\n"
            "  - Completeness: Does it address all parts of the question?\n"
            "  - Grounding: Is every claim supported by the retrieved context?\n"
            "  - Clarity: Is it well-structured and easy to follow?\n\n"
            "Return ONLY valid JSON:\n"
            '{\n  "completeness": <score>,\n  "grounding": <score>,\n'
            '  "clarity": <score>,\n  "overall": <average>,\n'
            '  "feedback": "<specific suggestions>"\n}'
        )

        user = (
            f"Question: {state['query']}\n\n"
            f"Task type: {task_type}\n\n"
            f"Retrieved context:\n{state.get('retrieved_context', '')}\n\n"
            f"Answer to evaluate:\n{state.get('draft_answer', '')}"
        )

        raw = call_llm(system, user)

        try:
            data = json.loads(_clean_json_response(raw))
            score = float(data.get("overall", 0))
            feedback = data.get("feedback", "No specific feedback.")
            critique_text = (
                f"Completeness: {data.get('completeness', '?')}/10\n"
                f"Grounding:    {data.get('grounding', '?')}/10\n"
                f"Clarity:      {data.get('clarity', '?')}/10\n"
                f"Overall:      {score}/10\n"
                f"Feedback:     {feedback}"
            )
            parse_error = False
        except (json.JSONDecodeError, TypeError, ValueError):
            score = 0.0
            critique_text = f"Could not parse critique. Raw:\n{raw}"
            parse_error = True

        iteration = state.get("iteration", 0) + 1
        print(f"    Score: {score}/10 | Iteration: {iteration}/{max_iterations}")
        return {
            "critique": critique_text,
            "quality_score": score,
            "critique_parse_error": parse_error,
            "iteration": iteration,
        }

    def refine_queries_node(state: ResearchState) -> dict:
        """Rewrite sub-questions using the critique feedback before re-retrieval."""
        print("\n>>> NODE: REFINE")

        task_type = state.get("task_type", "Question Answering")
        current_sub_questions = state.get("sub_questions") or [state["query"]]
        system = (
            "You improve retrieval plans for a document-grounded research agent.\n\n"
            f"Task type: {task_type}.\n"
            "Given the original question, the current sub-questions, and critique feedback, rewrite 2-4 focused "
            "sub-questions that target missing evidence, unsupported claims, or comparison gaps.\n\n"
            "Return ONLY a JSON array of strings."
        )
        user = (
            f"Original question: {state['query']}\n\n"
            f"Current sub-questions: {json.dumps(current_sub_questions, ensure_ascii=False)}\n\n"
            f"Current critique:\n{state.get('critique', '')}\n\n"
            f"Current sources:\n{state.get('source_table', '')}"
        )

        raw = call_llm(system, user)
        try:
            refined = json.loads(_clean_json_response(raw))
        except (json.JSONDecodeError, TypeError):
            refined = current_sub_questions

        refined = _normalise_sub_questions(refined, state["query"])
        print(f"    Refined sub-questions: {refined}")
        return {"sub_questions": refined}

    def finalise_node(state: ResearchState) -> dict:
        """Format the final output with metadata."""
        print("\n>>> NODE: FINALISE")

        answer = state.get("draft_answer", "")
        final = (
            f"{'=' * 60}\n"
            f"RESEARCH AGENT - FINAL ANSWER\n"
            f"{'=' * 60}\n"
            f"Query: {state['query']}\n"
            f"Task Type: {state.get('task_type', 'Question Answering')}\n"
            f"Iterations: {state.get('iteration', 1)}\n"
            f"Quality Score: {state.get('quality_score', 'N/A')}/10\n"
            f"{'=' * 60}\n\n"
            f"{answer}\n\n"
            f"{'-' * 60}\n"
            f"CRITIQUE SUMMARY\n{'-' * 60}\n"
            f"{state.get('critique', 'N/A')}\n\n"
            f"{'-' * 60}\n"
            f"RETRIEVED SOURCES\n{'-' * 60}\n"
            f"{state.get('source_table', 'N/A')}\n"
        )
        return {
            "answer": answer,
            "final_answer": final,
        }

    def should_refine(state: ResearchState) -> str:
        score = state.get("quality_score", 10)
        iteration = state.get("iteration", 0)
        critique_parse_error = state.get("critique_parse_error", False)

        if iteration >= max_iterations:
            print("    -> ROUTING: Max iterations reached, finalising")
            return "finalise"
        if critique_parse_error or score < quality_threshold:
            reason = "critique parse error" if critique_parse_error else f"score {score} < {quality_threshold}"
            print(f"    -> ROUTING: {reason}, refining")
            return "refine"
        print("    -> ROUTING: Finalising")
        return "finalise"

    graph = StateGraph(ResearchState)

    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("synthesise", synthesise_node)
    graph.add_node("critique", critique_node)
    graph.add_node("refine_queries", refine_queries_node)
    graph.add_node("finalise", finalise_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "synthesise")
    graph.add_edge("synthesise", "critique")
    graph.add_conditional_edges(
        "critique",
        should_refine,
        {"refine": "refine_queries", "finalise": "finalise"},
    )
    graph.add_edge("refine_queries", "retrieve")
    graph.add_edge("finalise", END)

    return graph.compile()


# -- Convenience helpers -----------------------------------------------


def build_langgraph_pipeline(pdf_folder=PDF_FOLDER, force_recompute=False):
    """Load PDFs, chunk, embed, then build and return the compiled graph."""
    pdf_data = load_all_pdfs(pdf_folder)
    df_chunks = create_document_chunks(pdf_data)
    df_chunks, embeddings = embed_chunks(df_chunks, force_recompute=force_recompute)
    return build_research_graph(df_chunks, embeddings)



def run_research_agent(
    agent,
    query: str,
    *,
    task_type: str = "Question Answering",
    top_k: int = TOP_K,
) -> PipelineResult:
    """Run a query through the research agent and return a PipelineResult."""
    initial_state = {
        "query": query,
        "task_type": task_type,
        "top_k": top_k,
        "sub_questions": [],
        "retrieved_context": "",
        "sources": [],
        "source_table": "",
        "draft_answer": "",
        "answer": "",
        "critique": "",
        "quality_score": 0.0,
        "critique_parse_error": False,
        "iteration": 0,
        "final_answer": "",
    }
    state = agent.invoke(initial_state)

    source_chunks = [
        SourceChunk(
            file=s["file"],
            page=s["page"],
            score=s.get("score"),
            chunk_id=s.get("chunk_id", ""),
        )
        for s in state.get("sources", [])
    ]

    answer_text = state.get("answer") or state.get("final_answer") or ""
    metadata = {
        "backend": "langgraph",
        "retrieval_mode": "cosine+crossencoder" if RERANK_ENABLED else "cosine_only",
        "rerank_enabled": RERANK_ENABLED,
        "chunking_style": "character",
        "top_k": top_k,
        "quality_score": state.get("quality_score"),
        "iterations": state.get("iteration"),
        "sub_questions": state.get("sub_questions", []),
        "critique": state.get("critique", ""),
    }

    if task_type == "Structured Extraction":
        answer_text, extraction_meta = validate_and_format_extraction(answer_text, attempt_repair=True)
        metadata.update(extraction_meta)

    return PipelineResult(
        answer=answer_text,
        sources=source_chunks,
        metadata=metadata,
    )
