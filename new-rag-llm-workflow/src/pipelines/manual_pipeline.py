# -*- coding: utf-8 -*-
"""
Manual RAG Pipeline — from-scratch implementation.

Each stage is explicit: PDF extraction, character-based chunking,
OpenAI embedding, cosine-similarity retrieval, and grounded LLM response.

Author: Tingyi Zhang
"""

from src.core.pdf_loader import load_all_pdfs
from src.core.chunker import create_document_chunks
from src.core.embedder import embed_chunks
from src.core.retriever import retrieve_top_k, format_context
from src.core.llm import call_llm
from src.config import PDF_FOLDER, TOP_K


# ── Task functions ──────────────────────────────────────────────

def answer_question(query, df_chunks, embeddings, top_k=TOP_K):
    """Grounded Q&A over retrieved document chunks."""
    retrieved = retrieve_top_k(query, df_chunks, embeddings, top_k=top_k)
    context = format_context(retrieved)

    system_prompt = (
        "You are a technical document assistant.\n\n"
        "Rules:\n"
        "1. Answer using ONLY the provided context.\n"
        "2. Do not use outside knowledge.\n"
        "3. If the answer is not clearly supported, say: "
        "\"Insufficient evidence in retrieved documents.\"\n"
        "4. Be concise but specific.\n"
        "5. At the end, provide a short source list."
    )
    user_prompt = (
        f"Question:\n{query}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Return:\n1. Answer\n2. Key supporting points\n3. Source list"
    )

    answer = call_llm(system_prompt, user_prompt)
    return answer, retrieved


def extract_structured_summary(query, df_chunks, embeddings, top_k=6):
    """Extract a structured JSON-style summary from retrieved chunks."""
    retrieved = retrieve_top_k(query, df_chunks, embeddings, top_k=top_k)
    context = format_context(retrieved)

    system_prompt = (
        "You are a technical document extraction assistant.\n\n"
        "Rules:\n"
        "1. Use ONLY the provided context.\n"
        "2. If a field is not supported, return \"Not found in retrieved context\".\n"
        "3. Return valid JSON only."
    )
    user_prompt = (
        "Task:\nExtract a structured summary from the retrieved context.\n\n"
        "Requested fields:\n"
        "- title\n- objective\n- methodology\n- experimental_setup\n"
        "- main_findings\n- limitations\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Return JSON only."
    )

    content = call_llm(system_prompt, user_prompt)
    return content, retrieved


def compare_documents(query, df_chunks, embeddings, top_k=8):
    """Compare documents using retrieved chunks."""
    retrieved = retrieve_top_k(query, df_chunks, embeddings, top_k=top_k)
    context = format_context(retrieved)

    system_prompt = (
        "You are a technical comparison assistant.\n\n"
        "Rules:\n"
        "1. Use ONLY the provided context.\n"
        "2. Do not invent details.\n"
        "3. If comparison points are unsupported, say so clearly.\n"
        "4. Return a structured comparison."
    )
    user_prompt = (
        f"Task:\nCompare the relevant documents based on the user's request.\n\n"
        f"User request:\n{query}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Return:\n"
        "1. Documents involved\n2. Similarities\n3. Differences\n"
        "4. Key methodological differences\n5. Key findings differences\n"
        "6. Source list"
    )

    content = call_llm(system_prompt, user_prompt)
    return content, retrieved


# ── Convenience: build the RAG system ───────────────────────────

def build_manual_pipeline(pdf_folder=PDF_FOLDER, force_recompute=False):
    """Load PDFs, chunk, embed — returns (df_chunks, embeddings)."""
    pdf_data = load_all_pdfs(pdf_folder)
    df_chunks = create_document_chunks(pdf_data)
    df_chunks, embeddings = embed_chunks(df_chunks, force_recompute=force_recompute)
    return df_chunks, embeddings
