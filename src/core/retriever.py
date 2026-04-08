# -*- coding: utf-8 -*-
"""Retrieval — cosine-similarity search with optional CrossEncoder reranking."""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.core.embedder import get_embedding
from src.config import TOP_K, RERANK_ENABLED, RERANK_MODEL, RERANK_CANDIDATES


# ── Reranker (lazy-loaded singleton) ────────────────────────────

_reranker = None


def _get_reranker():
    """Load the CrossEncoder model once and cache it."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print(f"  Loading reranker: {RERANK_MODEL}...")
        _reranker = CrossEncoder(RERANK_MODEL)
        print("  Reranker loaded.")
    return _reranker


# ── Core retrieval ──────────────────────────────────────────────

def retrieve_top_k(
    query: str,
    df_chunks: pd.DataFrame,
    embeddings: np.ndarray,
    top_k: int = TOP_K,
    rerank: bool = RERANK_ENABLED,
) -> pd.DataFrame:
    """
    Retrieve top-k most relevant chunks.

    When reranking is enabled:
      1. Fetch RERANK_CANDIDATES chunks by cosine similarity (fast, broad recall)
      2. Score each candidate with a CrossEncoder (slow, precise relevance)
      3. Return the top_k by rerank score

    When reranking is disabled:
      Returns top_k by cosine similarity directly (original behaviour).
    """
    query_emb = np.array(get_embedding(query), dtype=np.float32).reshape(1, -1)
    sims = cosine_similarity(query_emb, embeddings)[0]

    if rerank:
        # Stage 1: broad recall via cosine similarity
        n_candidates = max(top_k, RERANK_CANDIDATES)
        candidate_indices = np.argsort(sims)[::-1][:n_candidates]

        candidates = df_chunks.iloc[candidate_indices].copy()
        candidates["similarity"] = sims[candidate_indices]

        # Stage 2: precision reranking via CrossEncoder
        model = _get_reranker()
        pairs = [(query, text) for text in candidates["text"].tolist()]
        rerank_scores = model.predict(pairs)

        candidates["rerank_score"] = rerank_scores
        results = candidates.nlargest(top_k, "rerank_score")
    else:
        # Original single-stage retrieval
        top_indices = np.argsort(sims)[::-1][:top_k]
        results = df_chunks.iloc[top_indices].copy()
        results["similarity"] = sims[top_indices]

    return results.reset_index(drop=True)


def format_context(retrieved_df: pd.DataFrame) -> str:
    """Format retrieved chunks as a prompt-ready context string."""
    blocks = []
    for _, row in retrieved_df.iterrows():
        block = (
            f"[Source: {row['doc_name']} | Page: {row['page_number']} "
            f"| Chunk: {row['chunk_id']}]\n{row['text']}"
        )
        blocks.append(block)
    return "\n\n" + ("\n" + "-" * 80 + "\n\n").join(blocks)
