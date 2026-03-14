# -*- coding: utf-8 -*-
"""Retrieval — cosine-similarity search over embedded chunks."""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.core.embedder import get_embedding
from src.config import TOP_K


def retrieve_top_k(
    query: str,
    df_chunks: pd.DataFrame,
    embeddings: np.ndarray,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """Retrieve top-k most relevant chunks using cosine similarity."""
    query_emb = np.array(get_embedding(query), dtype=np.float32).reshape(1, -1)
    sims = cosine_similarity(query_emb, embeddings)[0]
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
