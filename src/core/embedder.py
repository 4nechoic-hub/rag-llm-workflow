# -*- coding: utf-8 -*-
"""Embedding - generate and cache OpenAI embeddings for text chunks."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CACHE_FOLDER, EMBEDDING_MODEL
from src.core.llm import get_client
from src.core.usage import record_embedding_usage



def _extract_embedding_usage(response, model: str) -> dict:
    """Extract usage info from an embedding response and record it."""
    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", prompt_tokens) or prompt_tokens)
    embedding_tokens = prompt_tokens or total_tokens
    return record_embedding_usage(
        model=model,
        embedding_tokens=embedding_tokens,
        embedding_calls=1,
    )



def get_embedding(
    text: str,
    model: str = EMBEDDING_MODEL,
    return_usage: bool = False,
) -> list[float] | tuple[list[float], dict]:
    """Get a single embedding vector from OpenAI."""
    client = get_client()
    response = client.embeddings.create(model=model, input=text)
    usage = _extract_embedding_usage(response, model)
    embedding = response.data[0].embedding
    if return_usage:
        return embedding, usage
    return embedding



def _cache_fingerprint(df_chunks: pd.DataFrame, model: str) -> str:
    """Create a stable cache fingerprint from the corpus content and embedding model."""
    hasher = hashlib.sha256()
    hasher.update(model.encode("utf-8"))

    columns = [col for col in ["doc_name", "page_number", "chunk_id", "text"] if col in df_chunks.columns]
    serialised = df_chunks[columns].fillna("").astype(str)
    for row in serialised.itertuples(index=False, name=None):
        for value in row:
            hasher.update(value.encode("utf-8"))
            hasher.update(b"\x1f")
        hasher.update(b"\x1e")
    return hasher.hexdigest()[:16]



def embed_chunks(
    df_chunks: pd.DataFrame,
    force_recompute: bool = False,
    cache_folder: str = CACHE_FOLDER,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Create embeddings for all chunks with corpus-aware disk caching.

    Returns (df_chunks, embeddings_array).
    """
    fingerprint = _cache_fingerprint(df_chunks, EMBEDDING_MODEL)
    cache_path = Path(cache_folder) / f"chunk_embeddings_{fingerprint}.pkl"

    if cache_path.exists() and not force_recompute:
        print(f"  Loading cached embeddings from {cache_path.name}...")
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        print(f"  Loaded {len(payload['embeddings'])} cached embeddings.")
        return payload["df_chunks"], payload["embeddings"]

    print(f"  Creating embeddings for {len(df_chunks)} chunks...")
    embeddings = []
    for idx, row in df_chunks.iterrows():
        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"  Embedding chunk {idx + 1}/{len(df_chunks)}")
        emb = get_embedding(row["text"])
        embeddings.append(emb)

    embeddings = np.array(embeddings, dtype=np.float32)
    Path(cache_folder).mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(
            {
                "fingerprint": fingerprint,
                "model": EMBEDDING_MODEL,
                "df_chunks": df_chunks,
                "embeddings": embeddings,
            },
            f,
        )

    print(f"  Embeddings cached to {cache_path}")
    return df_chunks, embeddings
