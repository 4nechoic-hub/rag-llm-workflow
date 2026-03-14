# -*- coding: utf-8 -*-
"""Embedding — generate and cache OpenAI embeddings for text chunks."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.llm import get_client
from src.config import EMBEDDING_MODEL, CACHE_FOLDER


def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    """Get a single embedding vector from OpenAI."""
    client = get_client()
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding


def embed_chunks(
    df_chunks: pd.DataFrame,
    force_recompute: bool = False,
    cache_folder: str = CACHE_FOLDER,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Create embeddings for all chunks with disk caching.
    Returns (df_chunks, embeddings_array).
    """
    cache_path = Path(cache_folder) / "chunk_embeddings.pkl"

    if cache_path.exists() and not force_recompute:
        print("  Loading cached embeddings...")
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        print(f"  Loaded {len(payload['embeddings'])} cached embeddings.")
        return payload["df_chunks"], payload["embeddings"]

    print(f"  Creating embeddings for {len(df_chunks)} chunks...")
    embeddings = []
    for idx, row in df_chunks.iterrows():
        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"    Embedding chunk {idx+1}/{len(df_chunks)}")
        emb = get_embedding(row["text"])
        embeddings.append(emb)

    embeddings = np.array(embeddings, dtype=np.float32)

    Path(cache_folder).mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"df_chunks": df_chunks, "embeddings": embeddings}, f)
    print(f"  Embeddings cached to {cache_path}")

    return df_chunks, embeddings
