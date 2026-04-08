# -*- coding: utf-8 -*-
"""
Centralised configuration for all RAG pipeline components.
Loads environment variables and defines shared constants.
Author: Tingyi Zhang
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env from project root ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ── API ──
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ── Models ──
CHAT_MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
TEMPERATURE = 0.0

# ── Chunking ──
CHUNK_SIZE = 1200          # characters per chunk (manual + LangGraph)
CHUNK_OVERLAP = 200        # overlap between chunks
LI_CHUNK_SIZE = 1024       # LlamaIndex sentence-aware chunk size
LI_CHUNK_OVERLAP = 200

# ── Retrieval ──
TOP_K = 5

# ── Reranking ──
RERANK_ENABLED = True
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = 15     # retrieve this many candidates before reranking down to TOP_K

# ── LangGraph agent ──
QUALITY_THRESHOLD = 7      # critique score threshold (out of 10)
MAX_ITERATIONS = 3         # max refinement loops

# ── Paths ──
PDF_FOLDER = str(PROJECT_ROOT / "pdfs")
CACHE_FOLDER = str(PROJECT_ROOT / "cache")
LLAMAINDEX_STORAGE = str(PROJECT_ROOT / "llamaindex_storage")
EVAL_OUTPUT_DIR = str(PROJECT_ROOT / "eval_results")

# ── Ensure directories exist ──
for _dir in [CACHE_FOLDER, EVAL_OUTPUT_DIR]:
    os.makedirs(_dir, exist_ok=True)
