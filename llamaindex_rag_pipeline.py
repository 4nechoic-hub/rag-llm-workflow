# -*- coding: utf-8 -*-
"""
LlamaIndex RAG Pipeline for Research and Technical Document Analysis
A framework-based approach using LlamaIndex's high-level abstractions.

This script demonstrates the same RAG capabilities as the manual pipeline,
but using LlamaIndex's built-in document loading, indexing, retrieval,
and query engine — showing proficiency with production RAG frameworks.

Capabilities:
1. Document ingestion via SimpleDirectoryReader
2. Automatic chunking, embedding, and vector indexing
3. Query engine for grounded Q&A
4. Structured extraction via custom prompts
5. Document comparison
6. Index persistence (save/reload to avoid re-embedding)

Author: Tingyi Zhang
"""

# %% ========== IMPORTS ==========

import os
import sys
import json
from pathlib import Path

# --- Dependency check ---
_required = {
    "llama-index-core": "llama_index.core",
    "llama-index-llms-openai": "llama_index.llms.openai",
    "llama-index-embeddings-openai": "llama_index.embeddings.openai",
    "llama-index-readers-file": "llama_index.readers.file",
    "python-dotenv": "dotenv",
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

from dotenv import load_dotenv

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
    PromptTemplate,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


# %% ========== CONFIGURATION ==========

PDF_FOLDER = "pdfs"
PERSIST_DIR = "llamaindex_storage"     # where the index is persisted to disk
TOP_K = 5

CHAT_MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
TEMPERATURE = 0.0

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200

# --- Load API key ---
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found.\n"
        "Create a .env file with: OPENAI_API_KEY=sk-your_key_here"
    )

# --- Configure LlamaIndex global settings ---
Settings.llm = LlamaOpenAI(
    model=CHAT_MODEL,
    temperature=TEMPERATURE,
    api_key=api_key,
)
Settings.embed_model = OpenAIEmbedding(
    model=EMBEDDING_MODEL,
    api_key=api_key,
)
Settings.node_parser = SentenceSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

print(f"[OK] API key loaded")
print(f"[OK] LLM: {CHAT_MODEL} | Embeddings: {EMBEDDING_MODEL}")
print(f"[OK] Chunk size: {CHUNK_SIZE} | Overlap: {CHUNK_OVERLAP}")


# %% ========== CUSTOM PROMPT TEMPLATES ==========

# Grounded Q&A prompt — constrains the LLM to retrieved context only
QA_PROMPT = PromptTemplate(
    "You are a technical document assistant.\n\n"
    "Rules:\n"
    "1. Answer using ONLY the provided context below.\n"
    "2. Do not use outside knowledge.\n"
    "3. If the answer is not clearly supported by the context, say: "
    "\"Insufficient evidence in retrieved documents.\"\n"
    "4. Be concise but specific.\n"
    "5. Cite source filenames and page numbers where possible.\n"
    "6. End with a short source list.\n\n"
    "Context:\n"
    "{context_str}\n\n"
    "Question: {query_str}\n\n"
    "Answer:"
)

# Structured extraction prompt
EXTRACTION_PROMPT = PromptTemplate(
    "You are a technical document extraction assistant.\n\n"
    "Rules:\n"
    "1. Use ONLY the provided context.\n"
    "2. If a field is not supported by the context, return "
    "\"Not found in retrieved context\".\n"
    "3. Return valid JSON only, no markdown fences.\n\n"
    "Context:\n"
    "{context_str}\n\n"
    "Task: {query_str}\n\n"
    "Extract the following fields as JSON:\n"
    "- title\n"
    "- objective\n"
    "- methodology\n"
    "- experimental_setup\n"
    "- main_findings\n"
    "- limitations\n\n"
    "JSON:"
)

# Comparison prompt
COMPARISON_PROMPT = PromptTemplate(
    "You are a technical comparison assistant.\n\n"
    "Rules:\n"
    "1. Use ONLY the provided context.\n"
    "2. Do not invent details.\n"
    "3. If comparison points are unsupported, say so clearly.\n"
    "4. Return a structured comparison.\n\n"
    "Context:\n"
    "{context_str}\n\n"
    "Comparison request: {query_str}\n\n"
    "Return:\n"
    "1. Documents involved\n"
    "2. Similarities\n"
    "3. Differences\n"
    "4. Key methodological differences\n"
    "5. Key findings differences\n"
    "6. Source list\n\n"
    "Comparison:"
)


# %% ========== BUILD OR LOAD INDEX ==========

def build_index(pdf_folder=PDF_FOLDER, persist_dir=PERSIST_DIR, force_rebuild=False):
    """
    Build a VectorStoreIndex from PDFs, or load from disk if available.

    LlamaIndex handles:
    - PDF text extraction (via SimpleDirectoryReader)
    - Sentence-aware chunking (via SentenceSplitter)
    - Embedding generation (via OpenAIEmbedding)
    - In-memory vector store

    The index is persisted to disk so you only pay for embeddings once.
    """
    persist_path = Path(persist_dir)

    if persist_path.exists() and not force_rebuild:
        print("Loading persisted index from disk...")
        storage_context = StorageContext.from_defaults(persist_dir=str(persist_path))
        index = load_index_from_storage(storage_context)
        print(f"  Index loaded from {persist_path}")
        return index

    # Load documents
    pdf_path = Path(pdf_folder)
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF folder not found: {pdf_path.resolve()}\n"
            "Check your working directory."
        )

    print(f"Loading PDFs from {pdf_path.resolve()}...")
    documents = SimpleDirectoryReader(
        input_dir=str(pdf_path),
        required_exts=[".pdf"],
    ).load_data()

    print(f"  Loaded {len(documents)} document page(s)")

    # Build the vector index (this triggers chunking + embedding)
    print("Building vector index (chunking + embedding)...")
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,
    )

    # Persist to disk
    index.storage_context.persist(persist_dir=str(persist_path))
    print(f"  Index persisted to {persist_path}")

    return index


# %% ========== QUERY ENGINES ==========

def create_query_engine(index, prompt_template=QA_PROMPT, top_k=TOP_K):
    """
    Create a query engine with a custom prompt template.
    The retriever fetches top_k chunks, then the LLM synthesises an answer.
    """
    return index.as_query_engine(
        similarity_top_k=top_k,
        text_qa_template=prompt_template,
    )


def create_retriever(index, top_k=TOP_K):
    """
    Create a standalone retriever for inspecting what chunks are fetched.
    """
    return index.as_retriever(similarity_top_k=top_k)


# %% ========== TASK FUNCTIONS ==========

def answer_question(query, query_engine, retriever=None):
    """
    Answer a question using the LlamaIndex query engine.
    Returns the response text and source nodes.
    """
    response = query_engine.query(query)

    # Extract source info
    sources = []
    for node in response.source_nodes:
        meta = node.metadata
        sources.append({
            "file": meta.get("file_name", "unknown"),
            "page": meta.get("page_label", "?"),
            "score": round(node.score, 4) if node.score else None,
            "preview": node.text[:120] + "..."
        })

    return str(response), sources


def extract_structured(query, index, top_k=6):
    """
    Structured extraction using the extraction prompt template.
    """
    engine = create_query_engine(index, prompt_template=EXTRACTION_PROMPT, top_k=top_k)
    response = engine.query(query)

    raw = str(response)

    # Try to parse as JSON for pretty-printing
    try:
        cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
        parsed = json.loads(cleaned)
        formatted = json.dumps(parsed, indent=2)
    except (json.JSONDecodeError, TypeError):
        formatted = raw

    sources = []
    for node in response.source_nodes:
        meta = node.metadata
        sources.append({
            "file": meta.get("file_name", "unknown"),
            "page": meta.get("page_label", "?"),
            "score": round(node.score, 4) if node.score else None,
        })

    return formatted, sources


def compare_documents(query, index, top_k=8):
    """
    Document comparison using the comparison prompt template.
    """
    engine = create_query_engine(index, prompt_template=COMPARISON_PROMPT, top_k=top_k)
    response = engine.query(query)

    sources = []
    for node in response.source_nodes:
        meta = node.metadata
        sources.append({
            "file": meta.get("file_name", "unknown"),
            "page": meta.get("page_label", "?"),
            "score": round(node.score, 4) if node.score else None,
        })

    return str(response), sources


# %% ========== DISPLAY HELPERS ==========

def print_result(title, text, sources):
    """Pretty-print a result with sources."""
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)
    print(text)
    print(f"\n{'-' * 60}")
    print("SOURCES")
    print("-" * 60)
    for s in sources:
        score_str = f" (score: {s['score']})" if s['score'] else ""
        print(f"  {s['file']} — Page {s['page']}{score_str}")
    print()


# %% ========== BUILD SYSTEM ==========

print("\n" + "=" * 60)
print("BUILDING LLAMAINDEX RAG SYSTEM")
print("=" * 60)

index = build_index(force_rebuild=False)

# Create default query engine and retriever
qa_engine = create_query_engine(index)
retriever = create_retriever(index)

print("\n" + "=" * 60)
print("LLAMAINDEX RAG SYSTEM READY")
print(f"  Query engine : top-{TOP_K} retrieval + {CHAT_MODEL}")
print(f"  Prompt       : custom grounded Q&A template")
print("=" * 60)


# %% ========== QUERY: Ask a Question ==========
# Edit the query and run this cell.

query = "What experimental setup and measurement techniques were used?"

answer, sources = answer_question(query, qa_engine)
print_result("ANSWER", answer, sources)


# %% ========== QUERY: Structured Extraction ==========

query = "Extract the methodology and main findings of this paper."

result, sources = extract_structured(query, index, top_k=6)
print_result("STRUCTURED EXTRACTION", result, sources)


# %% ========== QUERY: Compare Documents ==========

query = "Compare the experimental approaches across the loaded documents."

result, sources = compare_documents(query, index, top_k=8)
print_result("DOCUMENT COMPARISON", result, sources)


# %% ========== BONUS: Inspect retrieved chunks directly ==========
# Useful for debugging retrieval quality.

query = "hot wire calibration procedure"

nodes = retriever.retrieve(query)

print(f"\n{'=' * 60}")
print(f"RAW RETRIEVAL: top {len(nodes)} chunks for '{query}'")
print("=" * 60)
for i, node in enumerate(nodes, 1):
    meta = node.metadata
    score = round(node.score, 4) if node.score else "N/A"
    print(f"\n--- Chunk {i} | Score: {score} | "
          f"{meta.get('file_name', '?')} p.{meta.get('page_label', '?')} ---")
    print(node.text[:300] + "...")


# %% ========== BONUS: Rebuild index (if you add new PDFs) ==========
# Uncomment to force a fresh index build:

# index = build_index(force_rebuild=True)
# qa_engine = create_query_engine(index)
# retriever = create_retriever(index)
# print("Index rebuilt!")
