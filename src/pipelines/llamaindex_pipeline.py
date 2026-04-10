# -*- coding: utf-8 -*-
"""
LlamaIndex RAG Pipeline — framework-based approach.

Uses LlamaIndex's high-level abstractions for document ingestion,
sentence-aware chunking, vector indexing, and query engines.

Author: Tingyi Zhang
"""

from pathlib import Path

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

from src.core.extraction import (
    EXTRACTION_MISSING_VALUE,
    extraction_field_bullets,
    extraction_schema_instruction,
    validate_and_format_extraction,
)
from src.core.types import PipelineResult, sources_from_llamaindex
from src.config import (
    OPENAI_API_KEY,
    CHAT_MODEL,
    EMBEDDING_MODEL,
    TEMPERATURE,
    LI_CHUNK_SIZE,
    LI_CHUNK_OVERLAP,
    TOP_K,
    PDF_FOLDER,
    LLAMAINDEX_STORAGE,
)


# ── Configure LlamaIndex global settings ────────────────────────

def _configure_settings():
    """Set LlamaIndex global LLM, embeddings, and node parser."""
    Settings.llm = LlamaOpenAI(
        model=CHAT_MODEL,
        temperature=TEMPERATURE,
        api_key=OPENAI_API_KEY,
    )
    Settings.embed_model = OpenAIEmbedding(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )
    Settings.node_parser = SentenceSplitter(
        chunk_size=LI_CHUNK_SIZE,
        chunk_overlap=LI_CHUNK_OVERLAP,
    )


# ── Prompt templates ────────────────────────────────────────────

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
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n\nAnswer:"
)

EXTRACTION_PROMPT = PromptTemplate(
    "You are a technical document extraction assistant.\n\n"
    "Rules:\n"
    "1. Use ONLY the provided context.\n"
    f"2. If a field is not supported, return \"{EXTRACTION_MISSING_VALUE}\".\n"
    f"3. {extraction_schema_instruction()}\n\n"
    "Context:\n{context_str}\n\n"
    "Task: {query_str}\n\n"
    "Required JSON fields:\n"
    f"{extraction_field_bullets()}\n\n"
    "JSON:"
)

COMPARISON_PROMPT = PromptTemplate(
    "You are a technical comparison assistant.\n\n"
    "Rules:\n"
    "1. Use ONLY the provided context.\n"
    "2. Do not invent details.\n"
    "3. If comparison points are unsupported, say so clearly.\n"
    "4. Return a structured comparison.\n\n"
    "Context:\n{context_str}\n\n"
    "Comparison request: {query_str}\n\n"
    "Return:\n1. Documents involved\n2. Similarities\n3. Differences\n"
    "4. Key methodological differences\n5. Key findings differences\n"
    "6. Source list\n\nComparison:"
)

CHAT_PROMPT = PromptTemplate(
    "You are a helpful research assistant with access to a collection of "
    "technical documents. Answer questions conversationally using ONLY the "
    "provided context. If you cannot answer from the context, say so.\n\n"
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n\nAnswer:"
)


# ── Index management ────────────────────────────────────────────

def build_index(
    pdf_folder: str = PDF_FOLDER,
    persist_dir: str = LLAMAINDEX_STORAGE,
    force_rebuild: bool = False,
) -> VectorStoreIndex:
    """Build or load a VectorStoreIndex from PDFs."""
    _configure_settings()
    persist_path = Path(persist_dir)

    if persist_path.exists() and not force_rebuild:
        print("Loading persisted index from disk...")
        storage_context = StorageContext.from_defaults(persist_dir=str(persist_path))
        index = load_index_from_storage(storage_context)
        print(f"  Index loaded from {persist_path}")
        return index

    pdf_path = Path(pdf_folder)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF folder not found: {pdf_path.resolve()}")

    print(f"Loading PDFs from {pdf_path.resolve()}...")
    documents = SimpleDirectoryReader(
        input_dir=str(pdf_path),
        required_exts=[".pdf"],
    ).load_data()
    print(f"  Loaded {len(documents)} document page(s)")

    print("Building vector index (chunking + embedding)...")
    index = VectorStoreIndex.from_documents(documents, show_progress=True)

    index.storage_context.persist(persist_dir=str(persist_path))
    print(f"  Index persisted to {persist_path}")
    return index


# ── Query engines ───────────────────────────────────────────────

def create_query_engine(index, prompt_template=QA_PROMPT, top_k=TOP_K):
    """Create a query engine with a custom prompt."""
    return index.as_query_engine(
        similarity_top_k=top_k,
        text_qa_template=prompt_template,
    )



def create_retriever(index, top_k=TOP_K):
    """Create a standalone retriever for inspecting chunks."""
    return index.as_retriever(similarity_top_k=top_k)



def _llamaindex_metadata(top_k: int) -> dict:
    """Metadata describing the framework-native LlamaIndex retrieval path."""
    return {
        "backend": "llamaindex",
        "retrieval_mode": "framework_similarity",
        "rerank_enabled": False,
        "chunking_style": "sentence",
        "top_k": top_k,
    }


# ── Task functions ──────────────────────────────────────────────

def answer_question(query, index, top_k=TOP_K) -> PipelineResult:
    """Grounded Q&A using LlamaIndex query engine."""
    engine = create_query_engine(index, prompt_template=QA_PROMPT, top_k=top_k)
    response = engine.query(query)
    return PipelineResult(
        answer=str(response),
        sources=sources_from_llamaindex(response),
        metadata=_llamaindex_metadata(top_k),
    )



def extract_structured(query, index, top_k=6) -> PipelineResult:
    """Structured extraction with shared schema validation and repair."""
    engine = create_query_engine(index, prompt_template=EXTRACTION_PROMPT, top_k=top_k)
    response = engine.query(query)

    raw_output = str(response)
    formatted, extraction_meta = validate_and_format_extraction(raw_output, attempt_repair=True)
    metadata = {**_llamaindex_metadata(top_k), **extraction_meta}

    return PipelineResult(
        answer=formatted,
        sources=sources_from_llamaindex(response),
        metadata=metadata,
    )



def compare_documents(query, index, top_k=8) -> PipelineResult:
    """Document comparison via custom prompt template."""
    engine = create_query_engine(index, prompt_template=COMPARISON_PROMPT, top_k=top_k)
    response = engine.query(query)
    return PipelineResult(
        answer=str(response),
        sources=sources_from_llamaindex(response),
        metadata=_llamaindex_metadata(top_k),
    )
