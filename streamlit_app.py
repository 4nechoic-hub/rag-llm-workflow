# -*- coding: utf-8 -*-
"""
Streamlit UI for RAG Pipeline Demo
Interactive web interface for querying technical documents using
three different RAG implementations.

Run with:  streamlit run streamlit_app.py

Author: Tingyi Zhang
"""

import os
import sys
import time
import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ── Load environment ──
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="RAG Pipeline Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# CUSTOM CSS
# ==========================================================
st.markdown("""
<style>
    /* Clean up the main area */
    .block-container { padding-top: 2rem; }

    /* Source cards */
    .source-card {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        border-left: 4px solid #4C78A8;
        font-size: 0.85em;
    }
    .source-card-langgraph {
        border-left-color: #F58518;
    }
    .source-card-llamaindex {
        border-left-color: #54A24B;
    }

    /* Pipeline badge colours */
    .badge-manual { background-color: #4C78A8; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.8em; }
    .badge-langgraph { background-color: #F58518; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.8em; }
    .badge-llamaindex { background-color: #54A24B; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.8em; }

    /* Metric highlight */
    .metric-box {
        background: #f8f9fb;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================================
# SIDEBAR
# ==========================================================
with st.sidebar:
    st.title("🔍 RAG Pipeline Demo")
    st.caption("Compare three RAG implementations side by side")

    st.divider()

    # API key check
    if api_key:
        st.success("API key loaded from .env", icon="✅")
    else:
        api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.success("API key set", icon="✅")
        else:
            st.warning("Enter your OpenAI API key to continue", icon="⚠️")

    st.divider()

    # Pipeline selection
    pipeline_choice = st.radio(
        "Select Pipeline",
        ["Manual Pipeline", "LangGraph Agent", "LlamaIndex", "Compare All"],
        help="Choose which RAG implementation to query",
    )

    st.divider()

    # Task type
    task_type = st.radio(
        "Task Type",
        ["Question Answering", "Structured Extraction", "Document Comparison"],
    )

    st.divider()

    # Settings
    with st.expander("⚙️ Settings", expanded=False):
        top_k = st.slider("Top-K chunks to retrieve", 3, 10, 5)
        show_sources = st.toggle("Show retrieved sources", value=True)
        show_timing = st.toggle("Show latency", value=True)

    st.divider()

    # PDF folder status
    pdf_folder = Path("pdfs")
    if pdf_folder.exists():
        pdf_files = list(pdf_folder.glob("*.pdf"))
        st.info(f"📄 {len(pdf_files)} PDF(s) in `pdfs/` folder", icon="📁")
        for f in pdf_files:
            st.caption(f"  • {f.name}")
    else:
        st.error("No `pdfs/` folder found. Create one and add your PDFs.")

    st.divider()
    st.caption("Built by Tingyi Zhang")
    st.caption("[GitHub](https://github.com/4nechoic-hub/rag-llm-workflow)")


# ==========================================================
# PIPELINE LOADING (cached)
# ==========================================================

@st.cache_resource(show_spinner="Loading Manual Pipeline...")
def load_manual_pipeline():
    """Load and cache the manual RAG pipeline."""
    from rag_pipeline_spyder import (
        load_all_pdfs, create_document_chunks, embed_chunks,
        answer_question, extract_structured_summary, compare_documents,
        retrieve_top_k, format_context
    )
    pdf_data = load_all_pdfs("pdfs")
    df_chunks = create_document_chunks(pdf_data)
    df_chunks, embeddings = embed_chunks(df_chunks, force_recompute=False)
    return df_chunks, embeddings


@st.cache_resource(show_spinner="Loading LangGraph Agent...")
def load_langgraph_agent():
    """Load and cache the LangGraph research agent."""
    from langgraph_research_agent import (
        load_all_pdfs, create_document_chunks, embed_chunks,
        build_research_graph
    )
    import langgraph_research_agent as lg_module

    pdf_data = load_all_pdfs("pdfs")
    df_chunks = create_document_chunks(pdf_data)
    try:
        df_chunks, embeddings = embed_chunks(df_chunks, force_recompute=False)
    except Exception:
        df_chunks, embeddings = embed_chunks(df_chunks, force_recompute=True)

    lg_module.df_chunks = df_chunks
    lg_module.embeddings = embeddings
    research_agent = build_research_graph()
    return research_agent


@st.cache_resource(show_spinner="Loading LlamaIndex Pipeline...")
def load_llamaindex_pipeline():
    """Load and cache the LlamaIndex index."""
    from llamaindex_rag_pipeline import build_index, create_query_engine, QA_PROMPT
    index = build_index(force_rebuild=False)
    return index


# ==========================================================
# QUERY FUNCTIONS
# ==========================================================

def query_manual(query, task, top_k):
    """Run a query through the manual pipeline."""
    df_chunks, embeddings = load_manual_pipeline()

    from rag_pipeline_spyder import (
        answer_question, extract_structured_summary, compare_documents
    )

    t0 = time.time()
    if task == "Question Answering":
        answer, retrieved = answer_question(query, df_chunks, embeddings, top_k=top_k)
    elif task == "Structured Extraction":
        answer, retrieved = extract_structured_summary(query, df_chunks, embeddings, top_k=top_k)
    else:
        answer, retrieved = compare_documents(query, df_chunks, embeddings, top_k=top_k)
    latency = time.time() - t0

    sources = []
    for _, row in retrieved.iterrows():
        sources.append({
            "file": row["doc_name"],
            "page": row["page_number"],
            "score": round(row["similarity"], 4),
            "chunk_id": row["chunk_id"],
        })

    return answer, sources, latency


def query_langgraph(query, task, top_k):
    """Run a query through the LangGraph agent."""
    research_agent = load_langgraph_agent()

    initial_state = {
        "query": query,
        "sub_questions": [],
        "retrieved_context": "",
        "source_table": "",
        "draft_answer": "",
        "critique": "",
        "quality_score": 0.0,
        "iteration": 0,
        "final_answer": "",
    }

    t0 = time.time()
    result = research_agent.invoke(initial_state)
    latency = time.time() - t0

    answer = result.get("draft_answer", "No answer generated.")
    critique = result.get("critique", "")
    quality = result.get("quality_score", 0)
    iterations = result.get("iteration", 1)

    # Add agent metadata to answer
    meta = f"\n\n---\n**Agent Metadata:** Quality Score: {quality}/10 | Iterations: {iterations}"
    if critique:
        meta += f"\n\n**Critique:**\n{critique}"

    return answer + meta, [], latency


def query_llamaindex(query, task, top_k):
    """Run a query through the LlamaIndex pipeline."""
    index = load_llamaindex_pipeline()

    from llamaindex_rag_pipeline import (
        create_query_engine, QA_PROMPT, EXTRACTION_PROMPT, COMPARISON_PROMPT
    )

    if task == "Question Answering":
        prompt = QA_PROMPT
    elif task == "Structured Extraction":
        prompt = EXTRACTION_PROMPT
    else:
        prompt = COMPARISON_PROMPT

    engine = create_query_engine(index, prompt_template=prompt, top_k=top_k)

    t0 = time.time()
    response = engine.query(query)
    latency = time.time() - t0

    answer = str(response)

    sources = []
    for node in response.source_nodes:
        meta = node.metadata
        sources.append({
            "file": meta.get("file_name", "unknown"),
            "page": meta.get("page_label", "?"),
            "score": round(node.score, 4) if node.score else None,
            "chunk_id": "",
        })

    return answer, sources, latency


# ==========================================================
# DISPLAY HELPERS
# ==========================================================

def display_result(pipe_name, answer, sources, latency, show_sources, show_timing):
    """Display a single pipeline result."""

    badge_class = {
        "Manual Pipeline": "badge-manual",
        "LangGraph Agent": "badge-langgraph",
        "LlamaIndex": "badge-llamaindex",
    }
    card_class = {
        "Manual Pipeline": "",
        "LangGraph Agent": "source-card-langgraph",
        "LlamaIndex": "source-card-llamaindex",
    }

    # Header with badge
    badge = badge_class.get(pipe_name, "badge-manual")
    st.markdown(f'<span class="{badge}">{pipe_name}</span>', unsafe_allow_html=True)

    if show_timing:
        st.caption(f"⏱️ {latency:.2f}s")

    # Answer
    st.markdown(answer)

    # Sources
    if show_sources and sources:
        with st.expander(f"📎 Sources ({len(sources)} chunks)", expanded=False):
            for s in sources:
                score_str = f" — Score: {s['score']}" if s.get('score') else ""
                css_class = f"source-card {card_class.get(pipe_name, '')}"
                st.markdown(
                    f'<div class="{css_class}">'
                    f'<strong>{s["file"]}</strong> — Page {s["page"]}{score_str}'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ==========================================================
# MAIN APP
# ==========================================================

# Title
st.title("🔍 RAG Pipeline Demo")
st.caption("Ask questions about your technical documents using three different RAG approaches")

# Check prerequisites
if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to get started.")
    st.stop()

if not pdf_folder.exists() or not list(pdf_folder.glob("*.pdf")):
    st.warning("No PDFs found. Add PDF files to the `pdfs/` folder and restart.")
    st.stop()

# Query input
query = st.text_area(
    "Enter your query",
    placeholder="e.g. What experimental setup was used in this study?",
    height=80,
)

col_btn, col_example = st.columns([1, 3])
with col_btn:
    run_button = st.button("🚀 Run Query", type="primary", use_container_width=True)
with col_example:
    st.caption("💡 Example queries: *What measurement techniques were used?* · "
               "*Summarise the methodology* · *Compare the experimental approaches*")

st.divider()

# Run query
if run_button and query.strip():

    if pipeline_choice == "Compare All":
        # Run all three pipelines side by side
        col1, col2, col3 = st.columns(3)

        with col1:
            with st.spinner("Manual Pipeline..."):
                try:
                    ans, src, lat = query_manual(query, task_type, top_k)
                    display_result("Manual Pipeline", ans, src, lat, show_sources, show_timing)
                except Exception as e:
                    st.error(f"Manual Pipeline error: {e}")

        with col2:
            with st.spinner("LangGraph Agent..."):
                try:
                    ans, src, lat = query_langgraph(query, task_type, top_k)
                    display_result("LangGraph Agent", ans, src, lat, show_sources, show_timing)
                except Exception as e:
                    st.error(f"LangGraph Agent error: {e}")

        with col3:
            with st.spinner("LlamaIndex..."):
                try:
                    ans, src, lat = query_llamaindex(query, task_type, top_k)
                    display_result("LlamaIndex", ans, src, lat, show_sources, show_timing)
                except Exception as e:
                    st.error(f"LlamaIndex error: {e}")

    else:
        # Run single pipeline
        pipe_fn = {
            "Manual Pipeline": query_manual,
            "LangGraph Agent": query_langgraph,
            "LlamaIndex": query_llamaindex,
        }

        with st.spinner(f"Running {pipeline_choice}..."):
            try:
                ans, src, lat = pipe_fn[pipeline_choice](query, task_type, top_k)
                display_result(pipeline_choice, ans, src, lat, show_sources, show_timing)
            except Exception as e:
                st.error(f"Error: {e}")

elif run_button:
    st.warning("Please enter a query.")