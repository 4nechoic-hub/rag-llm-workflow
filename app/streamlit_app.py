# -*- coding: utf-8 -*-
"""
Streamlit UI for RAG Pipeline Demo + Conversational Chatbot

Two modes:
  1. Pipeline Explorer — query documents with any of the three RAG pipelines
  2. Document Chatbot  — conversational Q&A with follow-up support

Run with:  streamlit run app/streamlit_app.py

Author: Tingyi Zhang
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ── Ensure project root is on sys.path ──
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

load_dotenv(project_root / ".env")
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
    .block-container { padding-top: 1.5rem; }
    .source-card {
        background-color: #f0f2f6; border-radius: 8px;
        padding: 10px 14px; margin-bottom: 6px;
        border-left: 4px solid #4C78A8; font-size: 0.85em;
    }
    .source-card-langgraph { border-left-color: #F58518; }
    .source-card-llamaindex { border-left-color: #54A24B; }
    .source-card-chatbot { border-left-color: #E45756; }

    .badge-manual { background-color: #4C78A8; color: white; padding: 3px 10px; border-radius: 12px; font-size: 0.8em; }
    .badge-langgraph { background-color: #F58518; color: white; padding: 3px 10px; border-radius: 12px; font-size: 0.8em; }
    .badge-llamaindex { background-color: #54A24B; color: white; padding: 3px 10px; border-radius: 12px; font-size: 0.8em; }
    .badge-chatbot { background-color: #E45756; color: white; padding: 3px 10px; border-radius: 12px; font-size: 0.8em; }

    /* Chat styling */
    .stChatMessage { max-width: 95%; }
</style>
""", unsafe_allow_html=True)


# ==========================================================
# SIDEBAR
# ==========================================================
with st.sidebar:
    st.title("🔍 RAG Pipeline Demo")
    st.caption("Explore & chat with your technical documents")
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

    # Mode selection
    app_mode = st.radio(
        "Mode",
        ["💬 Document Chatbot", "🔬 Pipeline Explorer"],
        help="Chatbot = conversational Q&A | Explorer = compare pipelines",
    )

    st.divider()

    if app_mode == "🔬 Pipeline Explorer":
        pipeline_choice = st.radio(
            "Select Pipeline",
            ["Manual Pipeline", "LangGraph Agent", "LlamaIndex", "Compare All"],
        )
        st.divider()
        task_type = st.radio(
            "Task Type",
            ["Question Answering", "Structured Extraction", "Document Comparison"],
        )
        st.divider()

    with st.expander("⚙️ Settings", expanded=False):
        top_k = st.slider("Top-K chunks", 3, 10, 5)
        show_sources = st.toggle("Show sources", value=True)
        show_timing = st.toggle("Show latency", value=True)

    st.divider()

    # PDF folder status
    pdf_folder = project_root / "pdfs"
    if pdf_folder.exists():
        pdf_files = list(pdf_folder.glob("*.pdf"))
        st.info(f"📄 {len(pdf_files)} PDF(s) loaded", icon="📁")
        for f in pdf_files:
            st.caption(f"  • {f.name}")
    else:
        st.error("No `pdfs/` folder found.")

    st.divider()
    st.caption("Built by [Tingyi Zhang](https://github.com/4nechoic-hub)")


# ==========================================================
# PIPELINE LOADING (cached)
# ==========================================================

@st.cache_resource(show_spinner="Loading Manual Pipeline...")
def load_manual_pipeline():
    from src.pipelines.manual_pipeline import build_manual_pipeline
    return build_manual_pipeline()


@st.cache_resource(show_spinner="Loading LangGraph Agent...")
def load_langgraph_agent():
    from src.pipelines.langgraph_agent import build_langgraph_pipeline
    return build_langgraph_pipeline()


@st.cache_resource(show_spinner="Loading LlamaIndex Pipeline...")
def load_llamaindex_index():
    from src.pipelines.llamaindex_pipeline import build_index
    return build_index(force_rebuild=False)


@st.cache_resource(show_spinner="Initialising chatbot...")
def load_chatbot():
    from src.pipelines.chatbot import RAGChatbot
    df_chunks, embeddings = load_manual_pipeline()
    return RAGChatbot(df_chunks, embeddings)


# ==========================================================
# QUERY FUNCTIONS — all return PipelineResult
# ==========================================================

def query_manual(query, task, top_k):
    df_chunks, embeddings = load_manual_pipeline()
    from src.pipelines.manual_pipeline import (
        answer_question, extract_structured_summary, compare_documents,
    )
    t0 = time.time()
    if task == "Question Answering":
        result = answer_question(query, df_chunks, embeddings, top_k=top_k)
    elif task == "Structured Extraction":
        result = extract_structured_summary(query, df_chunks, embeddings, top_k=top_k)
    else:
        result = compare_documents(query, df_chunks, embeddings, top_k=top_k)
    result.latency = time.time() - t0
    return result


def query_langgraph(query, task, top_k):
    agent = load_langgraph_agent()
    from src.pipelines.langgraph_agent import run_research_agent

    t0 = time.time()
    result = run_research_agent(agent, query, task_type=task, top_k=top_k)
    result.latency = time.time() - t0
    return result


def query_llamaindex(query, task, top_k):
    index = load_llamaindex_index()
    from src.pipelines.llamaindex_pipeline import (
        answer_question, compare_documents, extract_structured,
    )

    t0 = time.time()
    if task == "Question Answering":
        result = answer_question(query, index, top_k=top_k)
    elif task == "Structured Extraction":
        result = extract_structured(query, index, top_k=top_k)
    else:
        result = compare_documents(query, index, top_k=top_k)
    result.latency = time.time() - t0
    return result


# ==========================================================
# DISPLAY HELPERS — unified for all pipelines
# ==========================================================

BADGE_MAP = {
    "Manual Pipeline": "badge-manual",
    "LangGraph Agent": "badge-langgraph",
    "LlamaIndex": "badge-llamaindex",
}
CARD_MAP = {
    "Manual Pipeline": "",
    "LangGraph Agent": "source-card-langgraph",
    "LlamaIndex": "source-card-llamaindex",
}


def display_result(pipe_name, result, show_sources, show_timing):
    """Display a PipelineResult in the Streamlit UI."""
    st.markdown(
        f'<span class="{BADGE_MAP.get(pipe_name, "badge-manual")}">{pipe_name}</span>',
        unsafe_allow_html=True,
    )
    if show_timing:
        st.caption(f"⏱️ {result.latency:.2f}s")
    st.markdown(result.answer)

    # Retrieval metadata for all pipelines
    if result.metadata:
        retrieval_mode = result.metadata.get("retrieval_mode")
        rerank_enabled = result.metadata.get("rerank_enabled")
        chunking_style = result.metadata.get("chunking_style")
        top_k_value = result.metadata.get("top_k")

        if any(v is not None for v in [retrieval_mode, rerank_enabled, chunking_style, top_k_value]):
            with st.expander("⚙️ Retrieval details", expanded=False):
                if retrieval_mode:
                    st.write(f"Retrieval mode: {retrieval_mode}")
                if rerank_enabled is not None:
                    st.write(f"Reranking: {'Enabled' if rerank_enabled else 'Disabled'}")
                if chunking_style:
                    st.write(f"Chunking style: {chunking_style}")
                if top_k_value is not None:
                    st.write(f"Top-k: {top_k_value}")

    # LangGraph-specific metadata
    if result.metadata and pipe_name == "LangGraph Agent":
        with st.expander("🧠 Agent details", expanded=False):
            quality = result.metadata.get("quality_score")
            iterations = result.metadata.get("iterations")
            if quality is not None:
                st.write(f"Quality score: {quality}/10")
            if iterations is not None:
                st.write(f"Iterations: {iterations}")

            sub_questions = result.metadata.get("sub_questions") or []
            if sub_questions:
                st.markdown("**Planned sub-questions**")
                for item in sub_questions:
                    st.markdown(f"- {item}")

            critique = result.metadata.get("critique")
            if critique:
                st.markdown("**Critique**")
                st.code(critique)

    if show_sources and result.sources:
        card_css = CARD_MAP.get(pipe_name, "")
        with st.expander(f"📎 Sources ({result.num_chunks} chunks)", expanded=False):
            for s in result.sources:
                score_str = f" — Score: {s.score}" if s.score is not None else ""
                css = f"source-card {card_css}"
                st.markdown(
                    f'<div class="{css}"><strong>{s.file}</strong> — Page {s.page}{score_str}</div>',
                    unsafe_allow_html=True,
                )


def display_chat_sources(sources):
    """Display sources in a compact expander inside a chat message."""
    if sources:
        with st.expander(f"📎 {len(sources)} sources retrieved", expanded=False):
            for s in sources:
                score_str = f" — Score: {s['score']}" if s.get("score") else ""
                st.markdown(
                    f'<div class="source-card source-card-chatbot">'
                    f'<strong>{s["file"]}</strong> — Page {s["page"]}{score_str}</div>',
                    unsafe_allow_html=True,
                )


# ==========================================================
# CHATBOT MODE
# ==========================================================

def run_chatbot_mode():
    st.title("💬 Document Chatbot")
    st.caption("Ask questions about your documents — supports follow-up conversations")

    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
        st.stop()
    if not pdf_folder.exists() or not list(pdf_folder.glob("*.pdf")):
        st.warning("No PDFs found. Add PDF files to the `pdfs/` folder and restart.")
        st.stop()

    # Initialise session state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_sources" not in st.session_state:
        st.session_state.chat_sources = {}

    # Clear button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.chat_messages = []
            st.session_state.chat_sources = {}
            # Reset the chatbot's history
            bot = load_chatbot()
            bot.clear_history()
            st.rerun()

    # Display conversation history
    for i, msg in enumerate(st.session_state.chat_messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and show_sources:
                sources = st.session_state.chat_sources.get(i, [])
                display_chat_sources(sources)

    # Chat input
    if prompt := st.chat_input("Ask about your documents..."):
        # Display user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                bot = load_chatbot()
                t0 = time.time()
                response, sources = bot.chat(prompt)
                latency = time.time() - t0

            st.markdown(response)
            if show_timing:
                st.caption(f"⏱️ {latency:.2f}s | Turn {bot.turn_count}")
            if show_sources:
                display_chat_sources(sources)

        # Store response
        msg_idx = len(st.session_state.chat_messages)
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
        st.session_state.chat_sources[msg_idx] = sources


# ==========================================================
# PIPELINE EXPLORER MODE
# ==========================================================

def run_explorer_mode():
    st.title("🔬 Pipeline Explorer")
    st.caption("Query documents using three different RAG approaches — compare side by side")

    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
        st.stop()
    if not pdf_folder.exists() or not list(pdf_folder.glob("*.pdf")):
        st.warning("No PDFs found. Add PDF files to the `pdfs/` folder and restart.")
        st.stop()

    query = st.text_area(
        "Enter your query",
        placeholder="e.g. What experimental setup was used in this study?",
        height=80,
    )

    col_btn, col_example = st.columns([1, 3])
    with col_btn:
        run_button = st.button("🚀 Run Query", type="primary", use_container_width=True)
    with col_example:
        st.caption(
            "💡 Examples: *What measurement techniques were used?* · "
            "*Summarise the methodology* · *Compare the experimental approaches*"
        )
    st.divider()

    if run_button and query.strip():
        if pipeline_choice == "Compare All":
            col1, col2, col3 = st.columns(3)
            for col, (name, fn) in zip(
                [col1, col2, col3],
                [("Manual Pipeline", query_manual), ("LangGraph Agent", query_langgraph), ("LlamaIndex", query_llamaindex)],
            ):
                with col:
                    with st.spinner(f"{name}..."):
                        try:
                            result = fn(query, task_type, top_k)
                            display_result(name, result, show_sources, show_timing)
                        except Exception as e:
                            st.error(f"{name} error: {e}")
        else:
            pipe_fn = {
                "Manual Pipeline": query_manual,
                "LangGraph Agent": query_langgraph,
                "LlamaIndex": query_llamaindex,
            }
            with st.spinner(f"Running {pipeline_choice}..."):
                try:
                    result = pipe_fn[pipeline_choice](query, task_type, top_k)
                    display_result(pipeline_choice, result, show_sources, show_timing)
                except Exception as e:
                    st.error(f"Error: {e}")
    elif run_button:
        st.warning("Please enter a query.")


# ==========================================================
# MAIN
# ==========================================================

if app_mode == "💬 Document Chatbot":
    run_chatbot_mode()
else:
    run_explorer_mode()
