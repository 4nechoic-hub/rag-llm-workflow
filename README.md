# RAG-Based LLM Workflow for Research & Technical Document Analysis

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4.1--mini-412991?logo=openai&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_RAG-F58518)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-Framework_RAG-54A24B)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive_Demo-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-Portfolio_%26_Educational-grey)

An applied AI portfolio project demonstrating **three complementary approaches** to Retrieval-Augmented Generation (RAG) for analysing research papers, technical reports, and internal documentation — plus an **interactive conversational chatbot** for multi-turn document Q&A.

Each implementation solves the same core problem — grounded Q&A, structured extraction, and document comparison over PDF collections — but showcases different design philosophies and engineering trade-offs. The project includes a head-to-head evaluation framework with LLM-as-judge scoring, an interactive Streamlit demo, and a detailed writeup on the design decisions behind each approach.

---

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [The Three Pipelines](#the-three-pipelines)
- [Document Chatbot](#document-chatbot)
- [Evaluation Framework](#evaluation-framework)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Cost Notes](#cost-notes)
- [Technologies](#technologies)
- [Author](#author)

---

## Features

- **Three RAG implementations** — manual from-scratch, LangGraph agentic workflow, and LlamaIndex framework-based — each with different levels of abstraction and control
- **Conversational chatbot** — multi-turn document Q&A with conversation history, follow-up support, and source citations
- **Pipeline comparison mode** — side-by-side evaluation of all three approaches on the same query
- **LLM-as-judge evaluation** — automated scoring across completeness, grounding, clarity, latency, and cross-pipeline agreement
- **Interactive Streamlit UI** — two modes: chatbot for exploration and pipeline explorer for comparison
- **Modular architecture** — shared core components (PDF loading, chunking, embedding, retrieval, LLM calls) with no code duplication

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                        Streamlit Web UI                           │
│              ┌──────────────┐  ┌───────────────────┐              │
│              │  💬 Chatbot  │  │  🔬 Pipeline       │              │
│              │     Mode     │  │     Explorer       │              │
│              └──────┬───────┘  └─────────┬─────────┘              │
├─────────────────────┼────────────────────┼────────────────────────┤
│                     │       Pipelines     │                        │
│     ┌───────────────┼───────────┬────────┼────────┐               │
│     │               │           │        │        │               │
│  ┌──▼──┐     ┌──────▼──┐  ┌────▼────┐  ┌▼──────┐ │               │
│  │Chat │     │ Manual  │  │LangGraph│  │LlamaI.│ │               │
│  │ Bot │     │Pipeline │  │  Agent  │  │Pipeli.│ │               │
│  └──┬──┘     └────┬────┘  └────┬────┘  └───┬───┘ │               │
│     │             │            │            │     │               │
├─────┼─────────────┼────────────┼────────────┼─────┼───────────────┤
│     │         Shared Core Components        │     │               │
│     │    ┌─────────┬──────────┬──────────┐  │     │               │
│     └────► PDF     │ Chunker  │ Embedder ◄──┘     │               │
│          │ Loader  │          │          │         │               │
│          └────┬────┴─────┬───┴────┬─────┘         │               │
│               │          │        │                │               │
│          ┌────▼────┐ ┌───▼──────┐ │                │               │
│          │Retriever│ │LLM Client│ │                │               │
│          └─────────┘ └──────────┘ │                │               │
├───────────────────────────────────┼────────────────┼───────────────┤
│          External Services        │                │               │
│     ┌────────────┐  ┌────────────┐│   ┌────────────┐              │
│     │ OpenAI API │  │ PDF Files  ││   │ Disk Cache  │             │
│     │ (GPT-4.1)  │  │ (pdfs/)    ││   │ (cache/)    │             │
│     └────────────┘  └────────────┘│   └────────────┘              │
└────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
rag-llm-workflow/
│
├── src/                           # Source code (Python package)
│   ├── config.py                  # Centralised configuration & constants
│   │
│   ├── core/                      # Shared RAG components
│   │   ├── pdf_loader.py          #   PDF text extraction (PyMuPDF)
│   │   ├── chunker.py             #   Character-based text chunking
│   │   ├── embedder.py            #   OpenAI embedding with disk caching
│   │   ├── retriever.py           #   Cosine-similarity retrieval
│   │   └── llm.py                 #   OpenAI client & LLM call wrappers
│   │
│   ├── pipelines/                 # RAG pipeline implementations
│   │   ├── manual_pipeline.py     #   From-scratch RAG (full control)
│   │   ├── langgraph_agent.py     #   LangGraph agentic workflow
│   │   ├── llamaindex_pipeline.py #   LlamaIndex framework-based RAG
│   │   └── chatbot.py             #   Conversational RAG chatbot
│   │
│   └── evaluation/                # Evaluation framework
│       └── evaluate_pipelines.py  #   Head-to-head pipeline comparison
│
├── app/
│   └── streamlit_app.py           # Streamlit web UI (chatbot + explorer)
│
├── docs/
│   └── DESIGN_TRADEOFFS.md        # Design decisions writeup
│
├── pdfs/                          # Your PDF documents (user-provided)
├── cache/                         # Auto-created: embedding cache
├── llamaindex_storage/            # Auto-created: persisted LlamaIndex index
├── eval_results/                  # Auto-created: evaluation CSV & charts
│
├── requirements.txt               # Python dependencies
├── .env                           # Your OpenAI API key (not committed)
├── .gitignore
└── README.md
```

---

## The Three Pipelines

### 1. Manual Pipeline — Full Control

A from-scratch implementation where every RAG stage is explicit.

```
PDFs → PyMuPDF extraction → Character-based chunking (1200 chars, 200 overlap)
     → OpenAI embedding (text-embedding-3-small) → Cosine similarity retrieval
     → Grounded LLM response (gpt-4.1-mini)
```

**Strengths:** Full transparency over every pipeline stage. When retrieval is poor, you know exactly where to look — chunk size, embedding quality, or prompt design. No framework abstractions to debug through.

**Best for:** Prototyping, learning, custom pipeline stages, and debugging.

### 2. LangGraph Agent — Agentic Reasoning

A multi-step workflow with query decomposition and self-correction.

```
START → PLAN (decompose query into sub-questions)
      → RETRIEVE (embedding search per sub-question, deduplicate)
      → SYNTHESISE (grounded answer with inline citations)
      → CRITIQUE (score completeness, grounding, clarity)
      → ROUTE: score < 7/10 → back to RETRIEVE (max 3 loops)
              score ≥ 7/10 → FINALISE → END
```

**Strengths:** Query decomposition dramatically improves retrieval coverage for complex, multi-part questions. Self-critique loop catches incomplete or poorly grounded answers. Implements the generate-evaluate-refine pattern.

**Best for:** Complex research queries, document comparison, and cases where retrieval coverage matters more than speed.

### 3. LlamaIndex Pipeline — Framework Efficiency

A framework-based approach using LlamaIndex's high-level abstractions.

```
PDFs → SimpleDirectoryReader → SentenceSplitter (sentence-aware chunking)
     → OpenAIEmbedding → VectorStoreIndex → QueryEngine + custom prompts
```

**Strengths:** Minimal boilerplate (~350 lines vs ~500 for manual). Sentence-aware chunking respects natural language boundaries. Built-in index persistence. Swappable components — change LLM, embedding model, or vector store with one line.

**Best for:** Production RAG, team environments, and when maintainability and component swappability are priorities.

### Comparison Matrix

| Aspect | Manual | LangGraph | LlamaIndex |
|---|---|---|---|
| Abstraction level | Low (full control) | Medium (graph nodes) | High (framework) |
| Chunking | Character-based | Character-based | Sentence-aware |
| Retrieval | Single query | Multi-query (plan) | Single query |
| Self-correction | No | Yes (critique loop) | No |
| Index persistence | Pickle cache | Pickle cache | Built-in storage |
| Best for | Learning / custom | Agentic workflows | Production RAG |

---

## Document Chatbot

The chatbot provides a **conversational interface** for multi-turn document Q&A:

- **Follow-up questions** — maintains conversation history so you can drill into topics naturally (e.g. "What methods were used?" → "Can you elaborate on the PIV setup?")
- **Context-aware retrieval** — each message triggers a fresh retrieval pass, but the LLM sees the full conversation for continuity
- **Source citations** — every response includes expandable source cards with document names, page numbers, and similarity scores
- **Token management** — automatically trims old history to stay within context limits
- **Clear & reset** — one-click conversation reset

---

## Evaluation Framework

The evaluation runs all three pipelines head-to-head on a standardised query set:

| Metric | Description |
|---|---|
| Completeness (0-10) | Does the answer address all parts of the question? |
| Grounding (0-10) | Is every claim supported by retrieved context? |
| Clarity (0-10) | Is it well-structured and easy to follow? |
| Latency | Wall-clock time per query |
| Cross-pipeline agreement | Do the three pipelines converge on similar answers? |

**Outputs:**
- `eval_results/evaluation_results.csv` — detailed per-query scores
- Six comparison charts: overall quality, quality breakdown, latency, per-query scores, agreement, and radar profile

---

## Quick Start

### Prerequisites

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys) with billing enabled

### Installation

```bash
# Clone the repository
git clone https://github.com/4nechoic-hub/rag-llm-workflow.git
cd rag-llm-workflow

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-your_api_key_here
```

### Add Your Documents

Place PDF files in the `pdfs/` directory:

```
rag-llm-workflow/
├── pdfs/
│   ├── paper1.pdf
│   ├── paper2.pdf
│   └── ...
```

---

## Usage

### Interactive Demo (Streamlit)

```bash
streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501` with two modes:

- **💬 Document Chatbot** — conversational Q&A with follow-up support
- **🔬 Pipeline Explorer** — single pipeline or side-by-side comparison

### Run Pipelines Directly

```python
# Manual Pipeline
from src.pipelines.manual_pipeline import build_manual_pipeline, answer_question

df_chunks, embeddings = build_manual_pipeline()
answer, sources = answer_question("What methods were used?", df_chunks, embeddings)
print(answer)

# LangGraph Agent
from src.pipelines.langgraph_agent import build_langgraph_pipeline, run_research_agent

agent = build_langgraph_pipeline()
result = run_research_agent(agent, "Compare the experimental approaches.")
print(result["final_answer"])

# LlamaIndex Pipeline
from src.pipelines.llamaindex_pipeline import build_index, answer_question

index = build_index()
answer, sources = answer_question("Summarise the methodology.", index)
print(answer)

# Chatbot
from src.pipelines.chatbot import RAGChatbot
from src.pipelines.manual_pipeline import build_manual_pipeline

df_chunks, embeddings = build_manual_pipeline()
bot = RAGChatbot(df_chunks, embeddings)

response, sources = bot.chat("What measurement techniques were used?")
print(response)

response, sources = bot.chat("Can you go into more detail on the PIV setup?")
print(response)
```

### Run Evaluation

```bash
python -m src.evaluation.evaluate_pipelines
```

---

## Cost Notes

This project uses the OpenAI API. Approximate costs per run:

| Operation | Approximate Cost |
|---|---|
| Embedding 66 chunks | ~$0.001 (text-embedding-3-small) |
| Single Q&A query | ~$0.002 (gpt-4.1-mini) |
| LangGraph full run | ~$0.01–0.03 (multiple LLM calls) |
| Full evaluation | ~$0.10–0.20 (5 queries × 3 pipelines + judge) |
| Chatbot turn | ~$0.002–0.005 per message |

Embeddings are cached after the first run, so subsequent queries only incur LLM generation costs. A **$5 credit** is more than sufficient for extensive testing.

---

## Technologies

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| LLM & Embeddings | OpenAI API (gpt-4.1-mini, text-embedding-3-small) |
| Agentic Framework | LangGraph (StateGraph, conditional edges, stateful orchestration) |
| RAG Framework | LlamaIndex (VectorStoreIndex, SimpleDirectoryReader, PromptTemplate) |
| Web UI | Streamlit (interactive app with chat interface) |
| PDF Processing | PyMuPDF (text extraction) |
| ML / Retrieval | scikit-learn (cosine similarity), NumPy |
| Data Processing | pandas |
| Visualisation | Matplotlib (evaluation charts) |

---

## Author

**Tingyi Zhang**
Postdoctoral Research Associate, UNSW Sydney

- GitHub: [github.com/4nechoic-hub](https://github.com/4nechoic-hub)
- LinkedIn: [linkedin.com/in/tingyi-zhang-au](https://www.linkedin.com/in/tingyi-zhang-au/)

---

## License

This project is released for portfolio and educational purposes.
