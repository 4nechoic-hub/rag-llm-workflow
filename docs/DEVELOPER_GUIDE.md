# Developer Guide

This guide covers everything you need to set up, run, extend, and debug the RAG LLM Workflow project.

For the project overview and recruiter-facing summary, see the main [README](../README.md). For architectural rationale, see [DESIGN_TRADEOFFS.md](DESIGN_TRADEOFFS.md).

---

## Table of contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Project layout](#project-layout)
- [Configuration reference](#configuration-reference)
- [Running the pipelines](#running-the-pipelines)
- [Running the evaluation](#running-the-evaluation)
- [How retrieval and reranking work](#how-retrieval-and-reranking-work)
- [The PipelineResult contract](#the-pipelineresult-contract)
- [Adding a fourth pipeline](#adding-a-fourth-pipeline)
- [Switching models](#switching-models)
- [Extending the evaluation](#extending-the-evaluation)
- [Debugging common issues](#debugging-common-issues)
- [Cost management](#cost-management)

---

## Prerequisites

- **Python 3.10+**
- **OpenAI API key** with billing enabled ([get one here](https://platform.openai.com/api-keys))
- ~2 GB disk space for the CrossEncoder reranking model (downloaded on first run)
- PDF documents to analyse (place in `pdfs/`)

---

## Setup

```bash
# Clone
git clone https://github.com/4nechoic-hub/rag-llm-workflow.git
cd rag-llm-workflow

# Virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\Activate.ps1       # Windows PowerShell

# Dependencies
pip install -r requirements.txt

# API key
echo "OPENAI_API_KEY=sk-your_key_here" > .env

# Add documents
mkdir -p pdfs
# copy your PDF files into pdfs/
```

Verify everything works:

```bash
# Quick smoke test — should print an answer without errors
python -c "
from src.pipelines.manual_pipeline import build_manual_pipeline, answer_question
df, emb = build_manual_pipeline()
result = answer_question('What is this document about?', df, emb)
print(result.answer[:200])
"
```

---

## Project layout

```
rag-llm-workflow/
├── app/
│   └── streamlit_app.py           # Streamlit UI (chatbot + explorer)
├── assets/readme/                 # README images (hero banner, UI previews)
├── docs/
│   ├── DESIGN_TRADEOFFS.md        # Architectural rationale
│   ├── DEVELOPER_GUIDE.md         # This file
│   └── README_SCREENSHOT_PLAYBOOK.md
├── scripts/                       # Screenshot automation
├── src/
│   ├── config.py                  # All constants and paths
│   ├── core/
│   │   ├── chunker.py             # Character-based text chunking
│   │   ├── embedder.py            # OpenAI embeddings with corpus-aware cache
│   │   ├── llm.py                 # OpenAI client wrapper
│   │   ├── pdf_loader.py          # PyMuPDF text extraction
│   │   ├── retriever.py           # Cosine similarity + CrossEncoder reranking
│   │   └── types.py               # PipelineResult, SourceChunk dataclasses
│   ├── evaluation/
│   │   └── evaluate_pipelines.py  # Head-to-head comparison harness
│   └── pipelines/
│       ├── chatbot.py             # Multi-turn conversational RAG
│       ├── langgraph_agent.py     # LangGraph agentic workflow
│       ├── llamaindex_pipeline.py # LlamaIndex framework-based RAG
│       └── manual_pipeline.py     # From-scratch RAG
├── .env                           # Your API key (not committed)
├── .gitignore
├── requirements.txt
└── README.md
```

**Auto-created directories** (gitignored):

- `cache/` — pickle files for embedding vectors
- `llamaindex_storage/` — persisted LlamaIndex index
- `eval_results/` — CSV and PNG evaluation outputs
- `pdfs/` — your PDF corpus

---

## Configuration reference

All settings live in `src/config.py`. No constants are scattered across modules.

| Setting | Default | Purpose |
|---|---|---|
| `CHAT_MODEL` | `gpt-4.1-mini` | OpenAI model for generation |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI model for embeddings |
| `TEMPERATURE` | `0.0` | LLM temperature (deterministic) |
| `CHUNK_SIZE` | `1200` | Characters per chunk (manual + LangGraph) |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `LI_CHUNK_SIZE` | `1024` | LlamaIndex sentence-aware chunk size |
| `LI_CHUNK_OVERLAP` | `200` | LlamaIndex chunk overlap |
| `TOP_K` | `5` | Final number of chunks returned to the LLM |
| `RERANK_ENABLED` | `True` | Enable two-stage retrieval with CrossEncoder |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | CrossEncoder model |
| `RERANK_CANDIDATES` | `15` | Candidates fetched before reranking down to TOP_K |
| `QUALITY_THRESHOLD` | `7` | LangGraph critique score threshold (out of 10) |
| `MAX_ITERATIONS` | `3` | Maximum LangGraph refinement loops |

To change a setting, edit `src/config.py` directly. No environment variables needed beyond `OPENAI_API_KEY`.

---

## Running the pipelines

### Streamlit demo (recommended)

```bash
streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501` with two modes:

- **Document Chatbot** — multi-turn Q&A with source citations
- **Pipeline Explorer** — run one pipeline or compare all three side by side

### Python API

All three pipelines return a `PipelineResult` object:

```python
from src.pipelines.manual_pipeline import build_manual_pipeline, answer_question

df_chunks, embeddings = build_manual_pipeline()
result = answer_question("What methods were used?", df_chunks, embeddings)

print(result.answer)          # The generated answer
print(result.sources)         # List of SourceChunk objects
print(result.source_dicts)    # Same as above, as plain dicts
print(result.num_chunks)      # Number of sources retrieved
print(result.avg_similarity)  # Mean similarity score
```

```python
from src.pipelines.langgraph_agent import build_langgraph_pipeline, run_research_agent

agent = build_langgraph_pipeline()
result = run_research_agent(agent, "Compare the experimental approaches.")

print(result.answer)
print(result.metadata["quality_score"])   # Critique score (0-10)
print(result.metadata["iterations"])      # Refinement loop count
print(result.metadata["sub_questions"])   # Decomposed sub-questions
print(result.metadata["critique"])        # Final critique text
```

```python
from src.pipelines.llamaindex_pipeline import build_index, answer_question

index = build_index()
result = answer_question("Summarise the methodology.", index)
print(result.answer)
```

---

## Running the evaluation

```bash
python -m src.evaluation.evaluate_pipelines
```

This runs 5 test queries through all 3 pipelines (15 pipeline runs), scores each answer with an LLM-as-judge, and measures cross-pipeline agreement. Expect it to take 3–5 minutes and cost ~$0.10–0.20 in API calls.

**Outputs:**

| File | Contents |
|---|---|
| `eval_results/evaluation_results.csv` | Per-query scores, latency, similarity metrics |
| `eval_results/01_overall_quality.png` | Average quality by pipeline |
| `eval_results/02_quality_breakdown.png` | Completeness, grounding, clarity breakdown |
| `eval_results/03_latency.png` | Average query latency |
| `eval_results/04_per_query.png` | Per-query score comparison |
| `eval_results/05_agreement.png` | Cross-pipeline answer agreement |
| `eval_results/06_radar.png` | Radar profile (quality + speed) |

---

## How retrieval and reranking work

The retriever (`src/core/retriever.py`) uses a two-stage pipeline when `RERANK_ENABLED = True`:

```
Query
  │
  ▼
Stage 1: Cosine similarity over all chunk embeddings
  → Returns RERANK_CANDIDATES (default 15) broad candidates
  │
  ▼
Stage 2: CrossEncoder scores each (query, chunk_text) pair
  → Returns TOP_K (default 5) by rerank score
  │
  ▼
Passed to LLM as context
```

**Why two stages?** Cosine similarity over embeddings is fast but can miss nuance — it compares compressed vector representations. The CrossEncoder reads the full query and chunk text together, giving much more precise relevance judgments, but is too slow to run over the entire corpus. The two-stage approach gets the best of both: broad recall from embeddings, then precision filtering from the CrossEncoder.

**The CrossEncoder model** (`ms-marco-MiniLM-L-6-v2`) is loaded lazily on first use and cached in memory for the session. The initial download is ~80 MB.

**To disable reranking** (e.g., for faster iteration during development):

```python
# In src/config.py
RERANK_ENABLED = False
```

Or per-call:

```python
from src.core.retriever import retrieve_top_k
results = retrieve_top_k(query, df_chunks, embeddings, rerank=False)
```

---

## The PipelineResult contract

All three pipelines return a `PipelineResult` dataclass defined in `src/core/types.py`:

```python
@dataclass
class PipelineResult:
    answer: str                              # The generated response
    sources: list[SourceChunk]               # Retrieved source chunks
    latency: float = 0.0                     # Wall-clock time (set by caller)
    metadata: dict = field(default_factory=dict)  # Pipeline-specific extras

@dataclass
class SourceChunk:
    file: str                   # Source filename
    page: int | str             # Page number
    score: float | None = None  # Similarity or rerank score
    chunk_id: str = ""          # Chunk identifier
```

This means the Streamlit app, evaluation harness, and any future consumers never need to branch on pipeline name. They all read `result.answer`, `result.sources`, etc.

**Helper properties:**

- `result.source_dicts` — sources as a list of plain dicts
- `result.num_chunks` — number of sources
- `result.avg_similarity` — mean score across sources
- `result.max_similarity` — highest score

**Converter functions** (for building `PipelineResult` inside pipelines):

- `sources_from_dataframe(df)` — for manual/LangGraph (converts a retrieved DataFrame)
- `sources_from_llamaindex(response)` — for LlamaIndex (converts response.source_nodes)

---

## Adding a fourth pipeline

To add a new pipeline (e.g., a hybrid BM25 + dense retrieval approach):

1. **Create `src/pipelines/your_pipeline.py`** with at least an `answer_question()` function that returns a `PipelineResult`.

2. **Import from shared core:**

   ```python
   from src.core.types import PipelineResult, sources_from_dataframe
   from src.core.retriever import retrieve_top_k, format_context
   from src.core.llm import call_llm
   ```

3. **Register it in the Streamlit app** (`app/streamlit_app.py`):
   - Add a loader function with `@st.cache_resource`
   - Add a `query_your_pipeline()` function
   - Add it to the sidebar radio and the Compare All column layout

4. **Register it in the evaluation harness** (`src/evaluation/evaluate_pipelines.py`):
   - Add a `run_your_pipeline()` function
   - Add it to the `PIPELINES` dict, `PIPE_ORDER` list, and `COLOURS` dict

5. **Update `docs/DESIGN_TRADEOFFS.md`** with the rationale for the new approach.

---

## Switching models

### Change the LLM

Edit `src/config.py`:

```python
CHAT_MODEL = "gpt-4o"          # or "gpt-4.1-mini", "gpt-3.5-turbo", etc.
```

All pipelines read from this single constant.

### Change the embedding model

Edit `src/config.py`:

```python
EMBEDDING_MODEL = "text-embedding-3-large"
```

**Important:** Changing the embedding model invalidates the cache. The corpus-aware fingerprint in `embedder.py` includes the model name, so a new cache file will be created automatically on the next run. Old cache files can be deleted safely.

### Change the reranker

Edit `src/config.py`:

```python
RERANK_MODEL = "cross-encoder/ms-marco-TinyBERT-L-2-v2"  # faster, less accurate
# or
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"   # slower, more accurate
```

### Use a non-OpenAI LLM

The `src/core/llm.py` module wraps the OpenAI client. To switch to another provider (e.g., Anthropic, local Ollama), replace the `call_llm()` function in that file. All pipelines call `call_llm()` — no other file touches the LLM client directly.

---

## Extending the evaluation

### Add test queries

Edit the `TEST_QUERIES` list in `src/evaluation/evaluate_pipelines.py`:

```python
TEST_QUERIES = [
    # ... existing queries ...
    {"id": "Q6", "query": "What is the Reynolds number range studied?", "type": "factual"},
    {"id": "Q7", "query": "Is there sufficient data to support the conclusions?", "type": "critical"},
    {"id": "Q8", "query": "What is the capital of France?", "type": "unanswerable"},
]
```

Include a mix of factual, summary, technical, critical, and unanswerable queries to stress-test retrieval and grounding.

### Add evaluation dimensions

To add a new scoring dimension (e.g., "citation accuracy"), edit the `judge_answer()` function's system prompt to include the new dimension, then add the corresponding field to the results row dict and the chart generation.

---

## Debugging common issues

### "No PDFs found"

The `pdfs/` directory must exist at the project root and contain at least one `.pdf` file.

### Stale embedding cache

If you change PDFs or the embedding model and get unexpected results, force a cache refresh:

```python
df_chunks, embeddings = build_manual_pipeline(force_recompute=True)
```

Or delete the `cache/` directory manually.

### Stale LlamaIndex index

If you add or remove PDFs and the LlamaIndex pipeline returns outdated results:

```python
index = build_index(force_rebuild=True)
```

Or delete the `llamaindex_storage/` directory.

### LangGraph agent loops forever

The agent is capped at `MAX_ITERATIONS` (default 3) refinement loops. If critique scores are consistently low, check that the PDFs contain content relevant to the test queries. The quality threshold (`QUALITY_THRESHOLD = 7`) can be lowered for experimentation.

### CrossEncoder download fails

The reranking model is downloaded from HuggingFace on first use. If you're behind a firewall or proxy, set `RERANK_ENABLED = False` in `config.py` to fall back to single-stage cosine retrieval.

### OpenAI rate limits

If you hit rate limits during evaluation (15+ LLM calls in quick succession), the evaluation will print error messages but continue. Re-run to fill in any gaps. For sustained use, consider adding exponential backoff to `src/core/llm.py`.

---

## Cost management

| Operation | Approximate cost |
|---|---|
| Embedding 66 chunks | ~$0.001 |
| Single Q&A query | ~$0.002 |
| LangGraph full run (3–5 LLM calls) | ~$0.01–0.03 |
| Full evaluation (5 queries × 3 pipelines + judge) | ~$0.10–0.20 |
| Chatbot turn | ~$0.002–0.005 |

**Cost-saving tips:**

- Embeddings are cached after the first run — subsequent queries only cost LLM generation
- Set `RERANK_ENABLED = False` during development to skip the CrossEncoder (no API cost, just faster iteration)
- Use `gpt-4.1-mini` (the default) instead of `gpt-4o` for 5–10x lower generation cost
- The LlamaIndex index persists to disk — rebuilding is only needed when PDFs change
