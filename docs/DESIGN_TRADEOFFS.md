# Design Trade-offs: Three Approaches to RAG for Technical Document Analysis

*A practical comparison of building RAG systems from scratch, with agentic workflows, and with production frameworks.*

---

## Why Three Implementations?

When I set out to build a RAG system for analysing research papers, I wanted to answer a question that kept coming up in my transition from experimental physics to applied AI: **what's the right level of abstraction for a given problem?**

In experimental fluid dynamics, I routinely made similar decisions — do you build a custom sensor rig from scratch, use a commercial turnkey system, or design a multi-stage measurement campaign with feedback loops? Each has a place. The same logic applies to LLM application design.

So I built the same RAG pipeline three ways, each representing a different philosophy.

---

## The Three Approaches

### 1. Manual Pipeline — Full Control, Full Understanding

The first implementation uses raw OpenAI API calls, NumPy, pandas, and scikit-learn. Every stage is explicit: PDF extraction with PyMuPDF, character-based chunking with configurable overlap, embedding via the OpenAI API, cosine similarity retrieval, and hand-crafted prompt templates for each task type.

**What this teaches you:**
The manual pipeline forces you to understand what's actually happening at each step. When retrieval quality is poor, you know exactly where to look — is the chunk size too large? Are the embeddings capturing the right semantics? Is the prompt template too vague? There's nowhere to hide behind framework abstractions.

**The trade-off:**
You write significantly more code for the same outcome. The chunking is character-based rather than sentence-aware, which can split mid-sentence. There's no built-in index persistence beyond a pickle cache. And scaling to multiple document types or vector stores would require rewriting core logic.

**When this makes sense:**
Prototyping, learning, or when you need a custom pipeline stage that no framework supports. It's also valuable when debugging — if your LlamaIndex pipeline gives poor answers, being able to replicate the issue in raw code helps isolate the root cause.

---

### 2. LangGraph Agent — Intelligence Through Structure

The second implementation wraps the same retrieval engine in a LangGraph StateGraph with five nodes: Plan, Retrieve, Synthesise, Critique, and Finalise. The graph includes a conditional edge after the Critique node — if the quality score falls below a threshold, the agent loops back to Retrieve with refined queries, up to a configurable maximum number of iterations.

**What this teaches you:**
Agentic design is fundamentally about decomposition and feedback. The planning node breaks a broad question into focused sub-questions, which dramatically improves retrieval coverage. The critique node uses a separate LLM call to score the answer on completeness, grounding, and clarity — then feeds that feedback back into the next synthesis attempt.

This is the reflection pattern in practice: generate, evaluate, refine.

**The trade-off:**
Every iteration costs additional API calls. A single query through the LangGraph agent typically makes 5-8 LLM calls compared to 1-2 in the manual pipeline. Latency scales linearly with iteration count. And the quality improvement from self-critique has diminishing returns — in my testing, the first refinement loop often yields the biggest gain, with subsequent loops producing marginal improvements.

There's also a debugging complexity trade-off. When the agent produces a poor answer, you need to inspect the state at each node to find where it went wrong. Was the planning step too vague? Did retrieval miss key chunks? Did the critique misjudge quality?

**When this makes sense:**
Complex queries that span multiple topics or require synthesising information from different parts of a document collection. The planning step is particularly valuable when the user's question is broad — "compare the experimental approaches across these papers" benefits enormously from decomposition into specific sub-questions about instrumentation, methodology, and analysis.

---

### 3. LlamaIndex Pipeline — Framework Efficiency

The third implementation uses LlamaIndex's high-level abstractions. The entire ingestion-to-query pipeline fits in roughly 350 lines compared to 500+ for the manual version. SimpleDirectoryReader handles PDF parsing, SentenceSplitter does sentence-aware chunking, VectorStoreIndex manages embedding and storage, and the QueryEngine handles retrieval and synthesis.

**What this teaches you:**
Production RAG frameworks exist because the manual pipeline stages are well-understood patterns. LlamaIndex's value isn't in doing something novel — it's in providing tested, composable building blocks that handle edge cases you haven't thought of yet. Sentence-aware chunking respects natural language boundaries. Built-in index persistence means you don't need to manage pickle files. And swapping components (different embedding models, vector stores, or LLMs) requires changing one line rather than rewriting a pipeline stage.

**The trade-off:**
Abstraction comes at the cost of visibility. When retrieval is poor, the debugging path is less direct — you need to understand LlamaIndex's internal node parsing, embedding batching, and retrieval scoring to diagnose issues effectively. The framework also introduces dependency complexity: llama-index-core, llama-index-llms-openai, llama-index-embeddings-openai, and llama-index-readers-file are all separate packages that need to be version-compatible.

**When this makes sense:**
When you're building a RAG system that needs to be maintained, extended, or handed off to other developers. The standardised interfaces make it straightforward to upgrade components, add new document types, or integrate with external vector stores like Pinecone or ChromaDB.

---

## What the Evaluation Revealed

Running the same five test queries through all three pipelines with an LLM-as-judge produced some practical insights:

**Answer quality was surprisingly close across all three.** For straightforward factual questions, all three pipelines retrieved similar chunks and generated comparable answers. The differences emerged on complex, multi-part questions — where the LangGraph agent's planning step gave it a meaningful edge in completeness.

**Grounding scores were highest for the manual and LlamaIndex pipelines.** Both use direct prompt templates that strongly constrain the LLM to retrieved context. The LangGraph agent, with its multi-step synthesis, occasionally introduced slight generalisations — a known risk with agentic workflows where the LLM has more room to interpolate.

**Latency differences were significant.** The manual and LlamaIndex pipelines typically responded in 3-8 seconds. The LangGraph agent took 10-25 seconds depending on how many refinement loops it needed. For interactive use, this matters.

**The cross-pipeline agreement scores were high (typically 7-9/10)**, which is reassuring — it means the three approaches converge on similar answers despite different retrieval and synthesis strategies.

---

## Design Decision Framework

Based on building and evaluating all three, here's how I'd decide which approach to use for a given project:

**Use the manual pipeline when** you need maximum control, are working on a novel retrieval strategy, or are building a pipeline component that doesn't fit into existing frameworks. It's also the right starting point for learning.

**Use a LangGraph-style agent when** query complexity is high and users benefit from multi-step reasoning. The planning and critique loop adds real value for broad research questions, document comparison tasks, and situations where a single retrieval pass might miss relevant context. Accept the latency and cost trade-off.

**Use LlamaIndex (or a similar framework) when** you're building for production, need component swappability, or are working in a team where standardised interfaces reduce onboarding friction. The framework handles the boring but important details — index persistence, embedding batching, document metadata propagation — so you can focus on the application logic.

---

## Key Takeaways

1. **Abstraction level should match the problem.** Don't use an agentic framework for simple Q&A, and don't build from scratch when a framework handles your use case well.

2. **Self-critique loops have diminishing returns.** One refinement iteration often captures most of the improvement. Budget for it in latency-sensitive applications.

3. **Retrieval quality matters more than generation sophistication.** All three approaches use the same LLM for generation, but the differences in answer quality came primarily from retrieval strategy — how questions were decomposed, how chunks were split, and how many relevant passages were surfaced.

4. **Evaluate systematically.** LLM-as-judge isn't perfect, but it's far better than eyeballing outputs. Running the same queries through multiple pipelines and scoring on specific dimensions (completeness, grounding, clarity) reveals patterns that informal testing misses.

5. **The "right" approach depends on your constraints.** Cost, latency, maintainability, and team familiarity all factor into the decision. There is no universally best RAG architecture — only the best one for your specific context.

---

*This writeup accompanies the [RAG Pipeline project](https://github.com/4nechoic-hub/rag-llm-workflow), which contains all three implementations along with an evaluation framework and interactive Streamlit demo.*