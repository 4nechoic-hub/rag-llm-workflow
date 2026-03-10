RAG-Based LLM Workflow for Research and Technical Document Analysis
====================================================================

An applied AI portfolio project demonstrating three complementary
approaches to Retrieval-Augmented Generation (RAG) for analysing
research papers, technical reports, and internal documentation.

Each implementation solves the same core problem — grounded Q&A,
structured extraction, and document comparison over PDF collections —
but showcases different design philosophies and engineering trade-offs.

The project includes a head-to-head evaluation framework with
LLM-as-judge scoring, an interactive Streamlit demo, and a detailed
writeup on the design decisions behind each approach.


Project Overview
----------------

  1. Manual Pipeline        (rag_pipeline_spyder.py)
  2. LangGraph Agent        (langgraph_research_agent.py)
  3. LlamaIndex Pipeline    (llamaindex_rag_pipeline.py)
  4. Evaluation Framework   (evaluate_pipelines.py)
  5. Interactive Demo        (streamlit_app.py)
  6. Design Writeup          (DESIGN_TRADEOFFS.md)

All scripts share the same folder structure and .env configuration.


Architecture
------------

1. MANUAL PIPELINE (rag_pipeline_spyder.py)
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   A from-scratch implementation showing each RAG component explicitly.

   PDFs --> PyMuPDF text extraction
        --> Character-based chunking (1200 chars, 200 overlap)
        --> OpenAI embedding (text-embedding-3-small)
        --> Cosine similarity retrieval (top-k)
        --> Grounded LLM response (gpt-4.1-mini)

   Key features:
   - Full control over every pipeline stage
   - Pickle-based embedding cache for cost efficiency
   - Three task modes: Q&A, structured extraction, document comparison
   - Cell-by-cell execution via # %% markers in VS Code / Spyder


2. LANGGRAPH RESEARCH AGENT (langgraph_research_agent.py)
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   A multi-step agentic workflow with self-correction, built using
   LangGraph's StateGraph for stateful graph-based orchestration.

   Graph flow:

     START
       |
     PLAN ............. LLM decomposes query into 2-4 sub-questions
       |
     RETRIEVE ......... Embedding search per sub-question, deduplicate
       |
     SYNTHESISE ....... LLM produces grounded answer with inline citations
       |
     CRITIQUE ......... Separate LLM call scores completeness, grounding,
       |                clarity (each 0-10)
     ROUTE
       |--- score < 7/10 --> back to RETRIEVE (max 3 iterations)
       |--- score >= 7/10 --> FINALISE
       |
     FINALISE ......... Format final output with quality metadata
       |
     END

   Key features:
   - Query decomposition for better retrieval coverage
   - Self-critique loop with structured JSON scoring
   - Conditional routing (refine vs finalise) based on quality threshold
   - Configurable max iterations to control API costs


3. LLAMAINDEX PIPELINE (llamaindex_rag_pipeline.py)
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   A framework-based approach using LlamaIndex's high-level abstractions.

   PDFs --> SimpleDirectoryReader (auto PDF parsing)
        --> SentenceSplitter (sentence-aware chunking)
        --> OpenAIEmbedding (automatic batched embedding)
        --> VectorStoreIndex (in-memory vector store)
        --> QueryEngine with custom PromptTemplates

   Key features:
   - Minimal boilerplate — LlamaIndex handles chunking, embedding,
     indexing, and retrieval orchestration
   - Index persistence to disk (avoids re-embedding on restart)
   - Custom prompt templates for Q&A, extraction, and comparison
   - Swappable components (LLM, embeddings, vector store)


4. EVALUATION FRAMEWORK (evaluate_pipelines.py)
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   Head-to-head comparison of all three pipelines using LLM-as-judge.

   For each test query, the evaluation runs all three pipelines and
   measures:
   - Answer quality (completeness, grounding, clarity — each 0-10)
   - Retrieval metrics (similarity scores, chunk counts)
   - Latency per query
   - Cross-pipeline agreement (do the three converge on similar answers?)

   Outputs:
   - Console summary table with best pipeline per dimension
   - Detailed CSV (eval_results/evaluation_results.csv)
   - Six comparison charts:
       01_overall_quality.png ... Bar chart of average scores
       02_quality_breakdown.png . Grouped bars (completeness/grounding/clarity)
       03_latency.png .......... Latency comparison
       04_per_query.png ........ Per-query scores across pipelines
       05_agreement.png ........ Cross-pipeline answer agreement
       06_radar.png ............ Radar chart of pipeline profiles


5. INTERACTIVE DEMO (streamlit_app.py)
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   A Streamlit web interface for querying documents interactively.

   Features:
   - Sidebar with pipeline selector, task type, and settings
   - Single pipeline mode or "Compare All" side-by-side view
   - Expandable source cards with retrieval scores
   - Latency display per query
   - Live PDF folder status

   Run with:  streamlit run streamlit_app.py


6. DESIGN WRITEUP (DESIGN_TRADEOFFS.md)
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   A blog-style analysis of the design decisions and trade-offs
   across the three approaches, including:
   - When to use each approach
   - What the evaluation revealed
   - A practical decision framework
   - Key takeaways for production RAG systems


Comparison of Approaches
------------------------

  Aspect              | Manual Pipeline    | LangGraph Agent     | LlamaIndex
  --------------------|--------------------|---------------------|------------------
  Abstraction level   | Low (full control) | Medium (graph nodes)| High (framework)
  Lines of code       | ~500               | ~600                | ~350
  Chunking            | Character-based    | Character-based     | Sentence-aware
  Retrieval           | Single query       | Multi-query (plan)  | Single query
  Self-correction     | No                 | Yes (critique loop) | No
  Index persistence   | Pickle cache       | Pickle cache        | Built-in storage
  Best for            | Learning/custom    | Agentic workflows   | Production RAG


Setup
-----

Prerequisites:
  - Python 3.10+
  - An OpenAI API key with billing enabled

1. Clone the repository:

     git clone https://github.com/4nechoic-hub/rag-llm-workflow.git
     cd rag-llm-workflow

2. Create a virtual environment (recommended):

     python -m venv venv
     source venv/bin/activate       (Mac/Linux)
     venv\Scripts\activate          (Windows)

3. Install all dependencies:

     pip install PyMuPDF numpy pandas scikit-learn openai python-dotenv \
                 langgraph matplotlib streamlit \
                 llama-index-core llama-index-llms-openai \
                 llama-index-embeddings-openai llama-index-readers-file

4. Create a .env file in the project root:

     OPENAI_API_KEY=sk-your_api_key_here

5. Add PDF documents to the pdfs/ folder:

     rag-llm-workflow/
     |-- .env
     |-- pdfs/
     |   |-- paper1.pdf
     |   |-- paper2.pdf
     |-- (scripts...)


Usage
-----

All Python scripts use # %% cell markers for VS Code / Spyder.
Run cells individually or execute entire scripts.

  Manual Pipeline:      python rag_pipeline_spyder.py
  LangGraph Agent:      python langgraph_research_agent.py
  LlamaIndex Pipeline:  python llamaindex_rag_pipeline.py
  Evaluation:           python evaluate_pipelines.py
  Streamlit Demo:       streamlit run streamlit_app.py

Each script will:
  1. Load and process PDFs from the pdfs/ folder
  2. Create embeddings (cached after first run)
  3. Run example queries and display results

The Streamlit app opens at http://localhost:8501 in your browser.


Cost Notes
----------

This project uses OpenAI's API. Approximate costs per run:

  - Embedding 66 chunks: ~$0.001 (text-embedding-3-small)
  - Single Q&A query:    ~$0.002 (gpt-4.1-mini)
  - LangGraph full run:  ~$0.01-0.03 (multiple LLM calls per iteration)
  - Full evaluation:     ~$0.10-0.20 (5 queries x 3 pipelines + judge calls)

Embeddings are cached after the first run, so subsequent queries
only incur LLM generation costs. A $5 credit is more than sufficient
for extensive testing.


Project Structure
-----------------

  rag-llm-workflow/
  |
  |-- rag_pipeline_spyder.py ........ Manual RAG pipeline (from scratch)
  |-- langgraph_research_agent.py ... LangGraph multi-step research agent
  |-- llamaindex_rag_pipeline.py .... LlamaIndex framework-based pipeline
  |-- evaluate_pipelines.py ......... Head-to-head evaluation with charts
  |-- streamlit_app.py .............. Interactive Streamlit web demo
  |-- DESIGN_TRADEOFFS.md ........... Design trade-offs writeup
  |
  |-- pdfs/ ......................... Your PDF documents go here
  |-- cache/ ........................ Auto-created: embedding cache
  |-- llamaindex_storage/ ........... Auto-created: persisted LlamaIndex index
  |-- eval_results/ ................. Auto-created: evaluation CSV and charts
  |-- .env .......................... Your OpenAI API key (not committed)
  |-- .gitignore
  |-- README.txt


Technologies
------------

  - Python 3.10+
  - OpenAI API (gpt-4.1-mini, text-embedding-3-small)
  - LangGraph (StateGraph, conditional edges, stateful orchestration)
  - LlamaIndex (VectorStoreIndex, SimpleDirectoryReader, PromptTemplate)
  - Streamlit (interactive web UI)
  - Matplotlib (evaluation charts)
  - PyMuPDF (PDF text extraction)
  - scikit-learn (cosine similarity)
  - NumPy, Pandas (data processing)


Author
------

Tingyi Zhang
Postdoctoral Research Associate, UNSW Sydney
GitHub: https://github.com/4nechoic-hub


License
-------

This project is released for portfolio and educational purposes.