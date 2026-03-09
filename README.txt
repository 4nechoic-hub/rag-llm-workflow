RAG-Based LLM Workflow for Research and Technical Document Analysis
====================================================================

An applied AI portfolio project demonstrating three complementary
approaches to Retrieval-Augmented Generation (RAG) for analysing
research papers, technical reports, and internal documentation.

Each implementation solves the same core problem — grounded Q&A,
structured extraction, and document comparison over PDF collections —
but showcases different design philosophies and engineering trade-offs.


Project Overview
----------------

This project contains three standalone Python scripts, each
implementing a RAG pipeline using a different approach:

  1. Manual Pipeline        (rag_pipeline_spyder.py)
  2. LangGraph Agent        (langgraph_research_agent.py)
  3. LlamaIndex Pipeline    (llamaindex_rag_pipeline.py)

All three scripts share the same folder structure and .env
configuration, so you can run any of them from the same project root.


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

   Dependencies: PyMuPDF, numpy, pandas, scikit-learn, openai


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
   - Full state visibility at every node for debugging

   Dependencies: langgraph, PyMuPDF, numpy, pandas, scikit-learn, openai


3. LLAMAINDEX PIPELINE (llamaindex_rag_pipeline.py)
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   A framework-based approach using LlamaIndex's high-level abstractions
   for production-style RAG.

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
   - Direct retriever access for debugging chunk quality

   Dependencies: llama-index-core, llama-index-llms-openai,
                 llama-index-embeddings-openai, llama-index-readers-file


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
     venv\Scripts\activate          (Windows)
     source venv/bin/activate       (Mac/Linux)

3. Install dependencies:

     For the manual pipeline:
       pip install PyMuPDF numpy pandas scikit-learn openai python-dotenv

     For the LangGraph agent (adds langgraph):
       pip install langgraph

     For the LlamaIndex pipeline:
       pip install llama-index-core llama-index-llms-openai llama-index-embeddings-openai llama-index-readers-file

     Or install everything at once:
       pip install PyMuPDF numpy pandas scikit-learn openai python-dotenv langgraph llama-index-core llama-index-llms-openai llama-index-embeddings-openai llama-index-readers-file

4. Create a .env file in the project root:

     OPENAI_API_KEY=sk-your_api_key_here

5. Add PDF documents to the pdfs/ folder:

     rag-llm-workflow/
     |-- .env
     |-- pdfs/
     |   |-- paper1.pdf
     |   |-- paper2.pdf
     |-- rag_pipeline_spyder.py
     |-- langgraph_research_agent.py
     |-- llamaindex_rag_pipeline.py
     |-- README.txt


Usage
-----

All three scripts use # %% cell markers for VS Code / Spyder.
You can run cells individually (Ctrl+Enter in VS Code with the
Jupyter extension) or execute the entire script.

Manual Pipeline:
  python rag_pipeline_spyder.py

LangGraph Agent:
  python langgraph_research_agent.py

LlamaIndex Pipeline:
  python llamaindex_rag_pipeline.py

Each script will:
  1. Load and process PDFs from the pdfs/ folder
  2. Create embeddings (cached after first run)
  3. Run example queries and display results

Edit the query variables in the example cells to test your own
questions against your documents.


Example Output (LangGraph Agent)
--------------------------------

  >>> NODE: PLAN
      Sub-questions (3):
        1. What measurement instruments were used?
        2. What was the experimental configuration?
        3. What data acquisition parameters were applied?

  >>> NODE: RETRIEVE
      Retrieved 10 unique chunks

  >>> NODE: SYNTHESISE
      Draft produced (iteration 0)

  >>> NODE: CRITIQUE
      Score: 8.3/10 (threshold: 7)
      Iteration: 1/3

      -> ROUTING: Score 8.3 >= 7, finalising

  >>> NODE: FINALISE


Cost Notes
----------

This project uses OpenAI's API. Approximate costs per run:

  - Embedding 66 chunks: ~$0.001 (text-embedding-3-small)
  - Single Q&A query:    ~$0.002 (gpt-4.1-mini)
  - LangGraph full run:  ~$0.01-0.03 (multiple LLM calls per iteration)

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
  |
  |-- pdfs/ ......................... Your PDF documents go here
  |-- cache/ ........................ Auto-created: embedding cache (manual + LangGraph)
  |-- llamaindex_storage/ ........... Auto-created: persisted LlamaIndex index
  |-- .env .......................... Your OpenAI API key (not committed to git)
  |-- .gitignore
  |-- README.txt


Technologies
------------

  - Python 3.10+
  - OpenAI API (gpt-4.1-mini, text-embedding-3-small)
  - LangGraph (StateGraph, conditional edges, stateful orchestration)
  - LlamaIndex (VectorStoreIndex, SimpleDirectoryReader, PromptTemplate)
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
