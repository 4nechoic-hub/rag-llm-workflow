# -*- coding: utf-8 -*-
"""
RAG-Based LLM Workflow for Research and Technical Document Analysis
Optimised for Spyder IDE (cell-by-cell execution with # %%)

What this script does:
1. Load PDFs from a folder
2. Extract text and chunk it
3. Create embeddings (cached locally)
4. Retrieve top-k relevant chunks for a user query
5. Generate a grounded answer using retrieved context
6. Support structured extraction and document comparison

Author: Tingyi Zhang
"""

# %% ========== IMPORTS ==========
import os
import sys
import json
import pickle
from pathlib import Path

# --- Dependency check ---
_missing = []
for _pkg, _import in [
    ("PyMuPDF", "fitz"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("python-dotenv", "dotenv"),
    ("scikit-learn", "sklearn"),
    ("openai", "openai"),
]:
    try:
        __import__(_import)
    except ImportError:
        _missing.append(_pkg)

if _missing:
    print("=" * 60)
    print("Missing packages. Install them with:")
    print(f"  pip install {' '.join(_missing)}")
    print("=" * 60)
    sys.exit(1)

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI


# %% ========== CONFIGURATION ==========
# Adjust these to match your project layout.
# In Spyder, set the working directory to your project root
# (the folder containing `pdfs/` and `.env`).

PDF_FOLDER = "pdfs"                  # folder containing your PDF files
CACHE_FOLDER = "cache"               # where processed chunk/embedding data are saved
CHUNK_SIZE = 1200                    # characters per chunk
CHUNK_OVERLAP = 200                  # overlap between chunks
TOP_K = 5                            # number of chunks retrieved per question

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.0

# --- Load API key from .env ---
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found.\n"
        "Create a file named '.env' in your project root with:\n"
        "OPENAI_API_KEY=sk-your_key_here"
    )

client = OpenAI(api_key=api_key)
os.makedirs(CACHE_FOLDER, exist_ok=True)

print(f"[OK] API key loaded.  Model: {CHAT_MODEL}")
print(f"[OK] PDF folder  : {Path(PDF_FOLDER).resolve()}")
print(f"[OK] Cache folder: {Path(CACHE_FOLDER).resolve()}")


# %% ========== PDF INGESTION ==========

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF using PyMuPDF. Returns list of page dicts."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text and text.strip():
            pages.append({
                "page_number": page_num + 1,
                "text": text.strip()
            })
    doc.close()
    return pages


def load_all_pdfs(pdf_folder=PDF_FOLDER):
    """
    Load all PDFs from a folder.
    Returns dict: {"filename.pdf": [{"page_number": 1, "text": "..."}, ...]}
    """
    pdf_folder = Path(pdf_folder)
    if not pdf_folder.exists():
        raise FileNotFoundError(
            f"PDF folder not found: {pdf_folder.resolve()}\n"
            "Check that your Spyder working directory is set correctly."
        )

    pdf_files = sorted(pdf_folder.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {pdf_folder.resolve()}")

    pdf_data = {}
    for pdf_file in pdf_files:
        print(f"  Loading: {pdf_file.name}")
        pdf_data[pdf_file.name] = extract_text_from_pdf(pdf_file)

    print(f"  Loaded {len(pdf_data)} PDF(s), "
          f"{sum(len(p) for p in pdf_data.values())} page(s) total.")
    return pdf_data


# %% ========== CHUNKING ==========

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping character-based chunks."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_length:
            break
        start += chunk_size - overlap

    return chunks


def create_document_chunks(pdf_data, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Convert extracted PDF pages into chunk records.
    Each chunk keeps document and page metadata.
    Returns a DataFrame.
    """
    records = []
    for doc_name, pages in pdf_data.items():
        for page in pages:
            page_number = page["page_number"]
            page_chunks = chunk_text(
                page["text"], chunk_size=chunk_size, overlap=overlap
            )
            for i, chunk in enumerate(page_chunks):
                records.append({
                    "doc_name": doc_name,
                    "page_number": page_number,
                    "chunk_id": f"{doc_name}_p{page_number}_c{i+1}",
                    "text": chunk
                })

    df = pd.DataFrame(records)
    print(f"  Created {len(df)} chunks from {len(pdf_data)} document(s).")
    return df


# %% ========== EMBEDDINGS ==========

def get_embedding(text, model=EMBEDDING_MODEL):
    """Get a single embedding vector from OpenAI."""
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding


def embed_chunks(df_chunks, force_recompute=False):
    """
    Create embeddings for all chunks.
    Caches to disk so you only pay for embeddings once.
    """
    cache_path = Path(CACHE_FOLDER) / "chunk_embeddings.pkl"

    if cache_path.exists() and not force_recompute:
        print("  Loading cached embeddings...")
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        print(f"  Loaded {len(payload['embeddings'])} cached embeddings.")
        return payload["df_chunks"], payload["embeddings"]

    print(f"  Creating embeddings for {len(df_chunks)} chunks (this may take a moment)...")
    embeddings = []
    for idx, row in df_chunks.iterrows():
        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"    Embedding chunk {idx+1}/{len(df_chunks)}")
        emb = get_embedding(row["text"])
        embeddings.append(emb)

    embeddings = np.array(embeddings, dtype=np.float32)

    with open(cache_path, "wb") as f:
        pickle.dump({"df_chunks": df_chunks, "embeddings": embeddings}, f)
    print(f"  Embeddings cached to {cache_path}")

    return df_chunks, embeddings


# %% ========== RETRIEVAL ==========

def retrieve_top_k(query, df_chunks, embeddings, top_k=TOP_K):
    """Retrieve top-k most relevant chunks using cosine similarity."""
    query_emb = np.array(get_embedding(query), dtype=np.float32).reshape(1, -1)
    sims = cosine_similarity(query_emb, embeddings)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]

    results = df_chunks.iloc[top_indices].copy()
    results["similarity"] = sims[top_indices]
    return results.reset_index(drop=True)


def format_context(retrieved_df):
    """Format retrieved chunks as prompt context string."""
    blocks = []
    for _, row in retrieved_df.iterrows():
        block = (
            f"[Source: {row['doc_name']} | Page: {row['page_number']} "
            f"| Chunk: {row['chunk_id']}]\n{row['text']}"
        )
        blocks.append(block)
    return "\n\n" + ("\n" + "-" * 80 + "\n\n").join(blocks)


# %% ========== LLM TASK FUNCTIONS ==========

def answer_question(query, df_chunks, embeddings, top_k=TOP_K):
    """
    Answer a question using retrieved context only.
    Returns (answer_text, retrieved_df).
    """
    retrieved = retrieve_top_k(query, df_chunks, embeddings, top_k=top_k)
    context = format_context(retrieved)

    system_prompt = (
        "You are a technical document assistant.\n\n"
        "Rules:\n"
        "1. Answer using ONLY the provided context.\n"
        "2. Do not use outside knowledge.\n"
        "3. If the answer is not clearly supported by the retrieved context, say: "
        "\"Insufficient evidence in retrieved documents.\"\n"
        "4. Be concise but specific.\n"
        "5. At the end, provide a short source list."
    )

    user_prompt = (
        f"Question:\n{query}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Return:\n1. Answer\n2. Key supporting points\n3. Source list"
    )

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    answer = response.choices[0].message.content
    return answer, retrieved


def extract_structured_summary(query, df_chunks, embeddings, top_k=6):
    """
    Extract a structured JSON-style summary from retrieved chunks.
    Returns (json_string, retrieved_df).
    """
    retrieved = retrieve_top_k(query, df_chunks, embeddings, top_k=top_k)
    context = format_context(retrieved)

    system_prompt = (
        "You are a technical document extraction assistant.\n\n"
        "Rules:\n"
        "1. Use ONLY the provided context.\n"
        "2. If a field is not supported, return \"Not found in retrieved context\".\n"
        "3. Return valid JSON only."
    )

    user_prompt = (
        "Task:\nExtract a structured summary from the retrieved technical document context.\n\n"
        "Requested fields:\n"
        "- title\n- objective\n- methodology\n- experimental_setup\n"
        "- main_findings\n- limitations\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Return JSON only."
    )

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    content = response.choices[0].message.content
    return content, retrieved


def compare_documents(query, df_chunks, embeddings, top_k=8):
    """
    Compare documents using retrieved chunks.
    Returns (comparison_text, retrieved_df).
    """
    retrieved = retrieve_top_k(query, df_chunks, embeddings, top_k=top_k)
    context = format_context(retrieved)

    system_prompt = (
        "You are a technical comparison assistant.\n\n"
        "Rules:\n"
        "1. Use ONLY the provided context.\n"
        "2. Do not invent details.\n"
        "3. If comparison points are unsupported, say so clearly.\n"
        "4. Return a structured comparison."
    )

    user_prompt = (
        f"Task:\nCompare the relevant documents based on the user's request.\n\n"
        f"User request:\n{query}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Return:\n"
        "1. Documents involved\n2. Similarities\n3. Differences\n"
        "4. Key methodological differences\n5. Key findings differences\n"
        "6. Source list"
    )

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    content = response.choices[0].message.content
    return content, retrieved


# %% ========== UTILITY: SAVE RESULTS ==========

def save_text_output(text, filename):
    """Save output text to a file in the cache folder."""
    output_path = Path(CACHE_FOLDER) / filename
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved output to: {output_path}")


def save_retrieval_output(retrieved_df, filename):
    """Save retrieved chunks to CSV in the cache folder."""
    output_path = Path(CACHE_FOLDER) / filename
    retrieved_df.to_csv(output_path, index=False)
    print(f"Saved retrieval table to: {output_path}")


# %% ========== DISPLAY HELPERS (for Spyder console) ==========

def print_answer(answer, retrieved):
    """Pretty-print an answer and its sources in the Spyder console."""
    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(answer)
    print("\n" + "-" * 60)
    print("RETRIEVED SOURCES")
    print("-" * 60)
    print(retrieved[["doc_name", "page_number", "chunk_id", "similarity"]].to_string(index=False))
    print()


def print_structured(result, retrieved):
    """Pretty-print a structured extraction result."""
    print("\n" + "=" * 60)
    print("STRUCTURED OUTPUT")
    print("=" * 60)
    # Try to pretty-print as JSON, fall back to raw text
    try:
        parsed = json.loads(result)
        print(json.dumps(parsed, indent=2))
    except (json.JSONDecodeError, TypeError):
        print(result)
    print("\n" + "-" * 60)
    print("RETRIEVED SOURCES")
    print("-" * 60)
    print(retrieved[["doc_name", "page_number", "chunk_id", "similarity"]].to_string(index=False))
    print()


# %% ========== BUILD THE RAG SYSTEM ==========
# Run this cell first. It loads PDFs, creates chunks, and embeds them.
# After running once, embeddings are cached — subsequent runs are fast.

print("=" * 60)
print("BUILDING RAG SYSTEM")
print("=" * 60)

print("\nStep 1: Loading PDFs...")
pdf_data = load_all_pdfs(PDF_FOLDER)

print("\nStep 2: Creating chunks...")
df_chunks = create_document_chunks(pdf_data)

print("\nStep 3: Embedding chunks...")
df_chunks, embeddings = embed_chunks(df_chunks, force_recompute=False)

print("\n" + "=" * 60)
print("RAG SYSTEM READY")
print(f"  Documents : {df_chunks['doc_name'].nunique()}")
print(f"  Chunks    : {len(df_chunks)}")
print(f"  Embedding dim: {embeddings.shape[1]}")
print("=" * 60)


# %% ========== QUERY: Ask a Question ==========
# Edit the query below and run this cell.

query = "What experimental setup was used in the study?"

answer, retrieved = answer_question(query, df_chunks, embeddings, top_k=TOP_K)
print_answer(answer, retrieved)

# Optional: save results
# save_text_output(answer, "answer_output.txt")
# save_retrieval_output(retrieved, "answer_sources.csv")


# %% ========== QUERY: Structured Extraction ==========
# Edit the query below and run this cell.

query = "Summarise the methodology and main findings of this paper."

result, retrieved = extract_structured_summary(query, df_chunks, embeddings, top_k=6)
print_structured(result, retrieved)


# %% ========== QUERY: Compare Documents ==========
# Edit the query below and run this cell.

query = "Compare the experimental approaches across the loaded documents."

result, retrieved = compare_documents(query, df_chunks, embeddings, top_k=8)
print_answer(result, retrieved)


# %% ========== INTERACTIVE MODE (optional) ==========
# Uncomment the lines below if you prefer the terminal-style menu.
# Note: input() works in Spyder's IPython console but can be less
# convenient than the cell-by-cell approach above.

# def interactive_loop(df_chunks, embeddings):
#     """Simple terminal loop for Spyder console use."""
#     print("\nRAG system ready.")
#     print("Options: 1=Question  2=Extraction  3=Compare  4=Exit\n")
#
#     while True:
#         mode = input("Enter option (1/2/3/4): ").strip()
#
#         if mode == "1":
#             q = input("Enter your question:\n> ").strip()
#             ans, ret = answer_question(q, df_chunks, embeddings, top_k=TOP_K)
#             print_answer(ans, ret)
#
#         elif mode == "2":
#             q = input("Enter extraction request:\n> ").strip()
#             res, ret = extract_structured_summary(q, df_chunks, embeddings, top_k=6)
#             print_structured(res, ret)
#
#         elif mode == "3":
#             q = input("Enter comparison request:\n> ").strip()
#             res, ret = compare_documents(q, df_chunks, embeddings, top_k=8)
#             print_answer(res, ret)
#
#         elif mode == "4":
#             print("Exiting.")
#             break
#
#         else:
#             print("Invalid option. Enter 1, 2, 3, or 4.")
#
# interactive_loop(df_chunks, embeddings)