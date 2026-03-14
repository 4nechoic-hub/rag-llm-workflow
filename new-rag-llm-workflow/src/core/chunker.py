# -*- coding: utf-8 -*-
"""Text chunking — split extracted text into overlapping chunks."""

import pandas as pd


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
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


def create_document_chunks(
    pdf_data: dict,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> pd.DataFrame:
    """
    Convert extracted PDF pages into chunk records.
    Each chunk retains document and page metadata.
    Returns a DataFrame with columns: doc_name, page_number, chunk_id, text.
    """
    records = []
    for doc_name, pages in pdf_data.items():
        for page in pages:
            page_number = page["page_number"]
            page_chunks = chunk_text(page["text"], chunk_size=chunk_size, overlap=overlap)
            for i, chunk in enumerate(page_chunks):
                records.append({
                    "doc_name": doc_name,
                    "page_number": page_number,
                    "chunk_id": f"{doc_name}_p{page_number}_c{i+1}",
                    "text": chunk,
                })

    df = pd.DataFrame(records)
    print(f"  Created {len(df)} chunks from {len(pdf_data)} document(s).")
    return df
