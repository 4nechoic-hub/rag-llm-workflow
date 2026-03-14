# -*- coding: utf-8 -*-
"""PDF ingestion — extract text from PDF files using PyMuPDF."""

from pathlib import Path
import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text from a PDF. Returns list of page dicts."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text")
        if text and text.strip():
            pages.append({
                "page_number": page_num + 1,
                "text": text.strip(),
            })
    doc.close()
    return pages


def load_all_pdfs(pdf_folder: str) -> dict:
    """
    Load all PDFs from a folder.
    Returns dict: {"filename.pdf": [{"page_number": 1, "text": "..."}, ...]}
    """
    folder = Path(pdf_folder)
    if not folder.exists():
        raise FileNotFoundError(f"PDF folder not found: {folder.resolve()}")

    pdf_files = sorted(folder.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {folder.resolve()}")

    pdf_data = {}
    for pdf_file in pdf_files:
        print(f"  Loading: {pdf_file.name}")
        pdf_data[pdf_file.name] = extract_text_from_pdf(pdf_file)

    total_pages = sum(len(p) for p in pdf_data.values())
    print(f"  Loaded {len(pdf_data)} PDF(s), {total_pages} page(s) total.")
    return pdf_data
