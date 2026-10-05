"""MANUAL collection: extract/OCR Hindi text from PDFs you place in
hindi/data/raw/manual_pdfs/ (public-domain books, archive.org scans, etc.).

Run from repo root:
    python hindi/data/scraping/ocr_books.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from common.pdf_ocr_utils import process_pdf_directory  # noqa: E402

LANG_DIR = Path(__file__).resolve().parents[2]  # hindi/

if __name__ == "__main__":
    pdf_dir = LANG_DIR / "data" / "raw" / "manual_pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    out_path = LANG_DIR / "data" / "raw" / "manual_books.jsonl"
    n = process_pdf_directory(pdf_dir, out_path, source_name="hindi_books_manual", tesseract_lang="hin")
    print(f"Extracted {n} Hindi documents from PDFs in {pdf_dir}")
