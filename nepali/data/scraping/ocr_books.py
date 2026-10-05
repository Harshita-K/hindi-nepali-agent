"""MANUAL collection: extract/OCR Nepali text from PDFs you place in
nepali/data/raw/manual_pdfs/ (public-domain books, archive.org scans, etc.).

Run from repo root:
    python nepali/data/scraping/ocr_books.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from common.pdf_ocr_utils import process_pdf_directory  # noqa: E402

LANG_DIR = Path(__file__).resolve().parents[2]  # nepali/

if __name__ == "__main__":
    pdf_dir = LANG_DIR / "data" / "raw" / "manual_pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    out_path = LANG_DIR / "data" / "raw" / "manual_books.jsonl"
    n = process_pdf_directory(pdf_dir, out_path, source_name="nepali_books_manual", tesseract_lang="nep")
    print(f"Extracted {n} Nepali documents from PDFs in {pdf_dir}")
