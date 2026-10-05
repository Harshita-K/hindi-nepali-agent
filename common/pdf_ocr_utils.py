"""PDF text extraction + OCR utilities for MANUAL data collection
(books, scanned documents, PDFs you gather yourself).

Two paths:
  - digital PDFs (text already selectable): extract directly, much faster/cleaner.
  - scanned PDFs (images of pages): rasterize pages and run Tesseract OCR.

Requires system binaries: `tesseract` (with hin/nep language packs) and
`poppler` (for pdf2image). On macOS: `brew install tesseract tesseract-lang poppler`.
"""
import json
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path

MIN_CHARS_PER_PAGE_TO_SKIP_OCR = 40  # below this, assume the page is scanned (no text layer)
MAX_OCR_PAGES = 300  # cap per book: bounds memory/time on huge scans; plenty for corpus purposes


def extract_digital_pdf(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)


def ocr_pdf(pdf_path: Path, tesseract_lang: str, dpi: int = 300) -> str:
    """OCR page-by-page (NOT convert_from_path(...) for the whole doc at
    once -- that rasterizes every page into memory simultaneously, which
    OOM-kills the process on large scanned books, e.g. hundreds of pages at
    300dpi). Capped at MAX_OCR_PAGES for the same reason -- bounding one
    huge book's cost matters more than completeness when the goal is
    aggregate corpus volume across many books.
    """
    n_pages = len(fitz.open(pdf_path))
    texts = []
    for page_num in range(1, min(n_pages, MAX_OCR_PAGES) + 1):
        images = convert_from_path(pdf_path, dpi=dpi, first_page=page_num, last_page=page_num)
        if images:
            texts.append(pytesseract.image_to_string(images[0], lang=tesseract_lang))
    return "\n".join(texts)


def extract_pdf_auto(pdf_path: Path, tesseract_lang: str) -> tuple[str, str]:
    """Try direct extraction first; fall back to OCR per-page if the text
    layer looks empty/too sparse. Returns (text, method) where method is
    'digital' or 'ocr'.
    """
    digital_text = extract_digital_pdf(pdf_path)
    doc = fitz.open(pdf_path)
    avg_chars_per_page = len(digital_text) / max(len(doc), 1)
    if avg_chars_per_page >= MIN_CHARS_PER_PAGE_TO_SKIP_OCR:
        return digital_text, "digital"
    return ocr_pdf(pdf_path, tesseract_lang), "ocr"


def process_pdf_directory(pdf_dir: Path, out_path: Path, source_name: str, tesseract_lang: str) -> int:
    """Process every .pdf in pdf_dir, appending JSONL records (source_type=manual)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out_path.open("a", encoding="utf-8") as f:
        for pdf_path in sorted(pdf_dir.glob("*.pdf")):
            print(f"Processing {pdf_path.name} ...")
            text, method = extract_pdf_auto(pdf_path, tesseract_lang)
            if not text or len(text.strip()) < 200:
                print(f"  -> skipped (too little text extracted)")
                continue
            record = {
                "text": text,
                "source": source_name,
                "source_type": "manual",
                "file": pdf_path.name,
                "extraction_method": method,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1
            print(f"  -> {method}, {len(text)} chars")
    return n_written
