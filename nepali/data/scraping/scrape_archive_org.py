"""MANUAL collection: Nepali books from archive.org (language:nep), preferring
archive.org's own pre-OCR'd _djvu.txt over running Tesseract ourselves.
robots.txt has no AI-bot-specific restrictions. This is the highest
token-density manual source available -- a few hundred books can outweigh
tens of thousands of news articles. Only ~1,600 items are tagged
language:nep on archive.org (vs. ~680k for hin), consistent with Nepali
being the lower-resource language -- expect this to run out well before
--max-books on a full pull.

Run from repo root:
    python nepali/data/scraping/scrape_archive_org.py --max-books 500
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from common.archive_org_utils import collect_books  # noqa: E402

LANG_DIR = Path(__file__).resolve().parents[2]  # nepali/

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-books", type=int, default=500)
    args = parser.parse_args()
    n = collect_books(
        language_code="nep",
        tesseract_lang="nep",
        out_path=LANG_DIR / "data" / "raw" / "archive_org_books.jsonl",
        tmp_pdf_dir=LANG_DIR / "data" / "raw" / "_archive_org_pdf_tmp",
        max_books=args.max_books,
    )
    print(f"Collected {n} Nepali books from archive.org")
