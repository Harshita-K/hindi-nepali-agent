"""MANUAL collection: scrape + clean Nepali news/article pages ourselves.

Counts toward the >=20% manual-collection requirement because we are
fetching and extracting these pages ourselves (not downloading a pre-built
corpus). Fill in article URLs in urls_manual.txt (one per line) before running.

Run from repo root:
    python nepali/data/scraping/scrape_news.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from common.scraping_utils import crawl_urls, load_url_list  # noqa: E402

LANG_DIR = Path(__file__).resolve().parents[2]  # nepali/
URLS_FILE = Path(__file__).resolve().parent / "urls_manual.txt"

if __name__ == "__main__":
    if not URLS_FILE.exists():
        raise SystemExit(f"Add article URLs to {URLS_FILE} first (one per line).")
    urls = load_url_list(URLS_FILE)
    out_path = LANG_DIR / "data" / "raw" / "manual_scraped.jsonl"
    n = crawl_urls(urls, out_path, source_name="nepali_news_manual", delay_seconds=2.0)
    print(f"Wrote {n} manually scraped Nepali documents to {out_path}")
