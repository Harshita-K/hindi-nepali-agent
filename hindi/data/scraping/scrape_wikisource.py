"""MANUAL collection: Hindi Wikisource (hi.wikisource.org) -- community-
proofread public-domain books, NOT scanned images, so no OCR involved.
robots.txt is permissive (standard MediaWiki, no bot-specific restrictions).

Run from repo root:
    python hindi/data/scraping/scrape_wikisource.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from common.wikisource_utils import collect_wikisource  # noqa: E402

LANG_DIR = Path(__file__).resolve().parents[2]  # hindi/

if __name__ == "__main__":
    n = collect_wikisource(
        wiki_code="hi",
        out_path=LANG_DIR / "data" / "raw" / "wikisource.jsonl",
        max_pages=20000,
    )
    print(f"Collected {n} Hindi Wikisource pages")
