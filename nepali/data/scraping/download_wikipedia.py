"""Download + extract Nepali Wikipedia dump into nepali/data/raw/wikipedia.jsonl.
This is a DOWNLOADED source (not manual). Run from repo root in the background
because it is slow:
    python nepali/data/scraping/download_wikipedia.py > /tmp/nepali_wikipedia.log 2>&1 &
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from common.wikipedia_utils import build_wikipedia_corpus  # noqa: E402

LANG_DIR = Path(__file__).resolve().parents[2]  # nepali/

if __name__ == "__main__":
    n = build_wikipedia_corpus(
        wiki_code="ne",
        work_dir=LANG_DIR / "data" / "raw" / "_wikipedia_work",
        out_path=LANG_DIR / "data" / "raw" / "wikipedia.jsonl",
        source_name="ne_wikipedia",
    )
    print(f"Wrote {n} Nepali Wikipedia articles.")
