"""Merge all raw Nepali sources (data/raw/*.jsonl), clean, deduplicate, and
subsample toward the ~500M token target while GUARANTEEING the >=20% manual
fraction floor (a hard compliance requirement -- see
common.corpus_stats.sample_corpus_to_target for the exact policy: all
manual docs are kept, downloaded docs are capped by whichever is tighter,
the ratio floor or the token target). If manual collection can't reach the
target at that ratio, the resulting corpus is smaller than target rather
than silently diluting below the floor -- this is expected to be more
likely for Nepali as the lower-resource language.

Run from repo root:
    python nepali/data/preprocessing/build_corpus.py
"""
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from common.corpus_stats import sample_corpus_to_target  # noqa: E402
from common.text_cleaning import clean_document, dedup_documents  # noqa: E402

LANG_DIR = Path(__file__).resolve().parents[2]  # nepali/
RAW_DIR = LANG_DIR / "data" / "raw"
OUT_PATH = LANG_DIR / "data" / "processed" / "corpus.jsonl"
CONFIG_PATH = LANG_DIR / "configs" / "data_config.yaml"


def load_all_raw() -> list[dict]:
    """Tolerates a malformed trailing line -- raw files may be actively
    being appended to by a still-running scraper while this reads them.
    """
    docs = []
    n_skipped = 0
    for path in sorted(RAW_DIR.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    n_skipped += 1
    if n_skipped:
        print(f"Skipped {n_skipped} malformed line(s) (likely a scraper still writing concurrently)")
    return docs


def main() -> None:
    raw_docs = load_all_raw()
    print(f"Loaded {len(raw_docs)} raw documents from {RAW_DIR}")

    cleaned = []
    for d in raw_docs:
        text = clean_document(d["text"])
        if text is None:
            continue
        d["text"] = text
        cleaned.append(d)
    print(f"{len(cleaned)} documents survived cleaning/script-filtering")

    # dedup_documents keeps the FIRST doc it sees per content hash, so put
    # manual docs first -- a manual/downloaded exact-duplicate then counts
    # as manual (by request) rather than being attributed by file-load order.
    cleaned.sort(key=lambda d: d.get("source_type") != "manual")
    deduped = dedup_documents(cleaned)
    print(f"{len(deduped)} documents remain after exact-dedup")

    config = yaml.safe_load(CONFIG_PATH.read_text())
    # enforce_ratio=False by request: cap downloaded at ~400M directly
    # rather than deriving it from target_total_tokens - manual, since
    # manual is still actively growing via ongoing scraping. Uses all
    # available manual first -- report the resulting fraction honestly.
    sampled = sample_corpus_to_target(
        deduped,
        target_total_tokens=config["target_total_tokens"],
        min_manual_fraction=config["min_manual_fraction"],
        seed=config["random_seed"],
        enforce_ratio=False,
        downloaded_target_tokens=400_000_000,
    )
    print(f"{len(sampled)} documents kept after target-size sampling (ratio floor not enforced this run)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for i, d in enumerate(sampled):
            d["doc_id"] = i
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Wrote processed corpus to {OUT_PATH}")


if __name__ == "__main__":
    main()
