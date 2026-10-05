"""Download public Hindi text corpora via the HF `datasets` library
(data only -- no pretrained models/tokenizers, so this is allowed).
Counts as DOWNLOADED, not manual. Streams to avoid loading everything into RAM.

Run from repo root in the background because it is slow:
    python hindi/data/scraping/download_public_corpus.py --dataset oscar --limit 2000000 > /tmp/hindi_oscar.log 2>&1 &
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

LANG_DIR = Path(__file__).resolve().parents[2]  # hindi/

DATASET_SPECS = {
    # oscar-corpus/OSCAR-2301 is gated (requires manual HF approval, fails
    # headless) -- fineweb-2 is ungated, quality-filtered, per-language config.
    "fineweb2": dict(path="HuggingFaceFW/fineweb-2", name="hin_Deva", split="train", text_field="text"),
    "c4": dict(path="allenai/c4", name="hi", split="train", text_field="text"),
    # ai4bharat/IndicCorpV2 has a single config ('indiccorp_v2') but splits
    # BY LANGUAGE (hin_Deva, npi_Deva, ...) rather than the usual train/test
    # -- pass the language as `split`, not `name`.
    "indiccorp": dict(path="ai4bharat/IndicCorpV2", name="indiccorp_v2", split="hin_Deva", text_field="text"),
    # ai4bharat/sangraha: config is quality tier ('verified'/'unverified'/
    # 'synthetic'), language is the split -- 'verified' is curated/cleaned.
    "sangraha": dict(path="ai4bharat/sangraha", name="verified", split="hin", text_field="text"),
}


def main(dataset_key: str, limit: int) -> None:
    from datasets import load_dataset  # imported lazily; heavy dependency

    spec = DATASET_SPECS[dataset_key]
    out_path = LANG_DIR / "data" / "raw" / f"{dataset_key}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resumable: count existing lines and skip that many underlying dataset
    # rows so re-running with a higher --limit appends more instead of
    # re-downloading/overwriting everything from scratch.
    already = sum(1 for line in out_path.open(encoding="utf-8") if line.strip()) if out_path.exists() else 0
    ds = load_dataset(spec["path"], spec["name"], split=spec["split"], streaming=True)
    if already:
        ds = ds.skip(already)
        print(f"Resuming: {already} records already collected, skipping that many source rows")

    n = already
    with out_path.open("a", encoding="utf-8") as f:
        for row in ds:
            text = row.get(spec["text_field"], "").strip()
            if len(text) < 200:
                continue
            record = {"text": text, "source": dataset_key, "source_type": "downloaded"}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
            if n >= limit:
                break
    print(f"{n} total records in {out_path} ({n - already} added this run)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASET_SPECS), required=True)
    parser.add_argument("--limit", type=int, default=2_000_000, help="target total documents (existing + new)")
    args = parser.parse_args()
    main(args.dataset, args.limit)
