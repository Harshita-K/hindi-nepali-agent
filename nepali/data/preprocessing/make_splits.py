"""Split data/processed/corpus.jsonl into train/val/test (doc-level, so no
document leaks across splits), and write corpus statistics required for the
Phase 1 report (size, sources, manual vs downloaded token split).

Writes:
  nepali/data/splits/{train,val,test}.jsonl
  report/phase1/nepali_corpus_stats.json

Run from repo root:
    python nepali/data/preprocessing/make_splits.py
"""
import json
import random
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from common.corpus_stats import compute_corpus_stats, load_jsonl, save_stats  # noqa: E402

LANG_DIR = Path(__file__).resolve().parents[2]  # nepali/
CORPUS_PATH = LANG_DIR / "data" / "processed" / "corpus.jsonl"
SPLITS_DIR = LANG_DIR / "data" / "splits"
CONFIG_PATH = LANG_DIR / "configs" / "data_config.yaml"
REPORT_PATH = REPO_ROOT / "report" / "phase1" / "nepali_corpus_stats.json"


def write_split(docs: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    random.seed(config["random_seed"])

    docs = load_jsonl(CORPUS_PATH)
    random.shuffle(docs)

    n = len(docs)
    ratios = config["split_ratios"]
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])

    train_docs = docs[:n_train]
    val_docs = docs[n_train:n_train + n_val]
    test_docs = docs[n_train + n_val:]

    write_split(train_docs, SPLITS_DIR / "train.jsonl")
    write_split(val_docs, SPLITS_DIR / "val.jsonl")
    write_split(test_docs, SPLITS_DIR / "test.jsonl")
    print(f"train={len(train_docs)} val={len(val_docs)} test={len(test_docs)}")

    stats = {
        "language": config["language"],
        "splits": {
            "train": compute_corpus_stats(train_docs),
            "val": compute_corpus_stats(val_docs),
            "test": compute_corpus_stats(test_docs),
        },
        "overall": compute_corpus_stats(docs),
    }
    save_stats(stats, REPORT_PATH)
    print(f"Stats written to {REPORT_PATH}")
    print(f"Manual token fraction (overall): {stats['overall']['manual_token_fraction']:.2%}")
    if stats["overall"]["manual_token_fraction"] < config["min_manual_fraction"]:
        print("WARNING: manual fraction is below the required "
              f"{config['min_manual_fraction']:.0%} -- collect more manual data.")
    target = config["target_total_tokens"]
    if stats["overall"]["total_whitespace_tokens"] < target:
        print(f"NOTE: total tokens ({stats['overall']['total_whitespace_tokens']:,}) is below the "
              f"~500M target ({target:,}). This is expected for Nepali -- report the exact count "
              "and justify the shortfall in the Phase 1 report.")


if __name__ == "__main__":
    main()
