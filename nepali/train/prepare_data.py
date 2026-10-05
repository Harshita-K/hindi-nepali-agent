"""Tokenize the Nepali train/val splits (Phase 1 output) into flat uint16
token-id binaries for training. Run once before nepali/train/train.py (and
again if the splits or tokenizer change).

    python nepali/train/prepare_data.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from common.train.data import build_bin

LANG_DIR = Path(__file__).resolve().parents[1]
SPM_MODEL = LANG_DIR / "tokenizer" / "vocab" / "nepali_spm.model"
SPLITS_DIR = LANG_DIR / "data" / "splits"
BIN_DIR = LANG_DIR / "data" / "bin"


def main() -> None:
    for split in ("train", "val"):
        meta = build_bin(
            jsonl_path=SPLITS_DIR / f"{split}.jsonl",
            spm_model_path=SPM_MODEL,
            out_bin_path=BIN_DIR / f"{split}.bin",
            out_meta_path=BIN_DIR / f"{split}_meta.json",
        )
        print(split, meta)


if __name__ == "__main__":
    main()
