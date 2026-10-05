"""Compute perplexity and bits-per-byte for Model L (Nepali) on the held-out
test split, using a trained checkpoint.

Phase 2 (pretrained checkpoint):
    python nepali/eval/run_intrinsic.py --ckpt /content/drive/MyDrive/LMA/nepali/output/best.pt

Phase 3 (finetuned checkpoint -- same general-domain news test split, to
check whether reasoning finetuning degraded general language-modeling
ability; writes to report/phase3/ instead of report/phase2/ so it doesn't
touch the already-graded Phase 2 output):
    python nepali/eval/run_intrinsic.py \\
        --ckpt /content/drive/MyDrive/LMA/nepali/finetune_output/best.pt \\
        --report_path report/phase3/nepali_intrinsic_finetuned.json
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from common.eval.intrinsic import bits_per_byte, count_utf8_bytes, evaluate_split
from common.model.config import GPTConfig
from common.model.transformer import GPT
from common.train.checkpoint import load_checkpoint
from common.train.data import build_bin

LANG_DIR = Path(__file__).resolve().parents[1]
MODEL_CFG_PATH = LANG_DIR / "configs" / "model_config.yaml"
TEST_JSONL = LANG_DIR / "data" / "splits" / "test.jsonl"
SPM_MODEL = LANG_DIR / "tokenizer" / "vocab" / "nepali_spm.model"
BIN_DIR = LANG_DIR / "data" / "bin"
REPORT_PATH = REPO_ROOT / "report" / "phase2" / "nepali_intrinsic.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to a checkpoint .pt file (latest.pt or best.pt)")
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--report_path", default=str(REPORT_PATH), help="Where to save the result JSON (default: report/phase2/nepali_intrinsic.json)")
    args = parser.parse_args()

    report_path = Path(args.report_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_bin = BIN_DIR / "test.bin"
    if not test_bin.exists():
        print("test.bin not found -- tokenizing test.jsonl now...")
        build_bin(TEST_JSONL, SPM_MODEL, test_bin, BIN_DIR / "test_meta.json")

    model_cfg = GPTConfig.from_yaml(MODEL_CFG_PATH)
    model = GPT(model_cfg).to(device)
    load_checkpoint(args.ckpt, model, map_location=device)

    result = evaluate_split(model, test_bin, args.block_size, device, args.batch_size)
    total_bytes = count_utf8_bytes(TEST_JSONL)
    result["total_bytes"] = total_bytes
    result["bits_per_byte"] = bits_per_byte(result["total_nll_nats"], total_bytes)

    print(json.dumps(result, indent=2))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved to {report_path}")


if __name__ == "__main__":
    main()
