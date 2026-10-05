"""Finetune Model H (Hindi) on the synthetic comparative-reasoning dataset
(hindi/finetune/generate_reasoning_data.py), starting from its Phase 2
pretrained checkpoint. Keeps the Hindi tokenizer/vocabulary fixed -- only
the model weights are loaded from pretraining, nothing about the tokenizer
changes. Safe to interrupt and re-run: automatically resumes from the
latest finetune checkpoint in --out_dir, same as pretraining.

Local:
    python hindi/finetune/generate_reasoning_data.py   # once, if not already run
    python hindi/finetune/finetune.py --pretrained_ckpt hindi/model/best.pt

Colab (mirrors Phase 2's pattern -- see README.md):
    !python hindi/finetune/finetune.py \
        --pretrained_ckpt /content/drive/MyDrive/LMA/hindi/output/best.pt \
        --out_dir /content/drive/MyDrive/LMA/hindi/finetune_output
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import sentencepiece as spm

from common.model.config import GPTConfig
from common.finetune.trainer import FinetuneConfig, FinetuneTrainer

LANG_DIR = Path(__file__).resolve().parents[1]
MODEL_CFG_PATH = LANG_DIR / "configs" / "model_config.yaml"
FINETUNE_CFG_PATH = LANG_DIR / "configs" / "finetune_config.yaml"
SPM_PATH = LANG_DIR / "tokenizer" / "vocab" / "hindi_spm.model"
DEFAULT_DATA_DIR = LANG_DIR / "data" / "reasoning"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", required=True, help="Path to Phase 2's best.pt to finetune from")
    parser.add_argument("--out_dir", default=None, help="Finetune checkpoint output dir (e.g. a mounted Google Drive path in Colab)")
    parser.add_argument("--data_dir", default=None, help="Reasoning-data dir (default: hindi/data/reasoning)")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max_steps from finetune_config.yaml")
    args = parser.parse_args()

    model_cfg = GPTConfig.from_yaml(MODEL_CFG_PATH)
    ft_cfg = FinetuneConfig.from_yaml(FINETUNE_CFG_PATH)

    ft_cfg.pretrained_ckpt = args.pretrained_ckpt
    ft_cfg.data_dir = args.data_dir or str(DEFAULT_DATA_DIR)
    if args.out_dir:
        ft_cfg.out_dir = args.out_dir
    if args.max_steps:
        ft_cfg.max_steps = args.max_steps

    sp = spm.SentencePieceProcessor(model_file=str(SPM_PATH))
    assert sp.get_piece_size() == model_cfg.vocab_size, (
        f"tokenizer vocab size {sp.get_piece_size()} != model_config vocab_size {model_cfg.vocab_size}"
    )

    trainer = FinetuneTrainer(model_cfg, ft_cfg, sp)
    trainer.train()


if __name__ == "__main__":
    main()
