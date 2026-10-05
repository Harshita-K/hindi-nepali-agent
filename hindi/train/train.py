"""Pretrain Model H (Hindi) -- a from-scratch decoder-only Transformer LM --
on the Hindi monolingual corpus (common/model/transformer.py). Safe to
interrupt and re-run: automatically resumes from the latest checkpoint in
--out_dir.

Local:
    python hindi/train/prepare_data.py          # once
    python hindi/train/train.py

Colab (see README.md "Phase 2 (Colab)" for the full snippet):
    !python hindi/train/train.py --out_dir /content/drive/MyDrive/lma_ckpts/hindi
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from common.model.config import GPTConfig
from common.train.trainer import Trainer, TrainConfig

LANG_DIR = Path(__file__).resolve().parents[1]
MODEL_CFG_PATH = LANG_DIR / "configs" / "model_config.yaml"
TRAIN_CFG_PATH = LANG_DIR / "configs" / "train_config.yaml"
DEFAULT_DATA_DIR = LANG_DIR / "data" / "bin"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=None, help="Checkpoint output dir (e.g. a mounted Google Drive path in Colab)")
    parser.add_argument("--data_dir", default=None, help="Tokenized-bin data dir (default: hindi/data/bin)")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max_steps from train_config.yaml")
    args = parser.parse_args()

    model_cfg = GPTConfig.from_yaml(MODEL_CFG_PATH)
    train_cfg = TrainConfig.from_yaml(TRAIN_CFG_PATH)

    train_cfg.data_dir = args.data_dir or str(DEFAULT_DATA_DIR)
    if args.out_dir:
        train_cfg.out_dir = args.out_dir
    if args.max_steps:
        train_cfg.max_steps = args.max_steps

    trainer = Trainer(model_cfg, train_cfg)
    trainer.train()


if __name__ == "__main__":
    main()
