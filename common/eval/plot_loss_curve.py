"""Plot train/val loss and val perplexity curves from a Trainer's
train_log.csv, for the Phase 2 report.

    python -m common.eval.plot_loss_curve \
        --log /content/drive/MyDrive/LMA/hindi/output/train_log.csv \
        --out report/phase2/figures/hindi_loss_curve.png \
        --title "Model H (Hindi)"
"""
import argparse
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd


def plot_loss_curve(log_csv: Union[str, Path], out_path: Union[str, Path], title: str) -> None:
    df = pd.read_csv(log_csv)
    val_df = df.dropna(subset=["val_loss"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(df["step"], df["train_loss"], label="train loss", alpha=0.7)
    ax1.plot(val_df["step"], val_df["val_loss"], label="val loss", marker="o", markersize=3)
    ax1.set_xlabel("training step")
    ax1.set_ylabel("cross-entropy loss")
    ax1.set_title(f"{title}: loss curve")
    ax1.legend()

    ax2.plot(val_df["step"], val_df["val_ppl"], label="val perplexity", color="tab:orange", marker="o", markersize=3)
    ax2.set_xlabel("training step")
    ax2.set_ylabel("perplexity")
    ax2.set_title(f"{title}: validation perplexity")
    ax2.legend()

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to train_log.csv")
    parser.add_argument("--out", required=True, help="Output image path")
    parser.add_argument("--title", required=True, help='e.g. "Model H (Hindi)"')
    args = parser.parse_args()
    plot_loss_curve(args.log, args.out, args.title)
    print(f"Saved loss curve to {args.out}")


if __name__ == "__main__":
    main()
