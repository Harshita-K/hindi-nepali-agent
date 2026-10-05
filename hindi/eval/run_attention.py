"""Attention analysis for Model H (Hindi): heatmaps for an early and a late
layer across several heads, plus per-(layer, head) entropy and mean
attention-distance summaries, on example sentences drawn from the test set.

Phase 2 (pretrained checkpoint, news test-set sentences):
    python hindi/eval/run_attention.py --ckpt /content/drive/MyDrive/LMA/hindi/output/best.pt

Phase 3 (finetuned checkpoint, comparative-reasoning prompts -- writes to
report/phase3/ instead of report/phase2/ so it doesn't touch the
already-graded Phase 2 outputs):
    python hindi/eval/run_attention.py \\
        --ckpt /content/drive/MyDrive/LMA/hindi/finetune_output/best.pt \\
        --text_jsonl hindi/data/reasoning/test.jsonl --text_field prompt \\
        --fig_dir report/phase3/figures --report_path report/phase3/hindi_attention_finetuned.json \\
        --prefix hindi_finetuned --label "Model H (Hindi, finetuned)"
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import sentencepiece as spm
import torch

from common.eval.attention import (
    attention_entropy,
    get_attention_maps,
    mean_attention_distance,
    plot_attention_heatmap,
)
from common.model.config import GPTConfig
from common.model.transformer import GPT
from common.train.checkpoint import load_checkpoint

LANG_DIR = Path(__file__).resolve().parents[1]
MODEL_CFG_PATH = LANG_DIR / "configs" / "model_config.yaml"
SPM_MODEL = LANG_DIR / "tokenizer" / "vocab" / "hindi_spm.model"
TEST_JSONL = LANG_DIR / "data" / "splits" / "test.jsonl"
FIG_DIR = REPO_ROOT / "report" / "phase2" / "figures"
REPORT_PATH = REPO_ROOT / "report" / "phase2" / "hindi_attention.json"

N_EXAMPLE_SENTENCES = 3  # how many sentences get plotted as heatmaps
MAX_TOKENS_FOR_STATS = 128  # window entropy/mean-distance are computed over
MAX_TOKENS_FOR_HEATMAP = 16  # separate, smaller window -- just for plot readability
HEADS_TO_PLOT = [0, 1, 2, 3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--n_sentences", type=int, default=200, help="How many test-set sentences to average entropy/distance over")
    parser.add_argument("--text_jsonl", default=str(TEST_JSONL), help="JSONL file to draw example sentences from (default: Phase 2's news test split)")
    parser.add_argument("--text_field", default="text", help="JSON key holding the text to analyze (reasoning data uses 'prompt')")
    parser.add_argument("--fig_dir", default=str(FIG_DIR), help="Where to save heatmap PNGs (default: report/phase2/figures)")
    parser.add_argument("--report_path", default=str(REPORT_PATH), help="Where to save the entropy/distance JSON summary (default: report/phase2/hindi_attention.json)")
    parser.add_argument("--prefix", default="hindi", help="Filename prefix for saved heatmaps")
    parser.add_argument("--label", default="Model H (Hindi)", help="Plot title label")
    args = parser.parse_args()

    text_jsonl = Path(args.text_jsonl)
    fig_dir = Path(args.fig_dir)
    report_path = Path(args.report_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp = spm.SentencePieceProcessor(model_file=str(SPM_MODEL))

    model_cfg = GPTConfig.from_yaml(MODEL_CFG_PATH)
    model = GPT(model_cfg).to(device)
    load_checkpoint(args.ckpt, model, map_location=device)

    docs = [
        json.loads(l)
        for l in text_jsonl.read_text(encoding="utf-8").splitlines()[: args.n_sentences]
        if l.strip()
    ]

    entropy_accum, distance_accum = [], []
    example_sentences_used = []
    n_used = 0

    for doc in docs:
        # Entropy/mean-distance are computed over the full MAX_TOKENS_FOR_STATS
        # window -- truncating to the small heatmap window here would cap mean
        # attention distance at that window size and make every head look
        # artificially "local", masking genuine long-range heads.
        text = doc[args.text_field]
        ids = sp.encode(text, out_type=int)[:MAX_TOKENS_FOR_STATS]
        if len(ids) < 6:
            continue

        attn_maps = get_attention_maps(model, ids, device)  # list of (h, T, T)
        entropy_accum.append(attention_entropy(attn_maps))
        distance_accum.append(mean_attention_distance(attn_maps))

        if n_used < N_EXAMPLE_SENTENCES:
            tokens_str = sp.encode(text, out_type=str)[:MAX_TOKENS_FOR_STATS][:MAX_TOKENS_FOR_HEATMAP]
            # Slice the already-computed attention down to the heatmap window
            # only for plotting -- the stats above still used the full window.
            plot_maps = [a[:, :MAX_TOKENS_FOR_HEATMAP, :MAX_TOKENS_FOR_HEATMAP] for a in attn_maps]
            for layer_idx, layer_name in [(0, "early"), (model_cfg.n_layers - 1, "late")]:
                save_path = fig_dir / f"{args.prefix}_attn_ex{n_used}_{layer_name}layer{layer_idx}.png"
                plot_attention_heatmap(plot_maps, layer_idx, HEADS_TO_PLOT, tokens_str, args.label, save_path)
            example_sentences_used.append(text[:120])

        n_used += 1

    mean_entropy = np.mean(entropy_accum, axis=0)  # (n_layers, n_heads)
    mean_distance = np.mean(distance_accum, axis=0)

    result = {
        "n_sentences_analyzed": n_used,
        "example_sentences_plotted": example_sentences_used,
        "attention_entropy_by_layer_head": mean_entropy.tolist(),
        "mean_attention_distance_by_layer_head": mean_distance.tolist(),
    }
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "example_sentences_plotted"}, indent=2))
    print(f"Saved heatmaps to {fig_dir}, summary to {report_path}")


if __name__ == "__main__":
    main()
