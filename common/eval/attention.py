"""Attention analysis for the hand-built multi-head causal self-attention:
extract per-layer/head attention maps for example sentences, summarize them
with entropy and mean attention distance, and plot query-vs-key heatmaps.
"""
import glob
import warnings
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch

from common.model.transformer import GPT

# matplotlib's default font (DejaVu Sans) has no Devanagari glyphs, so
# Hindi/Nepali tick labels render as empty boxes ("tofu") unless a font
# that actually covers the script is explicitly requested.
_DEVANAGARI_FONT_CANDIDATES = [
    "Noto Sans Devanagari",
    "Lohit Devanagari",
    "Nirmala UI",
    "Mangal",
    "Kohinoor Devanagari",
]

# Filesystem fallback if the font isn't in matplotlib's *cached* font list --
# matplotlib caches its font list to disk independently of fontconfig, so a
# font installed via apt after that cache was built (the common case: the
# cache already existed from an earlier matplotlib import in the same long-
# running Colab session) won't be picked up by name lookup alone, even after
# `fc-cache` and even across a kernel restart, since the cache file persists
# on the VM's disk. Scanning for the file directly and registering it with
# `addfont()` sidesteps the cache entirely.
_DEVANAGARI_FONT_FILE_GLOBS = [
    "/usr/share/fonts/**/*Devanagari*.[ot]tf",
    "/usr/share/fonts/**/*Lohit*Deva*.[ot]tf",
    "/usr/local/share/fonts/**/*Devanagari*.[ot]tf",
    str(Path.home() / ".fonts/**/*Devanagari*.[ot]tf"),
    str(Path.home() / ".local/share/fonts/**/*Devanagari*.[ot]tf"),
]


def _find_devanagari_font() -> Optional[fm.FontProperties]:
    available = {f.name for f in fm.fontManager.ttflist}
    for name in _DEVANAGARI_FONT_CANDIDATES:
        if name in available:
            return fm.FontProperties(family=name)

    for pattern in _DEVANAGARI_FONT_FILE_GLOBS:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            fm.fontManager.addfont(matches[0])
            return fm.FontProperties(fname=matches[0])

    warnings.warn(
        "No Devanagari-capable font found (checked matplotlib's cached font "
        "list and common Linux font directories directly) -- attention "
        "heatmap tick labels will render as empty boxes. On Colab/Ubuntu, "
        "install one first: `!apt-get update -qq && apt-get install -y "
        "fonts-noto-core` (falls back to `fonts-lohit-deva` if that "
        "package name doesn't exist on this image)."
    )
    return None


@torch.no_grad()
def get_attention_maps(model: GPT, token_ids: List[int], device: str = "cpu") -> List[np.ndarray]:
    """Run the model on one token sequence and return its per-layer
    attention weights, each of shape (n_heads, T, T) (batch dim squeezed)."""
    model.eval()
    idx = torch.tensor([token_ids], device=device)
    _, _, attn_maps = model(idx, return_attn=True)
    return [a[0].cpu().numpy() for a in attn_maps]


def attention_entropy(attn_maps: List[np.ndarray]) -> np.ndarray:
    """Mean entropy (over query positions, in nats) per (layer, head):
    higher = more diffuse attention, lower = more peaked/focused."""
    n_layers = len(attn_maps)
    n_heads = attn_maps[0].shape[0]
    entropy = np.zeros((n_layers, n_heads))
    for l, attn in enumerate(attn_maps):  # attn: (h, T, T)
        p = np.clip(attn, 1e-12, 1.0)
        ent_per_query = -(p * np.log(p)).sum(axis=-1)  # (h, T)
        entropy[l] = ent_per_query.mean(axis=-1)
    return entropy  # (n_layers, n_heads)


def mean_attention_distance(attn_maps: List[np.ndarray]) -> np.ndarray:
    """Mean |query_pos - key_pos| weighted by attention, per (layer, head):
    large -> long-range/content-based head, small -> local/positional head."""
    n_layers = len(attn_maps)
    n_heads = attn_maps[0].shape[0]
    dist = np.zeros((n_layers, n_heads))
    for l, attn in enumerate(attn_maps):  # (h, T, T)
        T = attn.shape[-1]
        positions = np.arange(T)
        distance_matrix = np.abs(positions[:, None] - positions[None, :])  # (T, T)
        weighted = attn * distance_matrix[None, :, :]  # (h, T, T)
        dist[l] = weighted.sum(axis=-1).mean(axis=-1)
    return dist  # (n_layers, n_heads)


def plot_attention_heatmap(
    attn_maps: List[np.ndarray],
    layer_idx: int,
    head_indices: List[int],
    tokens: List[str],
    title_prefix: str,
    save_path: Union[str, Path],
) -> None:
    """Grid of query-vs-key attention heatmaps for one layer, one panel per
    head in `head_indices`. Each panel has a title and axis labels; a
    shared colorbar acts as the legend for the attention-weight color scale.
    """
    n = len(head_indices)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    if n == 1:
        axes = [axes]

    devanagari_font = _find_devanagari_font()
    # SentencePiece's "▁" (lower one-eighth block) marks a word start
    # but isn't a Devanagari glyph, so no Devanagari font covers it either
    # -- it renders as a tofu box regardless of the font fix above. Swap it
    # for a plain space, which is what it represents visually anyway.
    display_tokens = [t.replace("▁", " ") for t in tokens]

    im = None
    for ax, h in zip(axes, head_indices):
        attn = attn_maps[layer_idx][h]  # (T, T)
        im = ax.imshow(attn, cmap="viridis", vmin=0, vmax=attn.max())
        ax.set_title(f"{title_prefix} -- layer {layer_idx}, head {h}")
        ax.set_xlabel("key position")
        ax.set_ylabel("query position")
        if len(tokens) <= 20:
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(display_tokens, rotation=90, fontsize=7, fontproperties=devanagari_font)
            ax.set_yticks(range(len(tokens)))
            ax.set_yticklabels(display_tokens, fontsize=7, fontproperties=devanagari_font)

    fig.colorbar(im, ax=axes, label="attention weight", fraction=0.02, pad=0.02)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
