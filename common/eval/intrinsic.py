"""Intrinsic language-modeling metrics: cross-entropy, perplexity, and
bits-per-byte (BPB) on a held-out split.

Unlike training's random-window batch sampling, evaluation here makes one
deterministic pass over the whole split (non-overlapping blocks) so the
reported number is reproducible and covers every token exactly once.
"""
import json
import math
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from common.model.transformer import GPT


def count_utf8_bytes(jsonl_path: Union[str, Path]) -> int:
    """Total UTF-8 byte length of the "text" field across a JSONL split --
    the denominator for bits-per-byte, which normalizes by raw bytes rather
    than tokens so models with different tokenizers stay comparable.
    """
    total = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += len(json.loads(line)["text"].encode("utf-8"))
    return total


@torch.no_grad()
def evaluate_split(
    model: GPT,
    bin_path: Union[str, Path],
    block_size: int,
    device: str = "cpu",
    batch_size: int = 32,
) -> dict:
    """Full deterministic pass over a tokenized split: chunk the token
    stream into non-overlapping `block_size` windows, sum cross-entropy (in
    nats) over every predicted token, and return totals so callers can
    derive perplexity and (combined with a byte count) bits-per-byte.
    """
    model.eval()
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")

    n_blocks = (len(data) - 1) // block_size
    total_nll = 0.0
    total_tokens = 0

    batch_x, batch_y = [], []
    for b in range(n_blocks):
        start = b * block_size
        x = data[start : start + block_size].astype(np.int64)
        y = data[start + 1 : start + 1 + block_size].astype(np.int64)
        batch_x.append(x)
        batch_y.append(y)

        if len(batch_x) == batch_size or b == n_blocks - 1:
            xb = torch.from_numpy(np.stack(batch_x)).to(device)
            yb = torch.from_numpy(np.stack(batch_y)).to(device)
            logits, _, _ = model(xb)
            nll = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), reduction="sum")
            total_nll += nll.item()
            total_tokens += yb.numel()
            batch_x, batch_y = [], []

    avg_nll = total_nll / total_tokens
    return {
        "total_tokens": total_tokens,
        "total_nll_nats": total_nll,
        "cross_entropy_nats": avg_nll,
        "perplexity": math.exp(avg_nll),
    }


def bits_per_byte(total_nll_nats: float, total_bytes: int) -> float:
    return total_nll_nats / (math.log(2) * total_bytes)
