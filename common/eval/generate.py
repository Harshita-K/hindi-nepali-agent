"""Autoregressive text generation from a trained GPT: greedy (argmax)
decoding and temperature-based sampling, used for the Phase 2.3
"Generation quality" evaluation.
"""
from typing import Optional

import torch
import torch.nn.functional as F

from common.model.transformer import GPT


@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: Optional[float] = None,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    """Extend `idx` (B, T) by up to `max_new_tokens` tokens.

    temperature=None (or 0) -> greedy decoding: always pick the argmax token.
    temperature>0 -> multinomial sampling from softmax(logits / temperature).

    Stops early once `eos_id` is generated, if given (only meaningful for
    batch size 1, since sequences in a batch would otherwise finish at
    different lengths).
    """
    model.eval()
    max_seq_len = model.cfg.max_seq_len

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -max_seq_len:]
        logits, _, _ = model(idx_cond)
        logits = logits[:, -1, :]  # (B, vocab) -- next-token logits only

        if not temperature:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_token], dim=1)

        if eos_id is not None and idx.size(0) == 1 and next_token.item() == eos_id:
            break

    return idx
