"""Tokenization and batching for the reasoning-finetune datasets.

Unlike pretraining (common/train/data.py), which samples random fixed-length
windows from one huge concatenated token stream, finetuning has a small
number of *discrete* (prompt, answer) examples -- a random window would risk
straddling two unrelated examples. So this module tokenizes every example
once at load time (the whole dataset is a few thousand short examples, small
enough to hold in memory) and pads/truncates each to a fixed length.

Loss masking: only the answer span (plus the trailing EOS) should be
supervised -- the model shouldn't be trained to predict the prompt, which is
given, not generated. Target positions outside the answer span (the prompt,
and any padding) are set to -100, PyTorch's standard "ignore this position"
sentinel for `F.cross_entropy(..., ignore_index=-100)`. This needs no
changes to the shared GPT model (common/model/transformer.py) -- the model
is called with `targets=None` to get raw logits, and the masked loss is
computed here instead, so Phase 2's already-graded model code is untouched.
"""
import json
from pathlib import Path
from typing import List, NamedTuple, Union

import numpy as np


class Example(NamedTuple):
    input_ids: np.ndarray  # (max_len - 1,) int64, padded with pad_id
    target_ids: np.ndarray  # (max_len - 1,) int64, -100 outside the answer span
    prompt: str
    answer: str


def load_jsonl(path: Union[str, Path]) -> List[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def encode_example(sp, prompt: str, answer: str, max_len: int, pad_id: int) -> Example:
    """`full_ids = prompt_tokens + answer_tokens + eos`. `input_ids` is
    `full_ids[:-1]`, `target_ids` is `full_ids[1:]` (standard next-token
    shift), masked so only positions predicting an answer token or the
    final EOS contribute to the loss."""
    eos_id = sp.eos_id()
    prompt_ids = sp.encode(prompt, out_type=int)
    # A leading space keeps the answer's first piece tokenized the way it
    # would be if it continued naturally from the prompt (SentencePiece is
    # space-sensitive), which matters for exact-match generation later.
    answer_ids = sp.encode(" " + answer, out_type=int)
    full_ids = prompt_ids + answer_ids + [eos_id]

    if len(full_ids) > max_len:
        # Truncate from the *front* of the prompt, never the answer -- an
        # over-length example should lose context, not its label.
        overflow = len(full_ids) - max_len
        full_ids = full_ids[overflow:]
        prompt_len = max(0, len(prompt_ids) - overflow)
    else:
        prompt_len = len(prompt_ids)

    input_ids = full_ids[:-1]
    target_ids = full_ids[1:]
    n = len(input_ids)

    target_arr = np.full(max_len - 1, -100, dtype=np.int64)
    # target_ids[j] is supervised iff it's part of (answer + eos), i.e. the
    # token at position j+1 in full_ids is not part of the prompt: j >= prompt_len - 1.
    start = max(0, prompt_len - 1)
    target_arr[start:n] = target_ids[start:n]

    input_arr = np.full(max_len - 1, pad_id, dtype=np.int64)
    input_arr[:n] = input_ids

    return Example(input_ids=input_arr, target_ids=target_arr, prompt=prompt, answer=answer)


class ReasoningDataset:
    """Pre-tokenizes an entire split at construction time (a few thousand
    short examples -- trivial memory footprint, no need for memmap
    streaming the way the multi-hundred-million-token pretraining corpus
    needed)."""

    def __init__(self, jsonl_path: Union[str, Path], sp, max_len: int, pad_id: int):
        raw = load_jsonl(jsonl_path)
        self.examples = [encode_example(sp, ex["prompt"], ex["answer"], max_len, pad_id) for ex in raw]
        self.max_len = max_len
        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.examples)

    def get_batch(self, indices, device: str = "cpu"):
        import torch

        x = torch.from_numpy(np.stack([self.examples[i].input_ids for i in indices]))
        y = torch.from_numpy(np.stack([self.examples[i].target_ids for i in indices]))
        if device != "cpu":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    def sample_batch(self, rng: np.random.Generator, batch_size: int, device: str = "cpu"):
        indices = rng.integers(0, len(self.examples), size=min(batch_size, len(self.examples)))
        return self.get_batch(indices, device=device)
