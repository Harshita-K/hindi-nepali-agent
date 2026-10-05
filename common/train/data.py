"""Tokenize a language's JSONL splits into flat token-id binaries for
memory-mapped random-access training (nanoGPT-style): avoids re-tokenizing
text or materializing the full corpus as Python objects on every epoch, and
keeps peak memory bounded regardless of corpus size.
"""
import json
from pathlib import Path
from typing import Union

import numpy as np
import sentencepiece as spm


def build_bin(
    jsonl_path: Union[str, Path],
    spm_model_path: Union[str, Path],
    out_bin_path: Union[str, Path],
    out_meta_path: Union[str, Path],
    flush_every: int = 2_000_000,
) -> dict:
    """Encode every doc's "text" field with the language's SentencePiece
    model, append an EOS between docs, and stream the resulting token ids
    to `out_bin_path` as uint16 (vocab sizes here are well under 65536).
    Writes a small JSON sidecar with corpus/token counts to `out_meta_path`.
    """
    sp = spm.SentencePieceProcessor(model_file=str(spm_model_path))
    eos_id = sp.eos_id()

    out_bin_path = Path(out_bin_path)
    out_bin_path.parent.mkdir(parents=True, exist_ok=True)

    n_tokens = 0
    n_docs = 0
    buffer = []

    with open(jsonl_path, encoding="utf-8") as fin, open(out_bin_path, "wb") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            text = json.loads(line)["text"]
            ids = sp.encode(text, out_type=int)
            ids.append(eos_id)
            buffer.extend(ids)
            n_tokens += len(ids)
            n_docs += 1

            if len(buffer) >= flush_every:
                np.array(buffer, dtype=np.uint16).tofile(fout)
                buffer = []

        if buffer:
            np.array(buffer, dtype=np.uint16).tofile(fout)

    meta = {
        "source": str(jsonl_path),
        "spm_model": str(spm_model_path),
        "vocab_size": sp.get_piece_size(),
        "num_docs": n_docs,
        "num_tokens": n_tokens,
        "dtype": "uint16",
    }
    Path(out_meta_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def get_batch(bin_path: Union[str, Path], block_size: int, batch_size: int, device: str = "cpu"):
    """Sample `batch_size` random (x, y) windows of length `block_size` from
    a memmap'd uint16 token stream, where y is x shifted by one position
    (next-token targets). Re-opens the memmap each call -- cheap (no data
    copy) and safe to call repeatedly from a single training loop.
    """
    import torch

    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    if device != "cpu":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
