"""Decoder-only (GPT-style) Transformer language model, implemented from
scratch in PyTorch using only primitive layers (nn.Linear, nn.Embedding,
nn.LayerNorm, nn.Dropout) -- no nn.Transformer*, no HuggingFace model
classes, no pre-built attention block.

Instantiated independently for Model H and Model L: each language gets its
own GPTConfig, its own weights, and no parameters are shared across
languages.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention, implemented from first principles."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model
        self.d_k = cfg.d_model // cfg.n_heads

        self.w_q = nn.Linear(cfg.d_model, cfg.d_model)
        self.w_k = nn.Linear(cfg.d_model, cfg.d_model)
        self.w_v = nn.Linear(cfg.d_model, cfg.d_model)
        self.w_o = nn.Linear(cfg.d_model, cfg.d_model)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Additive causal mask: 0 on allowed positions (key <= query), -inf
        # on future positions, added to the scores before softmax so future
        # positions get ~0 probability. Buffered (not a parameter) at the
        # model's max sequence length and sliced to the actual length T.
        causal_mask = torch.triu(
            torch.full((cfg.max_seq_len, cfg.max_seq_len), float("-inf")), diagonal=1
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, T, D = x.shape

        q = self.w_q(x)  # (B, T, D)
        k = self.w_k(x)
        v = self.w_v(x)

        # (B, T, D) -> (B, n_heads, T, d_k): split the model dimension into
        # n_heads independent d_k-dim subspaces so each head can attend
        # differently, then move the head dim next to batch so the matmuls
        # below batch cleanly over (B, n_heads).
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention. Dot products of d_k-dimensional
        # vectors have variance that grows with d_k; without the 1/sqrt(d_k)
        # scale, pre-softmax logits grow large, pushing softmax toward
        # near-one-hot outputs with vanishing gradients. Scaling keeps the
        # logit variance ~constant regardless of head dimension.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, h, T, T)
        scores = scores + self.causal_mask[:T, :T]
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v  # (B, h, T, d_k)
        # Concatenate heads back into the model dimension, then mix with W_O.
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.resid_dropout(self.w_o(out))

        return out, (attn.detach() if return_attn else None)


class MLP(nn.Module):
    """Position-wise feed-forward network: two linear layers with a GELU
    non-linearity, inner dimension larger than d_model."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc_in = nn.Linear(cfg.d_model, cfg.ffn_dim)
        self.act = nn.GELU()
        self.fc_out = nn.Linear(cfg.ffn_dim, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc_out(self.act(self.fc_in(x))))


class Block(nn.Module):
    """One Transformer block: pre-norm causal self-attention + pre-norm FFN,
    each wrapped in a residual connection.

    Pre-norm (LayerNorm applied before each sublayer, rather than after) is
    used because it leaves the residual stream unnormalized end-to-end,
    which keeps gradients well-behaved through deep stacks; post-norm GPT
    stacks are markedly more sensitive to learning rate / warmup and easier
    to destabilize at this depth.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        attn_out, attn_weights = self.attn(self.ln1(x), return_attn=return_attn)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, attn_weights


class GPT(nn.Module):
    """Decoder-only Transformer LM: token + positional embeddings -> N causal
    Transformer blocks -> final LayerNorm -> linear output head.

    Positional scheme: learned absolute positional embeddings (a lookup
    table of size max_seq_len). Chosen for simplicity and because it is
    sufficient at this parameter budget; the trade-off is that it hard-caps
    the model at max_seq_len tokens -- inputs longer than that cannot be
    represented (no extrapolation), unlike sinusoidal or relative/RoPE
    schemes which generalize past the training length.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.emb_dropout = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            # Output projection reuses the token embedding matrix: saves
            # vocab_size * d_model parameters and ties input/output token
            # representations (standard practice for LMs at this scale).
            self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.token_emb.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, return_attn: bool = False):
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len, (
            f"sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}"
        )

        pos = torch.arange(T, device=idx.device).unsqueeze(0)  # (1, T)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.emb_dropout(x)

        attn_maps = [] if return_attn else None
        for block in self.blocks:
            x, attn_w = block(x, return_attn=return_attn)
            if return_attn:
                attn_maps.append(attn_w)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Causal LM objective: cross-entropy between logits at position
            # t and the token at position t+1, averaged over all positions.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, attn_maps
