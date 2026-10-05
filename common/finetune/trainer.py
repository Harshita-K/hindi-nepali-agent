"""Finetuning loop for the reasoning task: starts from a language's
pretrained checkpoint, trains on the small synthetic reasoning dataset with
a masked causal-LM loss (only the answer span is supervised -- see
common/finetune/data.py), and checkpoints in the exact same resume-capable
format pretraining uses (common/train/checkpoint.py), so a Colab disconnect
mid-finetune is recoverable the same way a disconnect mid-pretraining was.

Deliberately reuses common/train/checkpoint.py and common/train/optim.py
rather than duplicating them -- the checkpoint format and the AdamW/
warmup-cosine schedule are exactly the mechanisms Phase 2 already built and
that this phase's spec explicitly asks to reuse ("save finetuned
checkpoints in the same resume-capable format as pretraining"). It does
*not* reuse common/train/trainer.py's `Trainer` directly, because that
class's batching (`common/train/data.py::get_batch`) samples random windows
from one long concatenated token stream, which is the right approach for
~500M tokens of raw corpus but wrong here: a random window could straddle
two unrelated reasoning examples. Finetuning needs example-level batching
with per-example padding and loss masking instead.
"""
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from common.model.config import GPTConfig
from common.model.transformer import GPT
from common.train.checkpoint import load_checkpoint, save_checkpoint
from common.train.optim import build_optimizer, build_scheduler

from .data import ReasoningDataset


@dataclass
class FinetuneConfig:
    batch_size: int = 32
    max_steps: int = 1000
    warmup_steps: int = 50
    lr: float = 2e-5  # much lower than pretraining's 3e-4: the dataset is
    # tiny relative to the ~500M-token pretraining corpus, and a large LR
    # risks catastrophically overwriting the pretrained language modeling
    # ability while fitting this narrow task.
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_iters: int = 20
    save_interval: int = 100
    log_interval: int = 20
    seed: int = 42
    amp: bool = True
    max_len: int = 96  # covers p99 sequence length for both languages' reasoning data with margin
    out_dir: str = "finetune_checkpoints"
    data_dir: str = "data/reasoning"
    pretrained_ckpt: str = ""  # required: path to the Phase 2 best.pt to start from

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "FinetuneConfig":
        import yaml

        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)


class FinetuneTrainer:
    def __init__(self, model_cfg: GPTConfig, ft_cfg: FinetuneConfig, sp, device: Optional[str] = None):
        self.model_cfg = model_cfg
        self.ft_cfg = ft_cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_type = "cuda" if self.device == "cuda" else "cpu"

        torch.manual_seed(ft_cfg.seed)

        self.model = GPT(model_cfg).to(self.device)
        self.optimizer = build_optimizer(self.model, ft_cfg.lr, ft_cfg.weight_decay)
        self.scheduler = build_scheduler(self.optimizer, ft_cfg.warmup_steps, ft_cfg.max_steps, ft_cfg.min_lr_ratio)

        self.amp_enabled = ft_cfg.amp and self.device_type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

        self.out_dir = Path(ft_cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / "finetune_log.csv"
        self.latest_ckpt = self.out_dir / "latest.pt"

        data_dir = Path(ft_cfg.data_dir)
        self.train_ds = ReasoningDataset(data_dir / "train.jsonl", sp, ft_cfg.max_len, model_cfg.pad_id)
        self.val_ds = ReasoningDataset(data_dir / "val.jsonl", sp, ft_cfg.max_len, model_cfg.pad_id)

        self.step = 0
        self.tokens_seen = 0
        self.best_val_loss = float("inf")

        self._maybe_resume_or_init_from_pretrained()
        self._init_log()

    def _init_log(self) -> None:
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["step", "examples_seen", "train_loss", "val_loss", "lr", "seconds"]
                )

    def _maybe_resume_or_init_from_pretrained(self) -> None:
        if self.latest_ckpt.exists():
            # A finetune run is already in progress (e.g. a Colab disconnect
            # mid-finetune) -- resume its full state, same as pretraining.
            ckpt = load_checkpoint(
                self.latest_ckpt, self.model, self.optimizer, self.scheduler, map_location=self.device
            )
            self.step = ckpt["step"]
            self.tokens_seen = ckpt["tokens_seen"]
            self.best_val_loss = ckpt.get("best_val_loss") or float("inf")
            print(f"Resumed finetuning from {self.latest_ckpt} at step {self.step}")
        else:
            # First time starting this finetune run: load the *pretrained*
            # checkpoint's weights only. Optimizer/scheduler are left fresh
            # -- Adam's moment estimates from pretraining aren't meaningful
            # for this new, much smaller loss landscape, and step starts at 0
            # so the warmup/cosine schedule above applies to finetuning only.
            assert self.ft_cfg.pretrained_ckpt, "ft_cfg.pretrained_ckpt is required to start a new finetune run"
            load_checkpoint(self.ft_cfg.pretrained_ckpt, self.model, optimizer=None, scheduler=None, map_location=self.device)
            print(f"Initialized from pretrained checkpoint {self.ft_cfg.pretrained_ckpt} (weights only)")

    def _masked_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.model(x, targets=None)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)

    @torch.no_grad()
    def estimate_val_loss(self) -> float:
        self.model.eval()
        rng = np.random.default_rng(123)  # fixed seed -> val loss is comparable across evals
        losses = []
        for _ in range(self.ft_cfg.eval_iters):
            x, y = self.val_ds.sample_batch(rng, self.ft_cfg.batch_size, self.device)
            with torch.autocast(device_type=self.device_type, enabled=self.amp_enabled):
                loss = self._masked_loss(x, y)
            losses.append(loss.item())
        self.model.train()
        return sum(losses) / len(losses)

    def save(self, tag: str = "latest") -> None:
        path = self.out_dir / f"{tag}.pt"
        save_checkpoint(
            path,
            self.model,
            self.optimizer,
            self.scheduler,
            self.step,
            self.tokens_seen,
            config={
                "model": self.model_cfg.to_dict(),
                "finetune": vars(self.ft_cfg),
            },
            best_val_loss=self.best_val_loss,
        )

    def train(self) -> None:
        cfg = self.ft_cfg
        rng = np.random.default_rng(cfg.seed)
        self.model.train()
        t0 = time.time()

        while self.step < cfg.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            x, y = self.train_ds.sample_batch(rng, cfg.batch_size, self.device)
            with torch.autocast(device_type=self.device_type, enabled=self.amp_enabled):
                loss = self._masked_loss(x, y)
            self.scaler.scale(loss).backward()
            self.tokens_seen += int((y != -100).sum().item())

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            self.step += 1
            lr = self.scheduler.get_last_lr()[0]
            train_loss = loss.item()

            if self.step % cfg.log_interval == 0:
                print(
                    f"step {self.step}/{cfg.max_steps} | train_loss {train_loss:.4f} "
                    f"| lr {lr:.2e} | {time.time() - t0:.1f}s"
                )

            val_loss = None
            if self.step % cfg.eval_interval == 0 or self.step == cfg.max_steps:
                val_loss = self.estimate_val_loss()
                print(f"  [eval] step {self.step} | val_loss {val_loss:.4f}")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save(tag="best")

            if self.step % cfg.log_interval == 0 or val_loss is not None:
                with open(self.log_path, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [
                            self.step,
                            self.step * cfg.batch_size,
                            round(train_loss, 4),
                            round(val_loss, 4) if val_loss is not None else "",
                            lr,
                            round(time.time() - t0, 1),
                        ]
                    )

            if self.step % cfg.save_interval == 0 or self.step == cfg.max_steps:
                self.save(tag="latest")

        print("Finetuning complete.")
