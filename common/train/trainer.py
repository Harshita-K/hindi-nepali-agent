"""Training loop for pretraining a from-scratch decoder-only Transformer LM
on one language's monolingual corpus.

Designed to run on Google Colab, where sessions can terminate unexpectedly:
checkpoints (model weights, optimizer state, scheduler state, training step,
tokens seen, config) are written to `out_dir` at every `save_interval`, and
the trainer automatically resumes from `out_dir/latest.pt` if present, so an
interrupted run loses at most `save_interval` steps of progress.
"""
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch

from common.model.config import GPTConfig
from common.model.transformer import GPT

from .checkpoint import load_checkpoint, save_checkpoint
from .data import get_batch as sample_batch
from .optim import build_optimizer, build_scheduler


@dataclass
class TrainConfig:
    batch_size: int = 32
    grad_accum_steps: int = 4
    block_size: int = 512
    max_steps: int = 8000
    warmup_steps: int = 200
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 200
    eval_iters: int = 50
    save_interval: int = 200
    log_interval: int = 20
    seed: int = 42
    amp: bool = True
    out_dir: str = "checkpoints"
    data_dir: str = "data/bin"

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TrainConfig":
        import yaml

        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)


class Trainer:
    def __init__(self, model_cfg: GPTConfig, train_cfg: TrainConfig, device: Optional[str] = None):
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_type = "cuda" if self.device == "cuda" else "cpu"

        assert train_cfg.block_size <= model_cfg.max_seq_len, (
            f"train_config.block_size ({train_cfg.block_size}) exceeds "
            f"model_config.max_seq_len ({model_cfg.max_seq_len}) -- the learned "
            f"positional embedding table isn't large enough for this many positions."
        )

        torch.manual_seed(train_cfg.seed)

        self.model = GPT(model_cfg).to(self.device)
        self.optimizer = build_optimizer(self.model, train_cfg.lr, train_cfg.weight_decay)
        self.scheduler = build_scheduler(
            self.optimizer, train_cfg.warmup_steps, train_cfg.max_steps, train_cfg.min_lr_ratio
        )

        self.amp_enabled = train_cfg.amp and self.device_type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

        self.out_dir = Path(train_cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / "train_log.csv"
        self.latest_ckpt = self.out_dir / "latest.pt"

        self.train_bin = Path(train_cfg.data_dir) / "train.bin"
        self.val_bin = Path(train_cfg.data_dir) / "val.bin"

        self.step = 0
        self.tokens_seen = 0
        self.best_val_loss = float("inf")

        self._maybe_resume()
        self._init_log()

    def _init_log(self) -> None:
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["step", "tokens_seen", "train_loss", "val_loss", "val_ppl", "lr", "seconds"]
                )

    def _maybe_resume(self) -> None:
        if self.latest_ckpt.exists():
            ckpt = load_checkpoint(
                self.latest_ckpt, self.model, self.optimizer, self.scheduler, map_location=self.device
            )
            self.step = ckpt["step"]
            self.tokens_seen = ckpt["tokens_seen"]
            self.best_val_loss = ckpt.get("best_val_loss") or float("inf")
            print(f"Resumed from {self.latest_ckpt} at step {self.step} ({self.tokens_seen:,} tokens seen)")

    def get_batch(self, split: str):
        path = self.train_bin if split == "train" else self.val_bin
        return sample_batch(path, self.train_cfg.block_size, self.train_cfg.batch_size, self.device)

    @torch.no_grad()
    def estimate_val_loss(self) -> float:
        self.model.eval()
        losses = []
        for _ in range(self.train_cfg.eval_iters):
            x, y = self.get_batch("val")
            with torch.autocast(device_type=self.device_type, enabled=self.amp_enabled):
                _, loss, _ = self.model(x, targets=y)
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
            config={"model": self.model_cfg.to_dict(), "train": vars(self.train_cfg)},
            best_val_loss=self.best_val_loss,
        )

    def train(self) -> None:
        cfg = self.train_cfg
        self.model.train()
        t0 = time.time()

        while self.step < cfg.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for _ in range(cfg.grad_accum_steps):
                x, y = self.get_batch("train")
                with torch.autocast(device_type=self.device_type, enabled=self.amp_enabled):
                    _, loss, _ = self.model(x, targets=y)
                    loss = loss / cfg.grad_accum_steps
                self.scaler.scale(loss).backward()
                accum_loss += loss.item()
                self.tokens_seen += x.numel()

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            self.step += 1
            lr = self.scheduler.get_last_lr()[0]

            if self.step % cfg.log_interval == 0:
                print(
                    f"step {self.step}/{cfg.max_steps} | train_loss {accum_loss:.4f} "
                    f"| lr {lr:.2e} | tokens {self.tokens_seen:,} | {time.time() - t0:.1f}s"
                )

            val_loss = None
            if self.step % cfg.eval_interval == 0 or self.step == cfg.max_steps:
                val_loss = self.estimate_val_loss()
                val_ppl = torch.exp(torch.tensor(val_loss)).item()
                print(f"  [eval] step {self.step} | val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save(tag="best")

            if self.step % cfg.log_interval == 0 or val_loss is not None:
                with open(self.log_path, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [
                            self.step,
                            self.tokens_seen,
                            round(accum_loss, 4),
                            round(val_loss, 4) if val_loss is not None else "",
                            round(torch.exp(torch.tensor(val_loss)).item(), 4) if val_loss is not None else "",
                            lr,
                            round(time.time() - t0, 1),
                        ]
                    )

            if self.step % cfg.save_interval == 0 or self.step == cfg.max_steps:
                self.save(tag="latest")

        print("Training complete.")
