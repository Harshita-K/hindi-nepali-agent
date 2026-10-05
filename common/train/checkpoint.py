"""Checkpoint save/resume. A checkpoint bundles model weights, optimizer
state, LR scheduler state, the training step, tokens seen, and the run
config into a single file, so that a Colab session that dies mid-run can
resume training exactly where it left off (mandatory per the assignment).
"""
from pathlib import Path
from typing import Optional, Union

import torch


def save_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    tokens_seen: int,
    config: dict,
    best_val_loss: Optional[float] = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "step": step,
            "tokens_seen": tokens_seen,
            "config": config,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def load_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    map_location: str = "cpu",
) -> dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt
