"""Checkpoint persistence for model weights and training counters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from utils.config import AppConfig
from utils.logger import get_logger

_LOGGER = get_logger("checkpoint")


def latest_checkpoint_path(config: AppConfig) -> Path:
    """Return path to ``latest.pt``."""
    return config.resolved_checkpoint_dir() / "latest.pt"


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    global_step: int,
    epoch: int,
) -> None:
    """Atomically persist training state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, tmp)
    tmp.replace(path)
    _LOGGER.info("Saved checkpoint", extra={"extra_fields": {"path": str(path)}})


def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer | None) -> tuple[int, int]:
    """Load weights into *model* and optionally *optimizer*. Returns (step, epoch)."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    step = int(payload.get("global_step", 0))
    epoch = int(payload.get("epoch", 0))
    _LOGGER.info(
        "Loaded checkpoint",
        extra={"extra_fields": {"path": str(path), "global_step": step, "epoch": epoch}},
    )
    return step, epoch


def maybe_save_epoch_copy(
    config: AppConfig,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    global_step: int,
) -> None:
    """Persist ``epoch_{n}.pt`` alongside latest."""
    path = config.resolved_checkpoint_dir() / f"epoch_{epoch}.pt"
    save_checkpoint(path, model, optimizer, global_step=global_step, epoch=epoch)
