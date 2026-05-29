"""Optimizer factory."""

from __future__ import annotations

import torch
from torch import nn

from utils.config import AppConfig


def build_optimizer(model: nn.Module, config: AppConfig) -> torch.optim.Optimizer:
    """Create AdamW optimizer for *model* parameters."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
