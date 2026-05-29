"""Learning-rate schedulers."""

from __future__ import annotations

import math
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from utils.config import AppConfig


class WarmupCosineScheduler(LRScheduler):
    """Linear warmup followed by cosine decay to a small floor."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        """Configure piecewise schedule."""
        self.warmup_steps = max(1, warmup_steps)
        self.max_steps = max(self.warmup_steps + 1, max_steps)
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Return learning rates for all param groups."""
        step = self.last_epoch + 1
        if step < self.warmup_steps:
            scale = float(step) / float(self.warmup_steps)
            return [base * scale for base in self.base_lrs]
        progress = (step - self.warmup_steps) / float(self.max_steps - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        floor = self.min_lr_ratio
        scale = floor + (1.0 - floor) * cosine
        return [base * scale for base in self.base_lrs]


def build_scheduler(optimizer: Optimizer, config: AppConfig) -> Any:
    """Return scheduler instance or ``None`` when warmup disabled."""
    if config.warmup_steps <= 0:
        return None
    return WarmupCosineScheduler(
        optimizer,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_train_steps,
    )
