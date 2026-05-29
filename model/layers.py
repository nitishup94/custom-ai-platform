"""Low-level building blocks for the custom transformer."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class LayerNorm(nn.Module):
    """Layer normalization with learnable scale and shift."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        """Initialize normalization over the last dimension."""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Apply layer norm to *x* with shape (..., dim)."""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        normed = (x - mean) / torch.sqrt(var + self.eps)
        return normed * self.weight + self.bias


class FeedForward(nn.Module):
    """Two-layer MLP with GELU activation."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        """Build feed-forward block."""
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Apply MLP to *x*."""
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


def sinusoidal_positions(length: int, dim: int, device: torch.device) -> Tensor:
    """Create fixed sinusoidal position encodings (unused by default; kept for scaling)."""
    position = torch.arange(length, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
