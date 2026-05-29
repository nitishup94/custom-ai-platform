"""Multi-head self-attention with causal masking."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product multi-head attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        """Initialize attention projections."""
        super().__init__()
        if d_model % n_heads != 0:
            msg = "d_model must be divisible by n_heads"
            raise ValueError(msg)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(float(self.head_dim))

        self.qkv = nn.Linear(d_model, d_model * 3, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """Apply attention to *x* with shape (batch, seq, d_model).

        *attn_mask* when provided must be additive (0 or -inf) with shape
        broadcastable to (batch, heads, seq, seq).
        """
        batch, seq_len, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def reshape_heads(t: Tensor) -> Tensor:
            """Split last dim into heads."""
            return t.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            scores = scores + attn_mask
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.proj(out)


def build_causal_mask(seq_len: int, device: torch.device) -> Tensor:
    """Build additive causal mask of shape (1, 1, seq, seq)."""
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.float32) * float("-inf"),
        diagonal=1,
    )
    return mask.view(1, 1, seq_len, seq_len)
