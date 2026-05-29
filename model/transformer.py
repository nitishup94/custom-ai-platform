"""Decoder-style transformer language model."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from model.attention import MultiHeadSelfAttention, build_causal_mask
from model.embedding import TokenAndPositionEmbedding
from model.layers import FeedForward, LayerNorm


class TransformerBlock(nn.Module):
    """Pre-norm transformer decoder block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        """Construct one decoder layer."""
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: Tensor, attn_mask: Tensor | None) -> Tensor:
        """Apply self-attention and feed-forward with residuals."""
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    """Causal language model built from scratch."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        """Initialize weights randomly."""
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.embed = TokenAndPositionEmbedding(vocab_size, max_seq_len, d_model, dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        )
        self.final_norm = LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def embed_inputs(self, input_ids: Tensor) -> Tensor:
        """Return token+position embeddings before transformer blocks."""
        return self.embed(input_ids)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Compute logits for next-token prediction (batch, seq, vocab)."""
        x = self.embed_inputs(input_ids)
        seq_len = x.size(1)
        mask = build_causal_mask(seq_len, x.device)
        for block in self.blocks:
            x = block(x, mask)
        x = self.final_norm(x)
        return self.lm_head(x)
