"""Token and positional embeddings."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TokenAndPositionEmbedding(nn.Module):
    """Learned token embeddings plus learned absolute positions."""

    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int, dropout: float) -> None:
        """Create embedding tables."""
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len

    def forward(self, input_ids: Tensor) -> Tensor:
        """Return combined embeddings for *input_ids* (batch, seq)."""
        batch, seq = input_ids.shape
        if seq > self.max_seq_len:
            msg = "sequence length exceeds max_seq_len"
            raise ValueError(msg)
        positions = torch.arange(seq, device=input_ids.device).unsqueeze(0).expand(batch, seq)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        return self.dropout(x)
