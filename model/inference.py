"""Inference utilities: retrieval embeddings and text generation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from model.transformer import TransformerLM


class InferenceEngine:
    """Pooling for vector retrieval and autoregressive generation."""

    def __init__(self, model: TransformerLM, pad_id: int, eos_id: int) -> None:
        """Bind *model* and special ids."""
        self._model = model
        self._pad_id = pad_id
        self._eos_id = eos_id

    @torch.inference_mode()
    def retrieval_embedding(self, input_ids: Tensor) -> Tensor:
        """Mean-pool pre-transformer embeddings (L2-normalized).

        *input_ids* has shape (1, seq). Returns shape (1, d_model).
        """
        self._model.eval()
        emb = self._model.embed_inputs(input_ids)
        mask = (input_ids != self._pad_id).float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = (emb * mask).sum(dim=1) / denom
        return F.normalize(pooled, dim=-1, p=2, eps=1e-12)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> Tensor:
        """Autoregressively extend *input_ids* (batch=1) by up to *max_new_tokens*."""
        self._model.eval()
        if input_ids.size(0) != 1:
            msg = "generation currently supports batch size 1"
            raise ValueError(msg)

        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            if generated.size(1) >= self._model.max_seq_len:
                break
            logits = self._model(generated)[:, -1, :]
            logits = logits / max(temperature, 1e-6)
            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                min_values = values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_id], dim=1)
            if int(next_id.item()) == self._eos_id:
                break
        return generated
