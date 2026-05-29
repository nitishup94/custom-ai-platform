"""Byte-token length helpers for LM / embedding windows."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model.tokenizer import ByteTokenizer


def clip_byte_token_ids(tokenizer: "ByteTokenizer", text: str, max_tokens: int) -> list[int]:
    """Encode *text* and keep at most *max_tokens* ids from the start (prefix)."""
    if max_tokens < 1:
        return []
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return ids
    return ids[:max_tokens]
