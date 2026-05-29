"""Structured log fields for supervised training rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model.tokenizer import ByteTokenizer


def training_sample_extra_fields(
    tokenizer: "ByteTokenizer",
    instruction: str,
    response: str | None,
    *,
    preview_chars: int = 280,
) -> dict[str, object]:
    """Return ``extra_fields`` for JSON logs: token counts and instruction preview."""
    in_ids = tokenizer.encode(instruction, add_special_tokens=False)
    out_ids = tokenizer.encode(response, add_special_tokens=False) if response else []
    preview = instruction[:preview_chars]
    if len(instruction) > preview_chars:
        preview += "..."
    return {
        "instruction_tokens": len(in_ids),
        "response_tokens": len(out_ids),
        "instruction_preview": preview,
    }
