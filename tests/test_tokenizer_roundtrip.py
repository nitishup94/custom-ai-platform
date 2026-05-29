"""Tokenizer roundtrip tests."""

from __future__ import annotations

import pytest

from model.tokenizer import ByteTokenizer, ensure_vocab_file


def test_tokenizer_utf8_roundtrip(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Byte tokenizer should roundtrip arbitrary UTF-8."""
    root = tmp_path_factory.mktemp("tok")
    vocab_path = root / "vocab.json"
    ensure_vocab_file(vocab_path, model_vocab_size=512)
    tokenizer = ByteTokenizer.from_path(vocab_path, model_vocab_size=512)
    text = "Hello 世界 byte-level"
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert tokenizer.decode(ids, skip_special_tokens=True) == text
