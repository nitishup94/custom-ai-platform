"""Pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture()
def tiny_model_env(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Shrink the transformer for fast integration tests."""
    root = tmp_path_factory.mktemp("storage")
    monkeypatch.setenv("STORAGE_ROOT", str(root))
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("D_MODEL", "64")
    monkeypatch.setenv("N_HEADS", "4")
    monkeypatch.setenv("N_LAYERS", "1")
    monkeypatch.setenv("D_FF", "128")
    monkeypatch.setenv("MAX_SEQ_LEN", "128")
    monkeypatch.setenv("VOCAB_SIZE", "320")
    monkeypatch.setenv("WARMUP_STEPS", "0")
    monkeypatch.setenv("CHECKPOINT_EVERY_STEPS", "9999")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "2")
