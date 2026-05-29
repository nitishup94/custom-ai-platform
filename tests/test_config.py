"""AppConfig environment defaults."""

from __future__ import annotations

import pytest

from pydantic import ValidationError

from utils.config import AppConfig


def test_default_language_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """DEFAULT_LANGUAGE normalizes and can be cleared."""
    monkeypatch.setenv("DEFAULT_LANGUAGE", "HI-in")
    cfg = AppConfig()
    assert cfg.default_language == "hi-in"

    monkeypatch.setenv("DEFAULT_LANGUAGE", "   ")
    cfg = AppConfig()
    assert cfg.default_language == ""

    monkeypatch.delenv("DEFAULT_LANGUAGE", raising=False)
    cfg = AppConfig()
    assert cfg.default_language == "en"


def test_resolved_log_file_path_relative(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """LOG_FILE_PATH resolves under STORAGE_ROOT when relative."""
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("LOG_FILE_PATH", "storage/logs/app.log")
    cfg = AppConfig()
    assert cfg.resolved_log_file_path() == tmp_path / "storage" / "logs" / "app.log"


def test_default_language_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid DEFAULT_LANGUAGE fails at settings load."""
    monkeypatch.setenv("DEFAULT_LANGUAGE", "not_a_lang")
    with pytest.raises(ValidationError):
        AppConfig()
