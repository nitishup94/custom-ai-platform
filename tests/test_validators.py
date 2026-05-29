"""Validator unit tests."""

from __future__ import annotations

import pytest

from utils.validators import validate_language_tag


def test_language_tag_optional() -> None:
    """Empty and None yield no tag."""
    assert validate_language_tag(None) is None
    assert validate_language_tag("") is None
    assert validate_language_tag("   ") is None


def test_language_tag_normalizes() -> None:
    """Tags are normalized to lowercase."""
    assert validate_language_tag("EN") == "en"
    assert validate_language_tag("hi-IN") == "hi-in"


def test_language_tag_rejects_invalid() -> None:
    """Invalid patterns raise."""
    with pytest.raises(ValueError):
        validate_language_tag("hindi")
    with pytest.raises(ValueError):
        validate_language_tag("h")
