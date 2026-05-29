"""Shared validation helpers for text and API payloads."""

from __future__ import annotations

import re
from typing import Final

_MAX_FIELD_LEN: Final[int] = 8192
_MAX_CATEGORY_LEN: Final[int] = 256
_MAX_LANGUAGE_TAG_LEN: Final[int] = 32
_CATEGORY_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_\- ]{1,256}$")
_LANGUAGE_TAG_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[a-zA-Z]{2}(-[a-zA-Z0-9]{1,12})?$",
)


def validate_non_empty_text(value: str, field: str) -> str:
    """Strip and validate that *value* is non-empty after strip."""
    stripped = value.strip()
    if not stripped:
        msg = f"{field} must be a non-empty string"
        raise ValueError(msg)
    return stripped


def validate_bounded_text(value: str, field: str, max_len: int = _MAX_FIELD_LEN) -> str:
    """Validate length of *value* after stripping."""
    text = validate_non_empty_text(value, field)
    if len(text) > max_len:
        msg = f"{field} exceeds maximum length of {max_len}"
        raise ValueError(msg)
    return text


def validate_category(value: str) -> str:
    """Validate training category label."""
    text = validate_non_empty_text(value, "category")
    if len(text) > _MAX_CATEGORY_LEN:
        msg = f"category exceeds maximum length of {_MAX_CATEGORY_LEN}"
        raise ValueError(msg)
    if not _CATEGORY_PATTERN.match(text):
        msg = "category must contain only letters, digits, space, hyphen, or underscore"
        raise ValueError(msg)
    return text


def validate_language_tag(value: str | None) -> str | None:
    """Validate optional BCP47-style language tag (e.g. ``en``, ``hi``, ``hi-in``)."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if len(text) > _MAX_LANGUAGE_TAG_LEN:
        msg = f"language exceeds maximum length of {_MAX_LANGUAGE_TAG_LEN}"
        raise ValueError(msg)
    if not _LANGUAGE_TAG_PATTERN.match(text):
        msg = "language must look like a BCP47 primary tag, e.g. en, hi, hi-IN"
        raise ValueError(msg)
    return text.lower()
