"""Discover JSON training files under ``storage/datasets/json/``."""

from __future__ import annotations

from pathlib import Path

from utils.config import AppConfig


def json_datasets_directory(config: AppConfig) -> Path:
    """Return absolute ``storage/datasets/json`` under the configured storage root."""
    rel = Path("storage") / "datasets" / "json"
    return (config.resolved_storage_root() / rel).resolve()


def list_json_dataset_files(config: AppConfig) -> list[str]:
    """Return sorted basenames of ``*.json`` files (file-level only, no recursion)."""
    root = json_datasets_directory(config)
    if not root.is_dir():
        return []
    names = sorted(p.name for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".json")
    return names
