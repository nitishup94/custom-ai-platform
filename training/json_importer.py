"""Load JSON array or JSONL records from a single training file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from utils.logger import get_logger

_LOGGER = get_logger("json_importer")


def _loads_records(raw_text: str) -> list[dict[str, Any]]:
    """Parse *raw_text* as a JSON array, a single JSON object, or JSONL (one object per line)."""
    text = raw_text.strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            msg = "JSON array root must be a list"
            raise ValueError(msg)
        return [x for x in data if isinstance(x, dict)]
    if text.startswith("{"):
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) == 1:
            obj = json.loads(lines[0])
            if not isinstance(obj, dict):
                msg = "JSON object must be a mapping"
                raise ValueError(msg)
            return [obj]
        out: list[dict[str, Any]] = []
        for line in lines:
            obj = json.loads(line)
            if not isinstance(obj, dict):
                msg = "JSONL lines must be JSON objects"
                raise ValueError(msg)
            out.append(obj)
        return out
    msg = "JSON dataset must start with '[' (array) or '{' (object / JSONL)"
    raise ValueError(msg)


class JsonImporter:
    """Holds decoded records for one JSON/JSONL file and yields fixed-size batches."""

    def __init__(self, path: Path) -> None:
        """Load and normalize *path* (must exist)."""
        if not path.is_file():
            msg = f"JSON dataset file not found: {path}"
            raise FileNotFoundError(msg)
        self._path = path
        raw = path.read_text(encoding="utf-8")
        self._rows = _loads_records(raw)
        _LOGGER.info(
            "Loaded JSON dataset",
            extra={"extra_fields": {"path": str(path), "records": len(self._rows)}},
        )

    @property
    def source_path(self) -> Path:
        """Filesystem path."""
        return self._path

    def total_records(self) -> int:
        """Row count after parsing."""
        return len(self._rows)

    def iter_record_batches(self, batch_size: int, *, start_index: int = 0) -> Iterator[list[dict[str, Any]]]:
        """Yield sublists of raw dict records (length up to *batch_size*)."""
        if batch_size < 1:
            msg = "batch_size must be positive"
            raise ValueError(msg)
        n = len(self._rows)
        idx = max(0, start_index)
        while idx < n:
            yield self._rows[idx : idx + batch_size]
            idx += batch_size
