"""Structured logging setup."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from utils.config import AppConfig


class JsonFormatter(logging.Formatter):
    """Serialize log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        """Format *record* as JSON."""
        payload: dict[str, Any] = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if hasattr(record, "extra_fields"):
            extra = getattr(record, "extra_fields")
            if isinstance(extra, dict):
                payload.update(extra)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(config: AppConfig) -> logging.Logger:
    """Configure root logging from ``config`` (console and/or file, JSON lines)."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(config.log_level.upper())

    formatter = JsonFormatter()
    use_console = bool(config.log_console_enabled)
    file_path = config.resolved_log_file_path()
    if not use_console and file_path is None:
        use_console = True

    if use_console:
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(formatter)
        root.addHandler(stream)

    if file_path is not None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    if config.log_startup_summary:
        file_label = str(file_path) if file_path is not None else "off"
        print(
            f"[logging] console={'on' if use_console else 'off'} file={file_label}",
            file=sys.stderr,
            flush=True,
        )

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("chromadb.segment.impl.vector.local_persistent_hnsw").setLevel(logging.ERROR)
    return logging.getLogger("app")


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the application namespace."""
    return logging.getLogger(f"app.{name}")
