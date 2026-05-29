"""Thread-safe dataset job progress and ETA estimation."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from utils.logger import get_logger

_LOGGER = get_logger("progress_tracker")


def _format_eta(seconds: float) -> str:
    """Format seconds as ``Xh YYm`` or ``Ym``."""
    if seconds <= 0 or seconds == float("inf") or seconds != seconds:
        return "unknown"
    total_minutes = int(seconds // 60)
    hours = total_minutes // 60
    minutes = total_minutes % 60
    if hours > 0:
        return f"{hours}h {minutes:02d}m"
    if total_minutes > 0:
        return f"{total_minutes}m"
    return "<1m"


@dataclass
class ProgressSnapshot:
    """Serializable training job snapshot."""

    dataset: str
    batch_size: int
    language: str
    total_rows: int
    processed_rows: int
    lifetime_batches: int
    last_loss: float
    running: bool
    paused: bool
    finished: bool
    ema_batch_seconds: float
    last_checkpoint_label: str = "latest.pt"

    def to_public_dict(self, global_step: int) -> dict[str, Any]:
        """Shape returned by ``GET /dataset/progress``."""
        remaining = max(self.total_rows - self.processed_rows, 0)
        eta_sec = 0.0
        if self.ema_batch_seconds > 0 and self.batch_size > 0 and remaining > 0:
            batches_left = (remaining + self.batch_size - 1) // self.batch_size
            eta_sec = batches_left * self.ema_batch_seconds
        return {
            "processed_records": int(self.processed_rows),
            "remaining_records": int(remaining),
            "global_step": int(global_step),
            "loss": float(self.last_loss),
            "checkpoint": self.last_checkpoint_label,
            "estimated_time_remaining": _format_eta(eta_sec),
        }


class ProgressTracker:
    """Tracks counters, EMA batch duration, and persists progress to disk."""

    def __init__(self, state_path: Path) -> None:
        """Bind persistence path."""
        self._path = state_path
        self._lock = threading.Lock()
        self._snapshot = ProgressSnapshot(
            dataset="",
            batch_size=32,
            language="en",
            total_rows=0,
            processed_rows=0,
            lifetime_batches=0,
            last_loss=0.0,
            running=False,
            paused=False,
            finished=False,
            ema_batch_seconds=0.0,
        )
        self._last_batch_started: float | None = None

    def load(self) -> None:
        """Load snapshot from disk if present."""
        if not self._path.is_file():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _LOGGER.warning("Failed to read progress file", extra={"extra_fields": {"error": str(exc)}})
            return
        with self._lock:
            self._snapshot = ProgressSnapshot(
                dataset=str(raw.get("dataset", "")),
                batch_size=int(raw.get("batch_size", 32)),
                language=str(raw.get("language", "en")),
                total_rows=int(raw.get("total_rows", 0)),
                processed_rows=int(raw.get("processed_rows", 0)),
                lifetime_batches=int(raw.get("lifetime_batches", 0)),
                last_loss=float(raw.get("last_loss", 0.0)),
                running=False,
                paused=bool(raw.get("paused", False)),
                finished=bool(raw.get("finished", False)),
                ema_batch_seconds=float(raw.get("ema_batch_seconds", 0.0)),
                last_checkpoint_label=str(raw.get("last_checkpoint_label", "latest.pt")),
            )

    def save(self) -> None:
        """Persist snapshot atomically (unique temp file to avoid races with concurrent saves)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = asdict(self._snapshot)
            tmp = self._path.parent / f".{self._path.stem}.{uuid.uuid4().hex}.tmp"
            try:
                tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                tmp.replace(self._path)
            except OSError as exc:
                _LOGGER.warning(
                    "Progress save failed",
                    extra={"extra_fields": {"path": str(self._path), "error": str(exc)}},
                )
            finally:
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except OSError:
                        pass

    def begin_job(
        self,
        dataset: str,
        batch_size: int,
        language: str,
        total_rows: int,
        *,
        auto_resume: bool,
    ) -> int:
        """Initialize or resume a dataset job. Returns parquet ``start_row`` cursor."""
        with self._lock:
            can_resume = (
                auto_resume
                and self._snapshot.dataset == dataset
                and self._snapshot.batch_size == batch_size
                and self._snapshot.language == language
                and self._snapshot.processed_rows > 0
                and self._snapshot.processed_rows < total_rows
                and not self._snapshot.finished
            )
            if can_resume:
                self._snapshot.total_rows = total_rows
                self._snapshot.running = True
                self._snapshot.paused = False
                self._snapshot.finished = False
                start_row = self._snapshot.processed_rows
            else:
                self._snapshot = ProgressSnapshot(
                    dataset=dataset,
                    batch_size=batch_size,
                    language=language,
                    total_rows=total_rows,
                    processed_rows=0,
                    lifetime_batches=0,
                    last_loss=0.0,
                    running=True,
                    paused=False,
                    finished=False,
                    ema_batch_seconds=0.0,
                )
                start_row = 0
        self.save()
        return start_row

    def mark_batch_start(self) -> None:
        """Record wall time at batch start for EMA."""
        self._last_batch_started = time.perf_counter()

    def mark_batch_end(self, rows_in_batch: int, mean_loss: float) -> None:
        """Update counters after a processed batch."""
        if rows_in_batch <= 0:
            return
        elapsed = 0.0
        if self._last_batch_started is not None:
            elapsed = max(time.perf_counter() - self._last_batch_started, 1e-9)
        with self._lock:
            self._snapshot.processed_rows += rows_in_batch
            self._snapshot.lifetime_batches += 1
            self._snapshot.last_loss = float(mean_loss)
            alpha = 0.1
            if self._snapshot.ema_batch_seconds <= 0:
                self._snapshot.ema_batch_seconds = elapsed
            else:
                self._snapshot.ema_batch_seconds = (1 - alpha) * self._snapshot.ema_batch_seconds + alpha * elapsed
        self.save()

    def set_checkpoint_label(self, label: str) -> None:
        """Update last milestone label for API consumers."""
        with self._lock:
            self._snapshot.last_checkpoint_label = label
        self.save()

    def mark_finished(self) -> None:
        """Mark the job complete and stop running flag."""
        with self._lock:
            self._snapshot.running = False
            self._snapshot.paused = False
            self._snapshot.finished = True
        self.save()

    def mark_stopped(self) -> None:
        """Mark job interrupted by operator."""
        with self._lock:
            self._snapshot.running = False
            self._snapshot.paused = False
        self.save()

    def set_paused(self, paused: bool) -> None:
        """Toggle pause flag."""
        with self._lock:
            self._snapshot.paused = paused
        self.save()

    def snapshot(self) -> ProgressSnapshot:
        """Return a copy of the snapshot under lock."""
        with self._lock:
            return ProgressSnapshot(**asdict(self._snapshot))

    def is_running(self) -> bool:
        """Return whether worker should be active."""
        with self._lock:
            return bool(self._snapshot.running and not self._snapshot.finished)

    def is_paused(self) -> bool:
        """Return pause flag."""
        with self._lock:
            return bool(self._snapshot.paused)
