"""Background dataset training coordinator (OpenOrca parquet → trainer + Chroma)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from training.batch_processor import rows_to_payloads, train_payload_batch
from training.checkpoint import save_checkpoint
from training.openorca_importer import OpenOrcaImporter
from training.progress_tracker import ProgressTracker
from utils.logger import get_logger
from utils.validators import validate_language_tag

if TYPE_CHECKING:
    from app.container import AppContainer

_LOGGER = get_logger("dataset_queue")

_DATASET_PATHS = {
    "openorca": Path("storage") / "datasets" / "1M-GPT4-Augmented.parquet",
}

_MILESTONE_BATCHES = 1000


class DatasetTrainingCoordinator:
    """Runs parquet batch training on a worker thread."""

    def __init__(self, container: "AppContainer") -> None:
        """Bind the application container for model and vector access."""
        self._container = container
        state_path = container.config.resolved_checkpoint_dir() / "dataset_progress.json"
        self._progress = ProgressTracker(state_path)
        self._progress.load()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._run_lock = threading.Lock()

    def start(self, dataset: str, batch_size: int, language: str, *, auto_resume: bool) -> None:
        """Start background training if idle."""
        lang = validate_language_tag(language.strip())
        if lang is None:
            msg = "language must be a valid BCP47-style tag"
            raise ValueError(msg)
        key = dataset.strip().lower()
        if key not in _DATASET_PATHS:
            msg = f"unsupported dataset: {dataset}"
            raise ValueError(msg)
        if batch_size < 1 or batch_size > 512:
            msg = "batch_size must be between 1 and 512"
            raise ValueError(msg)
        json_running = getattr(self._container, "_json_dataset_running", None)
        if callable(json_running) and json_running():
            msg = "json dataset training is already running; POST /dataset/json/stop first"
            raise RuntimeError(msg)
        with self._run_lock:
            if self._thread is not None and self._thread.is_alive():
                msg = "dataset training is already running"
                raise RuntimeError(msg)
            rel = _DATASET_PATHS[key]
            parquet_path = (self._container.config.resolved_storage_root() / rel).resolve()
            importer = OpenOrcaImporter(parquet_path)
            total_rows = importer.total_rows()
            start_row = self._progress.begin_job(
                dataset=key,
                batch_size=batch_size,
                language=lang,
                total_rows=total_rows,
                auto_resume=auto_resume,
            )
            self._stop.clear()
            self._pause.clear()
            if self._progress.snapshot().paused:
                self._pause.set()
            self._thread = threading.Thread(
                target=self._worker,
                name="dataset-training",
                kwargs={
                    "importer": importer,
                    "batch_size": batch_size,
                    "language": lang,
                    "start_row": start_row,
                },
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """Request cooperative shutdown of the worker."""
        self._stop.set()
        self._pause.clear()
        if self._thread is not None:
            self._thread.join(timeout=3600)
            self._thread = None
        self._progress.mark_stopped()

    def pause(self) -> None:
        """Pause the worker between batches."""
        self._pause.set()
        self._progress.set_paused(True)

    def resume(self) -> None:
        """Resume a paused worker."""
        self._pause.clear()
        self._progress.set_paused(False)

    def progress_dict(self) -> dict[str, object]:
        """Return API-shaped progress using the live trainer step."""
        snap = self._progress.snapshot()
        return snap.to_public_dict(self._container.trainer.global_step)

    def is_running(self) -> bool:
        """Return True while the worker thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def reset_progress_tracker(self) -> None:
        """Rebind progress state from disk (use after deleting ``dataset_progress.json``)."""
        state_path = self._container.config.resolved_checkpoint_dir() / "dataset_progress.json"
        self._progress = ProgressTracker(state_path)
        self._progress.load()

    def _maybe_save_milestone(self, lifetime_batches: int) -> None:
        """Persist ``epoch_{n}.pt`` every ``_MILESTONE_BATCHES`` completed batches."""
        if lifetime_batches <= 0 or lifetime_batches % _MILESTONE_BATCHES != 0:
            return
        mile = lifetime_batches // _MILESTONE_BATCHES
        path = self._container.config.resolved_checkpoint_dir() / f"epoch_{mile}.pt"
        optimizer = getattr(self._container.trainer, "_optimizer", None)
        save_checkpoint(
            path,
            self._container.trainer.model,
            optimizer,
            global_step=self._container.trainer.global_step,
            epoch=mile,
        )
        self._progress.set_checkpoint_label(f"epoch_{mile}.pt")

    def _worker(self, importer: OpenOrcaImporter, batch_size: int, language: str, start_row: int) -> None:
        """Consume parquet batches until completion or stop signal."""
        _LOGGER.info(
            "Dataset worker started",
            extra={"extra_fields": {"start_row": start_row, "batch_size": batch_size}},
        )
        try:
            for frame in importer.iter_batches(batch_size, start_row=start_row):
                while self._pause.is_set() and not self._stop.is_set():
                    time.sleep(0.05)
                if self._stop.is_set():
                    break
                self._progress.mark_batch_start()
                row_count = len(frame)
                payloads = rows_to_payloads(
                    importer.question_column,
                    importer.response_column,
                    frame,
                    language,
                )
                if not payloads:
                    self._progress.mark_batch_end(row_count, self._progress.snapshot().last_loss)
                    snap = self._progress.snapshot()
                    self._maybe_save_milestone(snap.lifetime_batches)
                    continue
                mean_loss = train_payload_batch(self._container, payloads)
                self._progress.mark_batch_end(row_count, mean_loss)
                snap = self._progress.snapshot()
                self._maybe_save_milestone(snap.lifetime_batches)
                if self._stop.is_set():
                    break
            else:
                self._progress.mark_finished()
                return
        except Exception:
            _LOGGER.exception("Dataset worker failed")
            self._progress.mark_stopped()
            raise
        self._progress.mark_stopped()
