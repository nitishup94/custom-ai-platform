"""Validate JSON training records, persist JSON-job progress, and run a background worker."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch

from training.checkpoint import save_checkpoint
from training.dataset_scanner import json_datasets_directory, list_json_dataset_files
from training.json_importer import JsonImporter
from utils.logger import get_logger
from utils.token_window import clip_byte_token_ids
from utils.training_sample_log import training_sample_extra_fields
from utils.validators import validate_bounded_text, validate_category, validate_language_tag

if TYPE_CHECKING:
    from app.container import AppContainer

_LOGGER = get_logger("json_batch_processor")

_JSON_FIELD_MAX = 8192
_MILESTONE_BATCHES = 1000


def record_to_payload(raw: dict[str, Any], default_language: str) -> dict[str, str]:
    """Validate one record and return a payload dict for training."""
    inp = str(raw.get("input", "")).strip()
    out = str(raw.get("output", "")).strip()
    cat = str(raw.get("category", "")).strip()
    lang_raw = raw.get("language")
    lang_str = str(lang_raw).strip() if lang_raw is not None else ""
    question = validate_bounded_text(inp, "input", max_len=_JSON_FIELD_MAX)
    answer = validate_bounded_text(out, "output", max_len=_JSON_FIELD_MAX)
    category = validate_category(cat)
    language = validate_language_tag(lang_str) if lang_str else None
    if language is None:
        language = validate_language_tag(default_language) if default_language else None
    return {
        "input": question,
        "output": answer,
        "category": category,
        "language": language or "",
    }


def train_json_payload_batch(container: "AppContainer", payloads: list[dict[str, str]]) -> float:
    """Chroma + ``train_on_sample`` for JSON dataset rows; ``source`` is ``json_dataset``."""
    if not payloads:
        return 0.0
    losses: list[float] = []
    for item in payloads:
        question = validate_bounded_text(item["input"], "input", max_len=_JSON_FIELD_MAX)
        answer = validate_bounded_text(item["output"], "output", max_len=_JSON_FIELD_MAX)
        category = validate_category(item["category"])
        language = validate_language_tag((item.get("language") or "").strip())
        if language is None and container.config.default_language:
            language = validate_language_tag(container.config.default_language)

        if container.config.log_training_inputs:
            _LOGGER.info(
                "JSON dataset train row",
                extra={
                    "extra_fields": training_sample_extra_fields(
                        container.tokenizer,
                        question,
                        answer,
                    )
                },
            )

        doc = f"Question: {question}\nAnswer: {answer}"
        cap = container.config.max_seq_len
        q_ids = clip_byte_token_ids(container.tokenizer, question, cap)
        ids = torch.tensor([q_ids], dtype=torch.long, device=container.device)
        embedding = container.trainer.inference.retrieval_embedding(ids)
        vector = embedding.detach().cpu().numpy().tolist()[0]
        container.knowledge.add_document(
            embedding=vector,
            document=doc,
            category=category,
            source="json_dataset",
            language=language,
        )
        loss = container.trainer.train_on_sample(question, answer)
        losses.append(loss)
    return sum(losses) / float(len(losses))


@dataclass
class JsonProgressSnapshot:
    """Persisted JSON dataset job state."""

    current_file: str
    batch_size: int
    total_records: int
    processed_records: int
    lifetime_batches: int
    last_loss: float
    running: bool
    paused: bool
    finished: bool
    last_checkpoint_label: str = "latest.pt"

    def to_api_dict(self, global_step: int) -> dict[str, Any]:
        """Shape for ``GET /dataset/json/progress``."""
        remaining = max(self.total_records - self.processed_records, 0)
        return {
            "current_file": self.current_file,
            "processed_records": int(self.processed_records),
            "remaining_records": int(remaining),
            "global_step": int(global_step),
            "loss": float(self.last_loss),
            "checkpoint": self.last_checkpoint_label,
        }


class JsonDatasetProgress:
    """Thread-safe progress for the JSON file worker."""

    def __init__(self, state_path: Path) -> None:
        self._path = state_path
        self._lock = threading.Lock()
        self._snap = JsonProgressSnapshot(
            current_file="",
            batch_size=32,
            total_records=0,
            processed_records=0,
            lifetime_batches=0,
            last_loss=0.0,
            running=False,
            paused=False,
            finished=False,
        )
        self._last_batch_started: float | None = None

    def load(self) -> None:
        if not self._path.is_file():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _LOGGER.warning("Failed to read JSON progress", extra={"extra_fields": {"error": str(exc)}})
            return
        with self._lock:
            self._snap = JsonProgressSnapshot(
                current_file=str(raw.get("current_file", "")),
                batch_size=int(raw.get("batch_size", 32)),
                total_records=int(raw.get("total_records", 0)),
                processed_records=int(raw.get("processed_records", 0)),
                lifetime_batches=int(raw.get("lifetime_batches", 0)),
                last_loss=float(raw.get("last_loss", 0.0)),
                running=False,
                paused=bool(raw.get("paused", False)),
                finished=bool(raw.get("finished", False)),
                last_checkpoint_label=str(raw.get("last_checkpoint_label", "latest.pt")),
            )

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = asdict(self._snap)
            tmp = self._path.parent / f".{self._path.stem}.{uuid.uuid4().hex}.tmp"
            try:
                tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                tmp.replace(self._path)
            except OSError as exc:
                _LOGGER.warning(
                    "JSON progress save failed",
                    extra={"extra_fields": {"path": str(self._path), "error": str(exc)}},
                )
            finally:
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except OSError:
                        pass

    def begin_job(self, filename: str, batch_size: int, total: int, *, auto_resume: bool) -> int:
        """Return start record index (0 or resumed)."""
        with self._lock:
            can_resume = (
                auto_resume
                and self._snap.current_file == filename
                and self._snap.batch_size == batch_size
                and self._snap.processed_records > 0
                and self._snap.processed_records < total
                and not self._snap.finished
            )
            if can_resume:
                self._snap.total_records = total
                self._snap.running = True
                self._snap.paused = False
                self._snap.finished = False
                start = self._snap.processed_records
            else:
                self._snap = JsonProgressSnapshot(
                    current_file=filename,
                    batch_size=batch_size,
                    total_records=total,
                    processed_records=0,
                    lifetime_batches=0,
                    last_loss=0.0,
                    running=True,
                    paused=False,
                    finished=False,
                )
                start = 0
        self.save()
        return start

    def mark_batch_start(self) -> None:
        self._last_batch_started = time.perf_counter()

    def mark_batch_end(self, rows: int, mean_loss: float) -> None:
        if rows <= 0:
            return
        with self._lock:
            self._snap.processed_records += rows
            self._snap.lifetime_batches += 1
            self._snap.last_loss = float(mean_loss)
        self.save()

    def set_checkpoint_label(self, label: str) -> None:
        with self._lock:
            self._snap.last_checkpoint_label = label
        self.save()

    def mark_finished(self) -> None:
        with self._lock:
            self._snap.running = False
            self._snap.paused = False
            self._snap.finished = True
        self.save()

    def mark_stopped(self) -> None:
        with self._lock:
            self._snap.running = False
            self._snap.paused = False
        self.save()

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self._snap.paused = paused
        self.save()

    def snapshot(self) -> JsonProgressSnapshot:
        with self._lock:
            return JsonProgressSnapshot(**asdict(self._snap))

    def is_running(self) -> bool:
        with self._lock:
            return bool(self._snap.running and not self._snap.finished)


class JsonDatasetTrainingCoordinator:
    """Background JSON file training (same LM + Chroma path as parquet worker)."""

    def __init__(
        self,
        container: "AppContainer",
        parquet_running: Callable[[], bool],
    ) -> None:
        self._container = container
        self._parquet_running = parquet_running
        state_path = container.config.resolved_checkpoint_dir() / "json_dataset_progress.json"
        self._progress = JsonDatasetProgress(state_path)
        self._progress.load()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._run_lock = threading.Lock()

    def list_files(self) -> list[str]:
        """Basenames under ``storage/datasets/json``."""
        return list_json_dataset_files(self._container.config)

    def start(self, filename: str, batch_size: int, *, auto_resume: bool = True) -> None:
        if self._parquet_running():
            msg = "parquet dataset training is already running; POST /dataset/stop first"
            raise RuntimeError(msg)
        if batch_size < 1 or batch_size > 512:
            msg = "batch_size must be between 1 and 512"
            raise ValueError(msg)
        name = filename.strip()
        if not name or "/" in name or "\\" in name or ".." in name:
            msg = "file must be a basename inside storage/datasets/json"
            raise ValueError(msg)
        if not name.lower().endswith(".json"):
            msg = "file must end with .json"
            raise ValueError(msg)
        available = set(self.list_files())
        if name not in available:
            msg = f"unknown JSON dataset file: {name}"
            raise ValueError(msg)
        path = json_datasets_directory(self._container.config) / name
        with self._run_lock:
            if self._thread is not None and self._thread.is_alive():
                msg = "json dataset training is already running"
                raise RuntimeError(msg)
            importer = JsonImporter(path)
            total = importer.total_records()
            if total == 0:
                msg = "JSON file contains no training records"
                raise ValueError(msg)
            start_idx = self._progress.begin_job(name, batch_size, total, auto_resume=auto_resume)
            self._stop.clear()
            self._pause.clear()
            if self._progress.snapshot().paused:
                self._pause.set()
            self._thread = threading.Thread(
                target=self._worker,
                name="json-dataset-training",
                kwargs={
                    "importer": importer,
                    "batch_size": batch_size,
                    "start_index": start_idx,
                },
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._pause.clear()
        if self._thread is not None:
            self._thread.join(timeout=3600)
            self._thread = None
        self._progress.mark_stopped()

    def pause(self) -> None:
        self._pause.set()
        self._progress.set_paused(True)

    def resume(self) -> None:
        self._pause.clear()
        self._progress.set_paused(False)

    def progress_dict(self) -> dict[str, object]:
        snap = self._progress.snapshot()
        return snap.to_api_dict(self._container.trainer.global_step)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def reset_progress_tracker(self) -> None:
        state_path = self._container.config.resolved_checkpoint_dir() / "json_dataset_progress.json"
        self._progress = JsonDatasetProgress(state_path)
        self._progress.load()

    def _maybe_save_milestone(self, lifetime_batches: int) -> None:
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

    def _worker(self, importer: JsonImporter, batch_size: int, start_index: int) -> None:
        default_lang = (self._container.config.default_language or "").strip()
        _LOGGER.info(
            "JSON dataset worker started",
            extra={"extra_fields": {"file": importer.source_path.name, "start_index": start_index}},
        )
        try:
            for batch in importer.iter_record_batches(batch_size, start_index=start_index):
                while self._pause.is_set() and not self._stop.is_set():
                    time.sleep(0.05)
                if self._stop.is_set():
                    break
                self._progress.mark_batch_start()
                row_count = len(batch)
                payloads: list[dict[str, str]] = []
                for raw in batch:
                    try:
                        payloads.append(record_to_payload(raw, default_lang))
                    except ValueError as exc:
                        _LOGGER.warning(
                            "Skipping invalid JSON record",
                            extra={"extra_fields": {"error": str(exc)}},
                        )
                if not payloads:
                    self._progress.mark_batch_end(row_count, self._progress.snapshot().last_loss)
                    snap = self._progress.snapshot()
                    self._maybe_save_milestone(snap.lifetime_batches)
                    continue
                mean_loss = train_json_payload_batch(self._container, payloads)
                self._progress.mark_batch_end(row_count, mean_loss)
                snap = self._progress.snapshot()
                self._maybe_save_milestone(snap.lifetime_batches)
                if self._stop.is_set():
                    break
            else:
                self._progress.mark_finished()
                return
        except Exception:
            _LOGGER.exception("JSON dataset worker failed")
            self._progress.mark_stopped()
            raise
        self._progress.mark_stopped()
