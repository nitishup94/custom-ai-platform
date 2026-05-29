"""Training API route."""

from __future__ import annotations

import torch
from fastapi import APIRouter, Body, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.container import AppContainer, get_container
from training.storage_wipe import remove_model_checkpoints
from training.training_queue import DatasetTrainingCoordinator
from utils.logger import get_logger
from utils.token_window import clip_byte_token_ids
from utils.training_sample_log import training_sample_extra_fields
from utils.validators import validate_bounded_text, validate_category, validate_language_tag

router = APIRouter(tags=["train"])
_LOG = get_logger("train_route")


def _dataset_coordinator(request: Request) -> DatasetTrainingCoordinator | None:
    return getattr(request.app.state, "dataset_coordinator", None)


def _json_dataset_coordinator(request: Request):
    return getattr(request.app.state, "json_dataset_coordinator", None)


class TrainRequest(BaseModel):
    """Payload for supervised fine-tuning on a single pair."""

    input: str = Field(..., min_length=1)
    output: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    language: str | None = Field(
        default=None,
        description="Optional BCP47-style tag, e.g. en, hi, hi-IN. Stored in Chroma for scoped retrieval.",
    )


class TrainResponse(BaseModel):
    """Acknowledgement with loss scalar."""

    loss: float
    global_step: int


class TrainStorageResetBody(BaseModel):
    """Select which persisted training artifacts to remove."""

    chromadb: bool = Field(default=True, description="Delete all vectors/documents in the knowledge collection.")
    checkpoints: bool = Field(default=True, description="Remove latest.pt, epoch_*.pt, and *.tmp under checkpoint dir.")
    dataset_progress: bool = Field(
        default=True,
        description="Remove parquet (dataset_progress.json) and JSON (json_dataset_progress.json) job cursors; reset in-memory coordinators.",
    )


class TrainStorageResetResponse(BaseModel):
    """Summary of a storage reset operation."""

    status: str = "ok"
    chromadb_documents_deleted: int
    checkpoint_files_removed: list[str]
    dataset_progress_cleared: bool
    note: str


@router.post("/train/reset-storage", response_model=TrainStorageResetResponse)
def reset_training_storage(
    request: Request,
    container: AppContainer = Depends(get_container),
    body: TrainStorageResetBody | None = Body(default=None),
) -> TrainStorageResetResponse:
    """Clear Chroma knowledge, checkpoint files, and/or dataset job cursors (idle workers only)."""
    opts = body or TrainStorageResetBody()
    coord = _dataset_coordinator(request)
    jcoord = _json_dataset_coordinator(request)
    if coord is not None and coord.is_running():
        raise HTTPException(
            status_code=409,
            detail="dataset training worker is running; POST /dataset/stop first",
        )
    if jcoord is not None and jcoord.is_running():
        raise HTTPException(
            status_code=409,
            detail="json dataset training worker is running; POST /dataset/json/stop first",
        )

    chromadb_n = 0
    if opts.chromadb:
        chromadb_n = container.knowledge.clear_all_documents()

    ckpt_dir = container.config.resolved_checkpoint_dir()
    removed: list[str] = []
    if opts.checkpoints:
        removed.extend(remove_model_checkpoints(ckpt_dir))

    progress_cleared = False
    if opts.dataset_progress:
        progress_path = ckpt_dir / "dataset_progress.json"
        if progress_path.is_file():
            progress_path.unlink()
            removed.append("dataset_progress.json")
        json_progress_path = ckpt_dir / "json_dataset_progress.json"
        if json_progress_path.is_file():
            json_progress_path.unlink()
            removed.append("json_dataset_progress.json")
        progress_cleared = True
        if coord is not None:
            coord.reset_progress_tracker()
        if jcoord is not None:
            jcoord.reset_progress_tracker()

    if opts.checkpoints:
        container.trainer.reset_runtime_after_storage_wipe()

    note = (
        "Checkpoint files removed from disk where requested; trainer step counter and optimizer were reset. "
        "When dataset_progress was cleared, both parquet and JSON job state files were removed. "
        "Model parameters in RAM are unchanged until you restart the service (startup then loads from disk or fresh init)."
    )
    return TrainStorageResetResponse(
        chromadb_documents_deleted=chromadb_n,
        checkpoint_files_removed=sorted(set(removed)),
        dataset_progress_cleared=progress_cleared,
        note=note,
    )


@router.post("/train", response_model=TrainResponse)
def train_sample(
    body: TrainRequest,
    container: AppContainer = Depends(get_container),
) -> TrainResponse:
    """Append knowledge, then run one LM training step."""
    try:
        question = validate_bounded_text(
            body.input,
            "input",
            container.config.max_train_field_chars,
        )
        answer = validate_bounded_text(
            body.output,
            "output",
            container.config.max_train_field_chars,
        )
        category = validate_category(body.category)
        language = validate_language_tag(body.language)
        if language is None and container.config.default_language:
            language = validate_language_tag(container.config.default_language)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if container.config.log_training_inputs:
        _LOG.info(
            "Train sample",
            extra={"extra_fields": training_sample_extra_fields(container.tokenizer, question, answer)},
        )

    doc = f"Question: {question}\nAnswer: {answer}"
    q_ids = clip_byte_token_ids(container.tokenizer, question, container.config.max_seq_len)
    ids = torch.tensor([q_ids], dtype=torch.long, device=container.device)
    embedding = container.trainer.inference.retrieval_embedding(ids)
    vector = embedding.detach().cpu().numpy().tolist()[0]

    container.knowledge.add_document(
        embedding=vector,
        document=doc,
        category=category,
        source="train_api",
        language=language,
    )

    loss = container.trainer.train_on_sample(question, answer)
    return TrainResponse(loss=loss, global_step=container.trainer.global_step)
