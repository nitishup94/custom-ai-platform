"""JSON file dataset training control API."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from training.json_batch_processor import JsonDatasetTrainingCoordinator

router = APIRouter(prefix="/dataset/json", tags=["dataset-json"])


def get_json_dataset_coordinator(request: Request) -> JsonDatasetTrainingCoordinator:
    coordinator = getattr(request.app.state, "json_dataset_coordinator", None)
    if coordinator is None:
        raise HTTPException(status_code=503, detail="json dataset coordinator is not initialized")
    return coordinator  # type: ignore[no-any-return]


class JsonDatasetListResponse(BaseModel):
    """Files discovered under ``storage/datasets/json``."""

    files: list[str]


class JsonDatasetStartRequest(BaseModel):
    """Start training from one JSON file."""

    file: str = Field(..., min_length=1, description="Basename only, e.g. greetings.json")
    batch_size: int = Field(32, ge=1, le=512)
    resume: bool = Field(
        True,
        description="Resume from json_dataset_progress.json when file and batch_size match.",
    )


class JsonDatasetStartResponse(BaseModel):
    status: str = "started"


class JsonDatasetStopResponse(BaseModel):
    status: str = "stopped"


@router.get("/list", response_model=JsonDatasetListResponse)
def json_dataset_list(
    coordinator: Annotated[JsonDatasetTrainingCoordinator, Depends(get_json_dataset_coordinator)],
) -> JsonDatasetListResponse:
    """Return sorted ``*.json`` basenames in ``storage/datasets/json``."""
    return JsonDatasetListResponse(files=coordinator.list_files())


@router.post("/start", response_model=JsonDatasetStartResponse)
def json_dataset_start(
    body: JsonDatasetStartRequest,
    coordinator: Annotated[JsonDatasetTrainingCoordinator, Depends(get_json_dataset_coordinator)],
) -> JsonDatasetStartResponse:
    """Begin or resume JSON/JSONL training on a background thread."""
    try:
        coordinator.start(body.file, body.batch_size, auto_resume=body.resume)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JsonDatasetStartResponse()


@router.get("/progress")
def json_dataset_progress(
    coordinator: Annotated[JsonDatasetTrainingCoordinator, Depends(get_json_dataset_coordinator)],
) -> dict[str, object]:
    """Return current file, row counts, trainer step, loss, and checkpoint label."""
    return coordinator.progress_dict()


@router.post("/stop", response_model=JsonDatasetStopResponse)
def json_dataset_stop(
    coordinator: Annotated[JsonDatasetTrainingCoordinator, Depends(get_json_dataset_coordinator)],
) -> JsonDatasetStopResponse:
    """Cooperatively stop the JSON dataset worker after the current batch."""
    coordinator.stop()
    return JsonDatasetStopResponse()
