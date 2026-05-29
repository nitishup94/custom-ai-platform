"""Background dataset training control API."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from training.training_queue import DatasetTrainingCoordinator

router = APIRouter(prefix="/dataset", tags=["dataset"])


def get_dataset_coordinator(request: Request) -> DatasetTrainingCoordinator:
    """Return the lifespan-initialized dataset worker."""
    coordinator = getattr(request.app.state, "dataset_coordinator", None)
    if coordinator is None:
        raise HTTPException(status_code=503, detail="dataset coordinator is not initialized")
    return coordinator  # type: ignore[no-any-return]


class DatasetStartRequest(BaseModel):
    """Start OpenOrca-style parquet training on a background thread."""

    dataset: str = Field(..., min_length=1)
    batch_size: int = Field(32, ge=1, le=512)
    language: str = Field(..., min_length=1)
    resume: bool = Field(
        True,
        description="Resume from saved cursor when dataset, batch_size, and language match the saved job.",
    )


class DatasetStartResponse(BaseModel):
    """Acknowledgement that the worker was scheduled."""

    status: str = "started"


class DatasetStopResponse(BaseModel):
    """Acknowledgement that stop was requested."""

    status: str = "stopped"


@router.post("/start", response_model=DatasetStartResponse)
def dataset_start(
    body: DatasetStartRequest,
    coordinator: Annotated[DatasetTrainingCoordinator, Depends(get_dataset_coordinator)],
) -> DatasetStartResponse:
    """Begin or resume streaming training from the configured parquet path."""
    try:
        coordinator.start(
            body.dataset,
            body.batch_size,
            body.language,
            auto_resume=body.resume,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return DatasetStartResponse()


@router.get("/progress")
def dataset_progress(
    coordinator: Annotated[DatasetTrainingCoordinator, Depends(get_dataset_coordinator)],
) -> dict[str, object]:
    """Return row counts, trainer step, loss, checkpoint label, and ETA."""
    return coordinator.progress_dict()


@router.post("/stop", response_model=DatasetStopResponse)
def dataset_stop(
    coordinator: Annotated[DatasetTrainingCoordinator, Depends(get_dataset_coordinator)],
) -> DatasetStopResponse:
    """Cooperatively stop the dataset worker after the current batch."""
    coordinator.stop()
    return DatasetStopResponse()
