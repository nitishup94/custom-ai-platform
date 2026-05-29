"""Model listing endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.container import AppContainer, get_container

router = APIRouter(tags=["models"])


@router.get("/models")
def list_models(container: AppContainer = Depends(get_container)) -> dict[str, list[dict[str, str]]]:
    """Return static model identifiers from configuration."""
    return {"models": [{"name": container.config.model_name}]}
