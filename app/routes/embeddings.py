"""OpenAI-compatible embeddings endpoint."""

from __future__ import annotations

import torch
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.container import AppContainer, get_container
from utils.validators import validate_bounded_text

router = APIRouter(tags=["embeddings"])


class EmbeddingsRequest(BaseModel):
    """Single text embedding request."""

    text: str = Field(..., min_length=1)


class EmbeddingData(BaseModel):
    """One embedding vector payload."""

    object: str
    embedding: list[float]
    index: int


class EmbeddingsResponse(BaseModel):
    """OpenAI-shaped embedding response."""

    object: str
    data: list[EmbeddingData]
    model: str


@router.post("/embeddings", response_model=EmbeddingsResponse)
def create_embeddings(
    body: EmbeddingsRequest,
    container: AppContainer = Depends(get_container),
) -> EmbeddingsResponse:
    """Return pooled, L2-normalized embedding vector."""
    try:
        text = validate_bounded_text(body.text, "text")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    ids = torch.tensor(
        [container.tokenizer.encode(text, add_special_tokens=False)],
        dtype=torch.long,
        device=container.device,
    )
    vector = container.trainer.inference.retrieval_embedding(ids)
    values = vector.detach().cpu().numpy().tolist()[0]
    return EmbeddingsResponse(
        object="list",
        data=[
            EmbeddingData(
                object="embedding",
                embedding=values,
                index=0,
            )
        ],
        model=container.config.model_name,
    )
