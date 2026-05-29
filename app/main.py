"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.container import AppContainer
from app.routes import chat, dataset, dataset_json, embeddings, health, models, train
from training.json_batch_processor import JsonDatasetTrainingCoordinator
from training.training_queue import DatasetTrainingCoordinator
from utils.config import AppConfig
from utils.logger import get_logger, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup and shutdown hooks."""
    config = AppConfig()
    setup_logging(config)
    log = get_logger("main")
    log.info("Starting service", extra={"extra_fields": {"app": config.app_name}})
    container = AppContainer.bootstrap(config)
    app.state.container = container
    parquet_coordinator = DatasetTrainingCoordinator(container)
    json_coordinator = JsonDatasetTrainingCoordinator(container, parquet_running=parquet_coordinator.is_running)
    setattr(container, "_json_dataset_running", json_coordinator.is_running)
    app.state.dataset_coordinator = parquet_coordinator
    app.state.json_dataset_coordinator = json_coordinator

    low = 5
    high = container.tokenizer.byte_token_high_exclusive()
    if high <= low + 1:
        high = min(low + 2, container.config.vocab_size)
    dummy = torch.randint(
        low=low,
        high=high,
        size=(1, 8),
        dtype=torch.long,
        device=container.device,
    )
    container.trainer.model.eval()
    with torch.inference_mode():
        _ = container.trainer.model(dummy)
    log.info("Warmup forward complete")

    yield

    log.info("Shutting down service")
    logging.shutdown()


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(title="Custom AI Platform", lifespan=lifespan)

    config = AppConfig()
    if config.cors_enabled:
        origins = [o.strip() for o in config.cors_allow_origins.split(",") if o.strip()]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins or ["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(train.router)
    app.include_router(dataset.router)
    app.include_router(dataset_json.router)
    app.include_router(chat.router, prefix="/v1")
    app.include_router(embeddings.router, prefix="/v1")

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Normalize validation errors."""
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        """Hide stack traces from clients."""
        if isinstance(exc, HTTPException):
            raise exc
        log = get_logger("main")
        log.exception("Unhandled error", exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": "internal server error"})

    return app


app = create_app()
