"""Chroma persistent client factory."""

from __future__ import annotations

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings

from utils.config import AppConfig


def build_chroma_client(config: AppConfig) -> ClientAPI:
    """Create a persistent Chroma client rooted at *config* path."""
    config.resolved_chroma_path().mkdir(parents=True, exist_ok=True)
    settings = Settings(
        anonymized_telemetry=False,
        chroma_product_telemetry_impl="vectorstore.chroma_telemetry_noop.NoOpProductTelemetry",
        chroma_telemetry_impl="vectorstore.chroma_telemetry_noop.NoOpProductTelemetry",
    )
    return chromadb.PersistentClient(path=str(config.resolved_chroma_path()), settings=settings)
