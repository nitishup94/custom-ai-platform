"""High-level knowledge collection operations."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from utils.config import AppConfig
from utils.logger import get_logger

_LOGGER = get_logger("vectorstore")

_LOG_DOC_PREVIEW_CHARS = 400


def _document_log_fields(document: str) -> dict[str, Any]:
    """Short preview for JSON logs (avoid huge payloads)."""
    n = len(document)
    if n <= _LOG_DOC_PREVIEW_CHARS:
        preview = document
    else:
        preview = document[:_LOG_DOC_PREVIEW_CHARS] + "..."
    return {"document_chars": n, "document_preview": preview}


class KnowledgeCollection:
    """Thin wrapper around a Chroma collection with explicit metadata."""

    def __init__(self, client: ClientAPI, config: AppConfig) -> None:
        """Open or create the backing collection."""
        self._collection: Collection = client.get_or_create_collection(
            name=config.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_document(
        self,
        embedding: Sequence[float],
        document: str,
        category: str,
        source: str,
        language: str | None = None,
    ) -> str:
        """Store a document with its vector and metadata. Returns the assigned id."""
        doc_id = str(uuid.uuid4())
        metadata: dict[str, Any] = {
            "category": category,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "source": source,
        }
        if language is not None:
            metadata["language"] = language
        vector = np.array(embedding, dtype=np.float32)
        self._collection.add(
            ids=[doc_id],
            embeddings=[vector.tolist()],
            documents=[document],
            metadatas=[metadata],
        )
        _LOGGER.info(
            "Stored document in Chroma",
            extra={
                "extra_fields": {
                    "id": doc_id,
                    "source": source,
                    "category": category,
                    **({"language": language} if language is not None else {}),
                    **_document_log_fields(document),
                }
            },
        )
        return doc_id

    def clear_all_documents(self) -> int:
        """Delete every embedding in this collection (batched). Returns rows removed."""
        total = 0
        batch_size = 5000
        while True:
            result = self._collection.get(limit=batch_size, offset=0, include=[])
            ids = result.get("ids") or []
            if not ids:
                break
            self._collection.delete(ids=ids)
            total += len(ids)
            if len(ids) < batch_size:
                break
        _LOGGER.info("Cleared Chroma collection", extra={"extra_fields": {"removed": total}})
        return total

    def query(
        self,
        embedding: Sequence[float],
        top_k: int,
        language: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return ranked documents similar to *embedding*.

        When *language* is set, only entries whose metadata ``language`` matches
        are considered (omit for mixed or legacy documents without a tag).
        """
        total = int(self._collection.count())
        if total <= 0:
            return []
        effective_k = min(int(top_k), total)
        if effective_k < 1:
            return []
        vector = np.array(embedding, dtype=np.float32)
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [vector.tolist()],
            "n_results": effective_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if language is not None:
            query_kwargs["where"] = {"language": language}
        result = self._collection.query(**query_kwargs)
        documents = (result.get("documents") or [[]])[0] or []
        metadatas = (result.get("metadatas") or [[]])[0] or []
        distances = (result.get("distances") or [[]])[0] or []
        out: list[dict[str, Any]] = []
        for doc, meta, dist in zip(documents, metadatas, distances, strict=False):
            out.append({"document": doc, "metadata": meta or {}, "distance": dist})
        return out
