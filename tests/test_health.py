"""Health endpoint smoke test."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app


def test_health_ok(tiny_model_env: None) -> None:
    """GET /health returns ok."""
    with TestClient(create_app()) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
