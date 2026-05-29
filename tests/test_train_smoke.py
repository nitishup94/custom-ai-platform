"""Training endpoint smoke test."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app


def test_train_endpoint_runs(tiny_model_env: None) -> None:
    """POST /train performs a step and returns loss."""
    with TestClient(create_app()) as client:
        payload = {
            "input": "What is PHP?",
            "output": "PHP is a server-side scripting language.",
            "category": "programming",
        }
        response = client.post("/train", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert "loss" in body
        assert body["global_step"] >= 1


def test_train_reset_storage(tiny_model_env: None) -> None:
    """POST /train/reset-storage clears checkpoints and resets global_step on next train."""
    with TestClient(create_app()) as client:
        payload = {
            "input": "Ping",
            "output": "Pong",
            "category": "general",
        }
        assert client.post("/train", json=payload).status_code == 200
        reset = client.post("/train/reset-storage", json={})
        assert reset.status_code == 200
        body = reset.json()
        assert body["status"] == "ok"
        assert isinstance(body["chromadb_documents_deleted"], int)
        again = client.post("/train", json=payload)
        assert again.status_code == 200
        assert again.json()["global_step"] == 1
