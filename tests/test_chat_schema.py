"""Chat completion schema smoke test."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app


def test_chat_completions_shape(tiny_model_env: None) -> None:
    """POST /v1/chat/completions returns OpenAI-like payload keys."""
    with TestClient(create_app()) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 8,
                "temperature": 0.9,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "chat.completion"
        assert "choices" in body and body["choices"]
        choice = body["choices"][0]
        assert choice["message"]["role"] == "assistant"
        assert isinstance(choice["message"]["content"], str)
        assert choice.get("finish_reason") in ("stop", "length")
        assert "usage" in body
