"""JSON dataset scanner and importer smoke tests."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from training.dataset_scanner import list_json_dataset_files
from training.json_importer import JsonImporter


def test_json_importer_array_and_jsonl(tmp_path) -> None:
    """JsonImporter parses array and JSONL."""
    p1 = tmp_path / "a.json"
    p1.write_text(
        json.dumps(
            [
                {"input": " hi ", "output": " there ", "category": "general", "language": "en"},
            ]
        ),
        encoding="utf-8",
    )
    j1 = JsonImporter(p1)
    assert j1.total_records() == 1
    batches = list(j1.iter_record_batches(10, start_index=0))
    assert len(batches) == 1
    assert batches[0][0]["input"] == " hi "

    p2 = tmp_path / "b.json"
    p2.write_text(
        '{"input":"a","output":"b","category":"general"}\n{"input":"c","output":"d","category":"math"}\n',
        encoding="utf-8",
    )
    j2 = JsonImporter(p2)
    assert j2.total_records() == 2


def test_json_dataset_list_endpoint(tiny_model_env: None, tmp_path_factory, monkeypatch: pytest.MonkeyPatch) -> None:
    """GET /dataset/json/list returns basenames from storage/datasets/json."""
    root = tmp_path_factory.mktemp("jr")
    js = root / "storage" / "datasets" / "json"
    js.mkdir(parents=True)
    (js / "greetings.json").write_text("[]", encoding="utf-8")
    (js / "readme.txt").write_text("x", encoding="utf-8")
    monkeypatch.setenv("STORAGE_ROOT", str(root))

    from utils.config import AppConfig

    cfg = AppConfig()
    assert list_json_dataset_files(cfg) == ["greetings.json"]

    with TestClient(create_app()) as client:
        r = client.get("/dataset/json/list")
        assert r.status_code == 200
        assert r.json() == {"files": ["greetings.json"]}
