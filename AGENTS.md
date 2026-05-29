# AGENTS.md — Custom AI Platform (`ai-project/`)

This document orients coding agents and maintainers: what exists, where it lives, how APIs behave, and conventions to follow when extending the codebase.

---

## Project intent

Self-hosted AI stack on **Ubuntu** / **Python 3.12+** (Dockerfile uses 3.12; local may use 3.10+):

- **No** external LLM APIs, Ollama, HuggingFace pretrained weights, LangChain, or third-party checkpoints.
- **Custom** PyTorch transformer (random init), **byte-level UTF-8 tokenizer** + `storage/vocab/vocab.json`.
- **ChromaDB 0.5.18** persistent vector store for RAG-style context.
- **OpenAI-shaped** `POST /v1/chat/completions` and `POST /v1/embeddings`.
- **Training**: `POST /train` stores Q/A in Chroma and runs **one** LM optimizer step per request; checkpoints under `storage/checkpoints/`.

---

## HTTP API (implemented)

| Method | Path | Module | Summary |
|--------|------|--------|---------|
| `GET` | `/health` | [`app/routes/health.py`](app/routes/health.py) | `{"status":"ok"}` |
| `GET` | `/models` | [`app/routes/models.py`](app/routes/models.py) | `{"models":[{"name": ...}]}` from `MODEL_NAME` / config |
| `POST` | `/train` | [`app/routes/train.py`](app/routes/train.py) | Body: `input`, `output`, `category`; optional `language`. Chroma doc `Question: …\nAnswer: …`; metadata `category`, `timestamp`, `source`, optional `language`. If `language` omitted, uses env **`DEFAULT_LANGUAGE`** (see [`utils/config.py`](utils/config.py)). Optional **`LOG_TRAINING_SAMPLES`** logs instruction/response token counts and preview per request. |
| `POST` | `/train/reset-storage` | [`app/routes/train.py`](app/routes/train.py) | Optional JSON: `chromadb`, `checkpoints`, `dataset_progress` (default `true`). Clears vectors, removes checkpoint files, removes **both** `dataset_progress.json` and `json_dataset_progress.json` when `dataset_progress` is true, resets parquet + JSON in-memory trackers. **409** if either dataset worker is running. |
| `POST` | `/dataset/start` | [`app/routes/dataset.py`](app/routes/dataset.py) | Body: `dataset` (e.g. `openorca`), `batch_size`, `language`; optional `resume` (default `true`). Streams `storage/datasets/1M-GPT4-Augmented.parquet` on a worker thread; same Chroma + LM step as `/train` per row (see [`training/training_queue.py`](training/training_queue.py)). |
| `GET` | `/dataset/progress` | [`app/routes/dataset.py`](app/routes/dataset.py) | `processed_records`, `remaining_records`, `global_step`, `loss`, `checkpoint`, `estimated_time_remaining`. |
| `POST` | `/dataset/stop` | [`app/routes/dataset.py`](app/routes/dataset.py) | Cooperative shutdown between batches. |
| `GET` | `/dataset/json/list` | [`app/routes/dataset_json.py`](app/routes/dataset_json.py) | `{"files":["…"]}` basenames under `storage/datasets/json`. |
| `POST` | `/dataset/json/start` | [`app/routes/dataset_json.py`](app/routes/dataset_json.py) | Body: `file` (basename), `batch_size`, optional `resume`. JSON/JSONL records matching `/train` fields. **409** if parquet worker running or JSON worker already running. |
| `GET` | `/dataset/json/progress` | [`app/routes/dataset_json.py`](app/routes/dataset_json.py) | `current_file`, row counts, `global_step`, `loss`, `checkpoint`. |
| `POST` | `/dataset/json/stop` | [`app/routes/dataset_json.py`](app/routes/dataset_json.py) | Stop JSON worker. |
| `POST` | `/v1/chat/completions` | [`app/routes/chat.py`](app/routes/chat.py) | OpenAI subset: `messages`, `max_tokens` (1–20000, **also capped** by `MAX_SEQ_LEN` minus prompt), optional `temperature`, optional `language` (Chroma `where`). `finish_reason`: `stop` (EOS) or `length` (budget / window). |
| `POST` | `/v1/embeddings` | [`app/routes/embeddings.py`](app/routes/embeddings.py) | Body: `{ "text": "..." }`; returns pooled L2-normalized vector (dim = `d_model`). |

**App wiring**: [`app/main.py`](app/main.py) mounts routers; chat/embeddings use prefix `/v1`. Lifespan sets `app.state.dataset_coordinator` ([`training/training_queue.py`](training/training_queue.py)) and `app.state.json_dataset_coordinator` ([`training/json_batch_processor.py`](training/json_batch_processor.py)). **DI**: [`app/container.py`](app/container.py) + `get_container(request)` ([`app/container.py`](app/container.py)).

**Reference docs for humans**: [`guide.txt`](guide.txt) (curl + payloads), [`postman/Custom-AI-Platform.postman_collection.json`](postman/Custom-AI-Platform.postman_collection.json), [`README.md`](README.md).

---

## Configuration and environment

- **Central settings**: [`utils/config.py`](utils/config.py) — `AppConfig` (`pydantic-settings`), loads **`.env`** from the working directory.
- **Template**: [`.env.example`](.env.example) — copy to `.env` and edit (`cp .env.example .env`).
- **Notable keys**: `STORAGE_ROOT`, `CHROMA_PATH`, `CHECKPOINT_DIR`, `VOCAB_PATH`, `DEVICE` (`cpu` / `cuda` / `auto`), transformer dims (`D_MODEL`, `N_HEADS`, …), `DEFAULT_LANGUAGE` (default `en`; empty string disables auto-tag on `/train` when body omits `language`), `CORS_*`, `LOG_LEVEL`, etc.

---

## Repository tree (source and ops; exclude `.venv`, `__pycache__`, `.pytest_cache`)

```
ai-project/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI factory, lifespan, CORS, exception handlers
│   ├── container.py            # Bootstrap: config, tokenizer, Chroma, model, trainer; get_container()
│   └── routes/
│       ├── __init__.py
│       ├── health.py
│       ├── models.py
│       ├── train.py
│       ├── dataset.py
│       ├── dataset_json.py
│       ├── chat.py
│       └── embeddings.py
├── model/
│   ├── __init__.py
│   ├── layers.py               # LayerNorm, FeedForward
│   ├── attention.py            # MHA + causal mask
│   ├── embedding.py            # Token + position embeddings
│   ├── transformer.py          # TransformerLM
│   ├── tokenizer.py            # ByteTokenizer + vocab.json
│   └── inference.py            # retrieval_embedding, generate()
├── training/
│   ├── __init__.py
│   ├── dataset.py              # LM sequence packing
│   ├── openorca_importer.py    # PyArrow ParquetFile.iter_batches
│   ├── dataset_scanner.py      # list *.json under storage/datasets/json
│   ├── json_importer.py        # JSON array or JSONL → records
│   ├── json_batch_processor.py # validate JSON rows, progress, JSON worker
│   ├── batch_processor.py      # rows → payloads; Chroma + train_on_sample
│   ├── category_detector.py    # heuristic category from Q/A text
│   ├── training_queue.py       # background coordinator + milestones
│   ├── progress_tracker.py     # JSON progress + ETA
│   ├── optimizer.py
│   ├── scheduler.py
│   ├── checkpoint.py           # latest.pt, epoch_n.pt
│   └── trainer.py
├── vectorstore/
│   ├── __init__.py
│   ├── chroma_client.py        # PersistentClient + Settings (telemetry noop, anonymized off)
│   ├── chroma_telemetry_noop.py # NoOpProductTelemetry for Chroma
│   └── collections.py          # KnowledgeCollection: add_document, query (+ optional language where)
├── utils/
│   ├── __init__.py
│   ├── config.py               # AppConfig + default_language validator
│   ├── logger.py               # JSON logs
│   └── validators.py         # text, category, language_tag
├── storage/
│   ├── checkpoints/.gitkeep
│   ├── chroma_data/.gitkeep
│   ├── datasets/json/.gitkeep  # place greetings.json, etc.
│   └── vocab/vocab.json        # Starter byte vocab schema
├── tests/
│   ├── conftest.py             # tiny_model_env fixture
│   ├── test_health.py
│   ├── test_train_smoke.py
│   ├── test_chat_schema.py
│   ├── test_tokenizer_roundtrip.py
│   ├── test_validators.py
│   ├── test_config.py
│   └── test_json_dataset.py
├── postman/
│   └── Custom-AI-Platform.postman_collection.json
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── run.sh
├── pytest.ini
├── .env.example
├── .gitignore
├── .dockerignore
├── guide.txt
├── README.md
└── AGENTS.md                   # (this file)
```

---

## Data and control flows (short)

**Train** (`/train`): validate → embed **question** only → Chroma `add` → LM step → save `latest.pt` (+ periodic `epoch_n.pt` per `CHECKPOINT_EVERY_STEPS`).

**Dataset** (`/dataset/start`): PyArrow lazy batches from OpenOrca parquet → map columns to Q/A → category heuristics → for each row: Chroma `add_document` (`source=openorca_dataset`) + `trainer.train_on_sample` (no HTTP to `/train`). Progress persisted next to checkpoints as `dataset_progress.json`; extra `epoch_{k}.pt` every 1000 completed parquet batches via [`save_checkpoint`](training/checkpoint.py).

**Dataset (JSON)** (`/dataset/json/start`): Scan `storage/datasets/json/*.json` → parse array or JSONL → validate `input`/`output`/`category` (8192-char cap) → Chroma `add_document` (`source=json_dataset`) + `train_on_sample`. Progress file `json_dataset_progress.json`; milestones every 1000 batches. Mutually exclusive with parquet dataset worker.

**Chat** (`/v1/chat/completions`): last user text → same retrieval embedding → Chroma `query` (optional `where` on `language`) → build context string → `build_lm_sequence` with empty assistant → strip trailing `eos` → `generate()`.

**Embeddings**: same pooled pre-transformer embedding as retrieval; dimension **`d_model`**.

---

## Engineering conventions (when you change code)

1. **Paths**: relative paths resolve under `STORAGE_ROOT` via [`AppConfig.resolved_*`](utils/config.py).
2. **Types**: prefer type hints and docstrings on public methods.
3. **Logging**: use [`utils/logger.py`](utils/logger.py); JSON to stdout and/or append-only file via **`LOG_CONSOLE_ENABLED`** / **`LOG_FILE_PATH`** (see [`.env.example`](.env.example)); optional **`[logging] …`** line on stderr from **`LOG_STARTUP_SUMMARY`**. Avoid logging secrets.
4. **Errors**: routes use `HTTPException(422, …)` for validation; generic handler in [`app/main.py`](app/main.py) hides stack traces from clients (`HTTPException` is re-raised).
5. **Chroma**: do not re-enable broken PostHog telemetry without pinning `posthog`; keep [`vectorstore/chroma_telemetry_noop.py`](vectorstore/chroma_telemetry_noop.py) wired from [`chroma_client.py`](vectorstore/chroma_client.py).
6. **Multilingual**: tokenizer is UTF-8 bytes; optional `language` on train/chat + `DEFAULT_LANGUAGE` in env — see README “Languages” section.
7. **Tests**: from `ai-project/`, `pytest -q`. Integration tests shrink the model via `tiny_model_env` in [`tests/conftest.py`](tests/conftest.py).

---

## Commands

```bash
cd ai-project
cp -n .env.example .env   # once
source .venv/bin/activate
pip install -r requirements.txt
./run.sh                  # or: uvicorn app.main:app --host 0.0.0.0 --port 8000
pytest -q
docker compose up --build
```

**PYTHONPATH**: run from `ai-project` root (see [`run.sh`](run.sh), [`pytest.ini`](pytest.ini)).

---

## What not to add (product constraints)

- No OpenAI/Anthropic/Gemini client calls for inference.
- No loading external transformer checkpoints or HF `transformers` model weights for the core LM.
- Keep **ChromaDB** pinned to **`chromadb==0.5.18`** unless you intentionally upgrade and re-test telemetry + APIs.

---

## Related files for common tasks

| Task | Start here |
|------|------------|
| New HTTP route | [`app/main.py`](app/main.py) + new module under [`app/routes/`](app/routes/), register router; use `Depends(get_container)`. |
| Model shape / device | [`utils/config.py`](utils/config.py), [`app/container.py`](app/container.py), [`model/transformer.py`](model/transformer.py) |
| Training step / loss | [`training/trainer.py`](training/trainer.py), [`training/dataset.py`](training/dataset.py) |
| Chroma schema / filters | [`vectorstore/collections.py`](vectorstore/collections.py) |
| Tokenizer / vocab | [`model/tokenizer.py`](model/tokenizer.py), [`storage/vocab/vocab.json`](storage/vocab/vocab.json) |
| API examples | [`guide.txt`](guide.txt), Postman collection |

---

*Last aligned with the implemented tree and behavior in this repository; update this file when you add routes, env vars, or structural changes.*
