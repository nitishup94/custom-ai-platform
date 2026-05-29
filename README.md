# Custom self-hosted AI platform

Fully local FastAPI service with a **custom PyTorch transformer** (no external LLM weights), **ChromaDB 0.5.18** for vector memory, **byte-level tokenizer** with JSON vocabulary, **OpenAI-shaped** chat and embeddings routes, and a **training** endpoint that performs one optimizer step per call and persists checkpoints.

**For AI agents / maintainers:** see [`AGENTS.md`](AGENTS.md) (APIs, tree, env, flows, conventions).

## Features

- **No third-party LLM APIs** or pretrained checkpoints.
- **RAG-style chat**: pooled embeddings query Chroma, context is prepended, then the transformer generates tokens.
- **Training**: each `POST /train` stores the Q/A in Chroma and runs backpropagation on a causal LM objective.
- **Docker Compose** with persistent volumes for Chroma data, checkpoints, and vocabulary.

## Languages (English, Hindi, multilingual)

- The **tokenizer is UTF-8 byte-level**: any script encodable in UTF-8—including **English**, **Hindi (Devanagari)**, and mixed text—is supported without swapping the model file. There is no English-only hard limit in the stack.
- **Tradeoff**: Indic text usually needs **more byte-tokens per character** than Latin, so effective context is shorter unless you raise `MAX_SEQ_LEN` or train with more data.
- **Scalable routing**: optional **`language`** on `POST /train` (stored in Chroma metadata) and on **`POST /v1/chat/completions`** (retrieval `where` filter). Use tags like `en`, `hi`, or `hi-in` so you can index Hindi Q/A now and query Hindi-only context later without schema changes.

## Requirements

- Ubuntu (or any Linux) with Python **3.12+**
- Optional: NVIDIA GPU + CUDA-enabled PyTorch wheel (default image is CPU)

## Installation (virtual environment)

The first line of [`requirements.txt`](requirements.txt) points `pip` at the **PyTorch CPU wheel index** for faster, smaller installs. For CUDA, replace that line with the [official CUDA wheel index](https://pytorch.org/get-started/locally/) and a matching `torch` build.

```bash
cd ai-project
cp .env.example .env
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

Environment variables map to [`utils/config.py`](utils/config.py) fields (uppercase names). Common overrides:

| Variable | Purpose |
|----------|---------|
| `STORAGE_ROOT` | Base directory for relative storage paths (default `.`) |
| `CHROMA_PATH` | Chroma persistence directory |
| `CHECKPOINT_DIR` | Model checkpoints |
| `VOCAB_PATH` | `vocab.json` location |
| `DEVICE` | `cpu`, `cuda`, or `auto` |
| `D_MODEL`, `N_HEADS`, `N_LAYERS`, `D_FF`, `MAX_SEQ_LEN`, `VOCAB_SIZE` | Model capacity |
| `CHECKPOINT_EVERY_STEPS` | Interval for `epoch_n.pt` snapshots |
| `RETRIEVAL_TOP_K` | Chroma hits injected into chat context |
| `DEFAULT_LANGUAGE` | Tag stored on `/train` when the JSON body omits `language` (e.g. `en`, `hi`). Set empty to omit metadata. |

## Running locally

```bash
cd ai-project
chmod +x run.sh
./run.sh
```

With auto-reload during development:

```bash
UVICORN_RELOAD=1 ./run.sh
```

Or directly:

```bash
cd ai-project
export PYTHONPATH=.
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
cd ai-project
docker compose up --build
```

Or:

```bash
./run.sh docker
```

Volumes:

- `chroma_data` → `/app/storage/chroma_data`
- `checkpoints` → `/app/storage/checkpoints`

## API examples

### Health

```bash
curl -s http://localhost:8000/health
```

### Models

```bash
curl -s http://localhost:8000/models
```

### Training

```bash
curl -s -X POST http://localhost:8000/train \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "What is PHP?",
    "output": "PHP is a server-side scripting language.",
    "category": "programming",
    "language": "en"
  }'
```

Each call:

1. Validates the payload.
2. Tokenizes the question and stores a pooled embedding in Chroma with metadata (`category`, `timestamp`, `source=train_api`, and optional `language`).
3. Runs one LM training step and saves `latest.pt` (plus periodic `epoch_n.pt`).

### Chat (OpenAI-compatible)

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64,
    "temperature": 0.8
  }'
```

Optional **`language`** on the same endpoint restricts Chroma retrieval to documents indexed with that tag on `POST /train` (omit for mixed or legacy vectors).

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "पीएचपी क्या है?"}],
    "max_tokens": 64,
    "language": "hi"
  }'
```

### Embeddings

```bash
curl -s -X POST http://localhost:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"text": "Artificial intelligence"}'
```

## Project layout

- [`app/`](app/) — FastAPI entrypoint and routers.
- [`model/`](model/) — Transformer, attention, tokenizer, inference helpers.
- [`training/`](training/) — Optimizer, scheduler, dataset packing, checkpoints.
- [`vectorstore/`](vectorstore/) — Chroma persistent client and collection wrapper.
- [`utils/`](utils/) — Config, logging, validation helpers.
- [`storage/`](storage/) — Checkpoints, vocabulary, and Chroma files (mounted in Docker).

## Retrieval embedding contract

Both training indexing and chat retrieval use **mean-pooled token+position embeddings before transformer blocks**, followed by **L2 normalization**, so stored vectors and queries share the same geometry for cosine distance in Chroma.

## Tests

```bash
cd ai-project
pytest -q
```

## Troubleshooting

- **CUDA requested but not available**: set `DEVICE=cpu` or install a CUDA-enabled `torch` wheel matching your driver.
- **Out of memory**: lower `D_MODEL`, `N_LAYERS`, `MAX_SEQ_LEN`, or `VOCAB_SIZE` via environment variables.
- **Chroma permission errors**: ensure Docker volumes are writable by the container user (`UID`/`GID` bind mounts on Linux hosts).
- **Chroma / PostHog telemetry log noise** (`capture() takes 1 positional argument...`): Chroma 0.5.18 still calls `posthog.capture` even when anonymized telemetry is off. The app registers a **no-op product telemetry** implementation in [`vectorstore/chroma_telemetry_noop.py`](vectorstore/chroma_telemetry_noop.py) via [`vectorstore/chroma_client.py`](vectorstore/chroma_client.py) so nothing is sent and logs stay clean.
- **Poor text quality at cold start**: weights are randomly initialized; quality improves only after many supervised `/train` updates on domain text.
- **Truncation errors during training**: increase `MAX_SEQ_LEN` if prompts plus answers exceed the configured window.

## Future scaling notes

- Swap CPU `torch` wheels for GPU builds and set `DEVICE=cuda`.
- Shard Chroma collections or migrate to a managed vector tier while keeping the same `KnowledgeCollection` interface.
- Add batched `/train` and asynchronous workers without changing route contracts.
