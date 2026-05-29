#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

export PYTHONPATH="${PYTHONPATH:-${ROOT}}"

MODE="${1:-local}"

if [[ "${MODE}" == "docker" ]]; then
  shift || true
  exec docker compose up --build "${@}"
fi

RELOAD_FLAG=()
if [[ "${UVICORN_RELOAD:-0}" == "1" ]]; then
  RELOAD_FLAG=(--reload)
fi

exec uvicorn app.main:app \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8000}" \
  "${RELOAD_FLAG[@]}"
