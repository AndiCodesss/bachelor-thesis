#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$ROOT_DIR/src/dashboard/frontend"
BACKEND_LOG="$(mktemp)"
FRONTEND_LOG="$(mktemp)"
TAIL_PIDS=()

cleanup() {
  local exit_code=$?

  for pid in "${TAIL_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done

  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill -- "-$BACKEND_PID" 2>/dev/null || kill "$BACKEND_PID" 2>/dev/null || true
  fi
  if [[ -n "${FRONTEND_PID:-}" ]]; then
    kill -- "-$FRONTEND_PID" 2>/dev/null || kill "$FRONTEND_PID" 2>/dev/null || true
  fi

  rm -f "$BACKEND_LOG" "$FRONTEND_LOG"
  exit "$exit_code"
}

trap cleanup EXIT INT TERM

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

start_log_tail() {
  local label=$1
  local logfile=$2

  tail -n 0 -F "$logfile" | sed "s/^/[$label] /" &
  TAIL_PIDS+=("$!")
}

start_service() {
  local pid_var=$1
  local logfile=$2
  shift 2

  setsid "$@" >"$logfile" 2>&1 &
  printf -v "$pid_var" '%s' "$!"
}

require_cmd uv
require_cmd npm
require_cmd setsid

if [[ ! -f "$ROOT_DIR/pyproject.toml" ]]; then
  echo "Run this script from the project root or keep it in the repository root." >&2
  exit 1
fi

if [[ ! -d "$FRONTEND_DIR" ]]; then
  echo "Frontend directory not found: $FRONTEND_DIR" >&2
  exit 1
fi

if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
  echo "Frontend dependencies are missing. Run 'cd src/dashboard/frontend && npm install' first." >&2
  exit 1
fi

echo "Starting dashboard backend on http://localhost:8000"
start_log_tail "backend" "$BACKEND_LOG"
start_service \
  BACKEND_PID \
  "$BACKEND_LOG" \
  bash \
  -lc \
  "cd '$ROOT_DIR' && exec uv run --with fastapi --with uvicorn --with websockets --with pydantic uvicorn src.dashboard.backend.main:app --port 8000"

echo "Starting dashboard frontend on http://localhost:5173"
start_log_tail "frontend" "$FRONTEND_LOG"
start_service \
  FRONTEND_PID \
  "$FRONTEND_LOG" \
  bash \
  -lc \
  "cd '$FRONTEND_DIR' && exec npm run dev"

sleep 2

if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
  echo "Backend failed to start." >&2
  exit 1
fi

if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
  echo "Frontend failed to start." >&2
  exit 1
fi

echo "Dashboard is running."
echo "Frontend: http://localhost:5173"
echo "Backend:  http://localhost:8000"
echo "Press Ctrl+C to stop both services."

wait "$BACKEND_PID" "$FRONTEND_PID"
