#!/usr/bin/env bash
# macOS launcher (Apple Silicon / Intel) — equivalent of run_server.bat.
# Double-clickable in Finder (.command) or run from a terminal:  ./scripts/run_server.command
set -euo pipefail

# Project root = parent of this script's directory, regardless of where it's run.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
: "${SENSECV_DATA_DIR:=$PROJECT_ROOT/data}"; export SENSECV_DATA_DIR
: "${PORT:=5000}"; export PORT

# Prefer the project venv (.venv/bin/python on macOS), else fall back to python3.
if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
  PYTHON="$PROJECT_ROOT/.venv/bin/python"
else
  PYTHON="$(command -v python3 || command -v python)"
fi

echo "Restarting senseCV server from \"$PROJECT_ROOT\""

# Stop whatever is already listening on $PORT (the previous server instance).
EXISTING="$(lsof -ti tcp:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
if [ -n "$EXISTING" ]; then
  echo "Stopping existing process on port $PORT: $EXISTING"
  kill -9 $EXISTING 2>/dev/null || true
fi

exec "$PYTHON" -m sensecv.app
