#!/usr/bin/env bash

# Usage:
#   chmod +x start_up.sh stop.sh
#   ./start_up.sh
#
# Notes:
# - Exports production env vars for Lightsail/EC2
# - Runs Streamlit via uv in background with logs and PID file

set -euo pipefail

# ------------ Configuration (edit as needed) ------------
export ENVIRONMENT="prod"
export AWS_REGION="eu-central-1"

# Secrets Manager entries
export OPENROUTER_SECRET_NAME="med-sim/openrouter"
export POSTGRES_SECRET_NAME="med-sim/postgres"

# Network
export PORT="8501"
export BIND_ADDR="0.0.0.0"

# Python/uv
export UV_LINK_MODE="copy"

# Log files
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
OUT_LOG="$LOG_DIR/app.out.log"
ERR_LOG="$LOG_DIR/app.err.log"
PID_FILE="app.pid"

# ------------ Start application ------------
echo "[start_up] Syncing environment with uv..."
uv sync

echo "[start_up] Starting Streamlit app on ${BIND_ADDR}:${PORT}..."
nohup uv run streamlit run app.py \
  --server.address "$BIND_ADDR" \
  --server.port "$PORT" \
  >> "$OUT_LOG" 2>> "$ERR_LOG" &

APP_PID=$!
echo "$APP_PID" > "$PID_FILE"
echo "[start_up] App started with PID ${APP_PID}. Logs: $OUT_LOG | $ERR_LOG"


