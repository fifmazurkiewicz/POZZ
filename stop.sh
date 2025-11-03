#!/usr/bin/env bash

# Usage:
#   ./stop.sh
#
# Stops the background Streamlit process started by start_up.sh

set -euo pipefail

PID_FILE="app.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "[stop] PID file not found: $PID_FILE"
  exit 1
fi

APP_PID="$(cat "$PID_FILE")"
if ps -p "$APP_PID" > /dev/null 2>&1; then
  echo "[stop] Stopping PID ${APP_PID}..."
  kill "$APP_PID" || true
  # Wait up to 10s
  for i in {1..10}; do
    if ps -p "$APP_PID" > /dev/null 2>&1; then
      sleep 1
    else
      break
    fi
  done
  # Force kill if still alive
  if ps -p "$APP_PID" > /dev/null 2>&1; then
    echo "[stop] Forcing kill..."
    kill -9 "$APP_PID" || true
  fi
  echo "[stop] Stopped."
else
  echo "[stop] No running process for PID ${APP_PID}."
fi

rm -f "$PID_FILE"
echo "[stop] PID file removed."


