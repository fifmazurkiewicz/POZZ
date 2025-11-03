#!/usr/bin/env bash
#
# POZZ Application Stop Script
# =============================
# Stops Streamlit application and Cloudflare Tunnel
#
# Usage:
#   ./stop.sh

set -euo pipefail

LOG_DIR="logs"
PID_FILE="app.pid"
TUNNEL_PID_FILE="cloudflared.pid"

echo "[stop] Stopping POZZ Application..."
echo

# Stop Streamlit
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE" 2>/dev/null || true)
    if [ -n "$PID" ] && ps -p "$PID" > /dev/null 2>&1; then
        kill "$PID" 2>/dev/null || true
        echo "[stop] ✓ Stopped Streamlit app (PID: $PID)"
    else
        echo "[stop] ⚠ Streamlit process not found (PID: $PID)"
    fi
    rm -f "$PID_FILE"
else
    echo "[stop] ⚠ Streamlit PID file not found. App may not be running."
fi

# Stop Cloudflare Tunnel
if [ -f "$TUNNEL_PID_FILE" ]; then
    TUNNEL_PID=$(cat "$TUNNEL_PID_FILE" 2>/dev/null || true)
    if [ -n "$TUNNEL_PID" ] && ps -p "$TUNNEL_PID" > /dev/null 2>&1; then
        kill "$TUNNEL_PID" 2>/dev/null || true
        echo "[stop] ✓ Stopped Cloudflare Tunnel (PID: $TUNNEL_PID)"
    else
        echo "[stop] ⚠ Cloudflare Tunnel process not found (PID: $TUNNEL_PID)"
    fi
    rm -f "$TUNNEL_PID_FILE"
else
    echo "[stop] ⚠ Cloudflare Tunnel PID file not found."
fi

# Also kill any remaining cloudflared processes
pkill -f "cloudflared tunnel" 2>/dev/null && echo "[stop] ✓ Stopped any remaining cloudflared processes" || true

echo
echo "[stop] Done!"
