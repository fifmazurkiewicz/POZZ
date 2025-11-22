#!/usr/bin/env bash
#
# POZZ Application Stop All (AWS/Linux)
# =====================================
# Stops Streamlit application, Cloudflare Tunnel, and Monitor
#
# Usage:
#   chmod +x stop_all.sh
#   ./stop_all.sh

set -euo pipefail

echo "========================================"
echo "Stopping POZZ Application (All)"
echo "========================================"
echo ""

# Stop monitor first
echo "[stop] Stopping monitor..."
./monitor.sh --stop

echo ""

# Stop application (Streamlit + Cloudflare Tunnel)
echo "[stop] Stopping application..."
./stop.sh

echo ""
echo "========================================"
echo "All services stopped"
echo "========================================"
echo ""

