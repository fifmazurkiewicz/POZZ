#!/usr/bin/env bash
#
# POZZ Application Startup with Monitor (AWS/Linux)
# =================================================
# Starts the Streamlit application and the monitor in one command
#
# Usage:
#   chmod +x start_with_monitor.sh
#   ./start_with_monitor.sh

set -euo pipefail

echo "========================================"
echo "Starting POZZ Application with Monitor"
echo "========================================"
echo ""

# Make sure scripts are executable
chmod +x start_up.sh stop.sh monitor.sh 2>/dev/null || true

# Start the application
echo "[start] Starting Streamlit application..."
./start_up.sh

echo ""
echo "[start] Waiting 10 seconds for application to start..."
sleep 10

# Start the monitor
echo "[start] Starting monitor..."
./monitor.sh --daemon

echo ""
echo "========================================"
echo "Startup Complete"
echo "========================================"
echo ""
echo "Application is running with monitoring enabled"
echo "To check status: ./monitor.sh --status"
echo "To stop: ./stop.sh"
echo ""

