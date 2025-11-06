#!/usr/bin/env bash
#
# POZZ Application Monitor
# ========================
# Monitors the Streamlit application and automatically restarts it if it stops responding.
# Can run as a daemon or in foreground.
#
# Usage:
#   Foreground: ./monitor.sh
#   Background: ./monitor.sh --daemon
#   Stop:       ./monitor.sh --stop
#
# Configuration:
#   CHECK_INTERVAL - How often to check (in seconds, default: 300 = 5 minutes)
#   MAX_RESTART_ATTEMPTS - Max restarts per hour (default: 5)
#   PORT - Port to check (default: 8501)

set -euo pipefail

# Configuration
CHECK_INTERVAL=${CHECK_INTERVAL:-300}  # Check every 5 minutes (300 seconds)
MAX_RESTART_ATTEMPTS=${MAX_RESTART_ATTEMPTS:-5}  # Max 5 restarts per hour
PORT=${PORT:-8501}
LOG_DIR="logs"
MONITOR_LOG="$LOG_DIR/monitor.log"
MONITOR_PID_FILE="monitor.pid"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MONITOR_LOG"
}

# Check if Streamlit is running and responding
check_app_status() {
    local pid_file="$SCRIPT_DIR/app.pid"
    local is_running=false
    local is_responding=false
    
    # Check if PID file exists and process is running
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file" 2>/dev/null || true)
        if [ -n "$pid" ] && ps -p "$pid" > /dev/null 2>&1; then
            is_running=true
        fi
    fi
    
    # Check if port is listening
    if netstat -tlnp 2>/dev/null | grep -q ":$PORT " || ss -tlnp 2>/dev/null | grep -q ":$PORT "; then
        is_responding=true
    elif curl -s --max-time 5 "http://localhost:$PORT" > /dev/null 2>&1; then
        is_responding=true
    fi
    
    # Return status
    if [ "$is_running" = true ] && [ "$is_responding" = true ]; then
        return 0  # OK
    else
        return 1  # Not OK
    fi
}

# Restart the application
restart_app() {
    log "Restarting application..."
    
    # Stop existing processes
    if [ -f "$SCRIPT_DIR/stop.sh" ]; then
        cd "$SCRIPT_DIR"
        bash stop.sh > /dev/null 2>&1 || true
        sleep 2
    fi
    
    # Start application
    if [ -f "$SCRIPT_DIR/start_up.sh" ]; then
        cd "$SCRIPT_DIR"
        bash start_up.sh >> "$LOG_DIR/monitor_restart.log" 2>&1
        sleep 5  # Wait a bit for startup
    else
        log "ERROR: start_up.sh not found in $SCRIPT_DIR"
        return 1
    fi
}

# Count restarts in the last hour
count_recent_restarts() {
    local one_hour_ago=$(date -d '1 hour ago' +%s 2>/dev/null || date -v-1H +%s 2>/dev/null || echo 0)
    local count=0
    
    if [ -f "$MONITOR_LOG" ]; then
        while IFS= read -r line; do
            if [[ "$line" =~ "Restarting application" ]]; then
                # Extract timestamp and compare
                local log_time=$(echo "$line" | grep -oP '\[\K[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}' || echo "")
                if [ -n "$log_time" ]; then
                    local log_epoch=$(date -d "$log_time" +%s 2>/dev/null || date -j -f "%Y-%m-%d %H:%M:%S" "$log_time" +%s 2>/dev/null || echo 0)
                    if [ "$log_epoch" -ge "$one_hour_ago" ]; then
                        ((count++)) || true
                    fi
                fi
            fi
        done < "$MONITOR_LOG"
    fi
    
    echo "$count"
}

# Main monitoring loop
monitor_loop() {
    log "Monitor started. Checking every ${CHECK_INTERVAL} seconds..."
    log "Port: $PORT, Max restarts/hour: $MAX_RESTART_ATTEMPTS"
    
    local consecutive_failures=0
    
    while true; do
        sleep "$CHECK_INTERVAL"
        
        if check_app_status; then
            # App is healthy
            if [ "$consecutive_failures" -gt 0 ]; then
                log "Application recovered after $consecutive_failures failed checks"
                consecutive_failures=0
            fi
        else
            # App is not responding
            ((consecutive_failures++)) || true
            log "WARNING: Application check failed (consecutive failures: $consecutive_failures)"
            
            # Only restart after 2 consecutive failures (to avoid false positives)
            if [ "$consecutive_failures" -ge 2 ]; then
                local recent_restarts=$(count_recent_restarts)
                
                if [ "$recent_restarts" -ge "$MAX_RESTART_ATTEMPTS" ]; then
                    log "ERROR: Too many restarts in the last hour ($recent_restarts >= $MAX_RESTART_ATTEMPTS). Stopping monitor."
                    log "Please investigate the issue manually."
                    exit 1
                fi
                
                log "Attempting restart (restarts in last hour: $recent_restarts/$MAX_RESTART_ATTEMPTS)"
                if restart_app; then
                    log "Restart completed. Waiting 30 seconds before next check..."
                    consecutive_failures=0
                    sleep 30
                else
                    log "ERROR: Restart failed. Will retry in next cycle."
                fi
            fi
        fi
    done
}

# Stop monitor daemon
stop_monitor() {
    if [ -f "$MONITOR_PID_FILE" ]; then
        local pid=$(cat "$MONITOR_PID_FILE" 2>/dev/null || true)
        if [ -n "$pid" ] && ps -p "$pid" > /dev/null 2>&1; then
            log "Stopping monitor (PID: $pid)..."
            kill "$pid" 2>/dev/null || true
            rm -f "$MONITOR_PID_FILE"
            log "Monitor stopped"
        else
            echo "Monitor is not running (PID file exists but process not found)"
            rm -f "$MONITOR_PID_FILE"
        fi
    else
        echo "Monitor is not running (PID file not found)"
    fi
}

# Main script logic
main() {
    case "${1:-}" in
        --daemon)
            if [ -f "$MONITOR_PID_FILE" ]; then
                local pid=$(cat "$MONITOR_PID_FILE" 2>/dev/null || true)
                if [ -n "$pid" ] && ps -p "$pid" > /dev/null 2>&1; then
                    echo "Monitor is already running (PID: $pid)"
                    exit 0
                fi
            fi
            
            log "Starting monitor in daemon mode..."
            nohup "$0" > /dev/null 2>&1 &
            local daemon_pid=$!
            echo "$daemon_pid" > "$MONITOR_PID_FILE"
            echo "Monitor started in background (PID: $daemon_pid)"
            echo "Logs: $MONITOR_LOG"
            echo "To stop: $0 --stop"
            ;;
        --stop)
            stop_monitor
            ;;
        --status)
            if [ -f "$MONITOR_PID_FILE" ]; then
                local pid=$(cat "$MONITOR_PID_FILE" 2>/dev/null || true)
                if [ -n "$pid" ] && ps -p "$pid" > /dev/null 2>&1; then
                    echo "Monitor is running (PID: $pid)"
                    echo "Log: $MONITOR_LOG"
                    echo "Recent activity:"
                    tail -5 "$MONITOR_LOG" 2>/dev/null || echo "No log entries"
                else
                    echo "Monitor is not running"
                fi
            else
                echo "Monitor is not running"
            fi
            ;;
        "")
            # Run in foreground
            monitor_loop
            ;;
        *)
            echo "Usage: $0 [--daemon|--stop|--status]"
            echo ""
            echo "Options:"
            echo "  (no args)  Run in foreground"
            echo "  --daemon   Run in background"
            echo "  --stop     Stop running monitor"
            echo "  --status   Check monitor status"
            echo ""
            echo "Environment variables:"
            echo "  CHECK_INTERVAL        Check interval in seconds (default: 300 = 5 minutes)"
            echo "  MAX_RESTART_ATTEMPTS  Max restarts per hour (default: 5)"
            echo "  PORT                   Port to check (default: 8501)"
            exit 1
            ;;
    esac
}

main "$@"


