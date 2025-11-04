#!/usr/bin/env bash
#
# POZZ Application Startup Script
# ================================
# This script:
# - Installs/verifies dependencies
# - Verifies AWS Secrets Manager access
# - Verifies database connection
# - Starts Streamlit application
# - Starts Cloudflare Tunnel for HTTPS access
#
# Usage:
#   chmod +x start_up.sh stop.sh
#   ./start_up.sh
#
# Requirements:
# - uv package manager
# - AWS credentials configured
# - Cloudflare Tunnel (cloudflared) installed

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

# Environment
export ENVIRONMENT="prod"
export AWS_REGION="eu-central-1"

# AWS Secrets Manager
export OPENROUTER_SECRET_NAME="POZZ"
export POSTGRES_SECRET_NAME="POZZ"

# Network - Streamlit
export PORT="8501"
export BIND_ADDR="0.0.0.0"

# Python/uv
export UV_LINK_MODE="copy"

# Log files
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
OUT_LOG="$LOG_DIR/app.out.log"
ERR_LOG="$LOG_DIR/app.err.log"
TUNNEL_LOG="$LOG_DIR/cloudflared.log"
PID_FILE="app.pid"
TUNNEL_PID_FILE="cloudflared.pid"

# ============================================================================
# Dependency Installation & Verification
# ============================================================================

echo "[start_up] ========================================"
echo "[start_up] Starting POZZ Application"
echo "[start_up] ========================================"
echo

# Install/update dependencies
echo "[start_up] Syncing environment with uv..."
uv sync
echo

# Verify psycopg (PostgreSQL driver)
echo "[start_up] Verifying psycopg installation..."
uv run python -c "import psycopg; print('psycopg OK')" || {
    echo "[start_up] ERROR: psycopg not found, installing directly..."
    uv add psycopg[binary]
}
echo

# Verify streamlit-mic-recorder (audio recording)
echo "[start_up] Verifying streamlit-mic-recorder installation..."
uv run python -c "from streamlit_mic_recorder import mic_recorder; print('streamlit-mic-recorder OK')" || {
    echo "[start_up] WARNING: streamlit-mic-recorder not found. Audio recording may not work."
    echo "[start_up] Attempting to install streamlit-mic-recorder..."
    uv add streamlit-mic-recorder || {
        echo "[start_up] ERROR: Failed to install streamlit-mic-recorder"
    }
}
echo

# ============================================================================
# AWS Secrets Manager Verification
# ============================================================================

echo "[start_up] Verifying AWS Secrets Manager access..."
uv run python -c "
import os
import sys
try:
    import boto3
    from botocore.exceptions import ClientError
    
    region = os.getenv('AWS_REGION', 'eu-central-1')
    secret_name = os.getenv('OPENROUTER_SECRET_NAME', 'POZZ')
    
    try:
        client = boto3.client('secretsmanager', region_name=region)
        response = client.get_secret_value(SecretId=secret_name)
        print(f'✓ Found secret: {secret_name}')
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == 'ResourceNotFoundException':
            print(f'✗ Secret not found: {secret_name}')
            print(f'  Create it in AWS Secrets Manager (region: {region})')
            sys.exit(1)
        elif error_code == 'AccessDeniedException':
            print(f'✗ Access denied to secret: {secret_name}')
            print(f'  Check IAM permissions for Secrets Manager')
            sys.exit(1)
        else:
            print(f'✗ Error accessing secret {secret_name}: {error_code}')
            sys.exit(1)
    except Exception as e:
        print(f'✗ Failed to connect to AWS: {e}')
        sys.exit(1)
        
except ImportError:
    print('⚠ boto3 not available - skipping AWS verification')
" || {
    echo "[start_up] WARNING: Could not verify AWS secrets. Continuing anyway..."
}
echo

# ============================================================================
# Database Connection Verification
# ============================================================================

echo "[start_up] Verifying database configuration..."
uv run python -c "
import os
import sys
os.environ['ENVIRONMENT'] = 'prod'
os.environ['AWS_REGION'] = os.getenv('AWS_REGION', 'eu-central-1')
os.environ['POSTGRES_SECRET_NAME'] = os.getenv('POSTGRES_SECRET_NAME', 'POZZ')

try:
    from modules.config import get_database_url
    db_url = get_database_url()
    if db_url:
        print(f'✓ Database URL configured (length: {len(db_url)} chars)')
        # Test connection
        try:
            import psycopg
            # Extract just host for display
            if '@' in db_url:
                host = db_url.split('@')[1].split('/')[0].split(':')[0]
                print(f'  Database host: {host}')
        except:
            pass
    else:
        print('✗ DATABASE_URL not found in Secrets Manager or env')
        sys.exit(1)
except Exception as e:
    print(f'⚠ Database check failed: {e}')
" || {
    echo "[start_up] WARNING: Database configuration issue. App may fail to connect."
}
echo

# ============================================================================
# Start Streamlit Application
# ============================================================================

echo "[start_up] Starting Streamlit app on ${BIND_ADDR}:${PORT}..."
nohup uv run streamlit run app.py \
  --server.address "$BIND_ADDR" \
  --server.port "$PORT" \
  >> "$OUT_LOG" 2>> "$ERR_LOG" &

APP_PID=$!
echo "$APP_PID" > "$PID_FILE"
echo "[start_up] ✓ Streamlit started with PID ${APP_PID}"
echo "[start_up]   Logs: $OUT_LOG | $ERR_LOG"
echo

# Wait a moment for Streamlit to start
sleep 3

# Verify Streamlit is running
if ps -p "$APP_PID" > /dev/null; then
    echo "[start_up] ✓ Streamlit is running"
else
    echo "[start_up] ✗ WARNING: Streamlit may have failed to start. Check logs: $ERR_LOG"
fi
echo

# ============================================================================
# Start Cloudflare Tunnel (HTTPS)
# ============================================================================

echo "[start_up] Starting Cloudflare Tunnel for HTTPS access..."

# Check if cloudflared is installed
if ! command -v cloudflared &> /dev/null; then
    echo "[start_up] ✗ WARNING: cloudflared not found. Installing..."
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -O /tmp/cloudflared.deb
    sudo dpkg -i /tmp/cloudflared.deb || {
        echo "[start_up] ✗ ERROR: Failed to install cloudflared"
        echo "[start_up]   HTTPS tunnel will not be available"
        echo "[start_up]   Install manually: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/"
        exit 1
    }
    echo "[start_up] ✓ cloudflared installed"
fi

# Stop any existing tunnel
if [ -f "$TUNNEL_PID_FILE" ]; then
    OLD_TUNNEL_PID=$(cat "$TUNNEL_PID_FILE" 2>/dev/null || true)
    if [ -n "$OLD_TUNNEL_PID" ] && ps -p "$OLD_TUNNEL_PID" > /dev/null 2>&1; then
        echo "[start_up] Stopping existing tunnel (PID: $OLD_TUNNEL_PID)..."
        kill "$OLD_TUNNEL_PID" 2>/dev/null || true
        sleep 2
    fi
fi

# Start Cloudflare Tunnel
echo "[start_up] Starting Cloudflare Tunnel (quick tunnel mode)..."
nohup cloudflared tunnel --url "http://localhost:${PORT}" \
  >> "$TUNNEL_LOG" 2>&1 &

TUNNEL_PID=$!
echo "$TUNNEL_PID" > "$TUNNEL_PID_FILE"
echo "[start_up] ✓ Cloudflare Tunnel started with PID ${TUNNEL_PID}"
echo "[start_up]   Logs: $TUNNEL_LOG"
echo

# Wait for tunnel to initialize and extract HTTPS URL
echo "[start_up] Waiting for tunnel to initialize..."
sleep 8

# Verify Streamlit is responding before starting tunnel
if ! curl -s http://localhost:${PORT} > /dev/null 2>&1; then
    echo "[start_up] ⚠ WARNING: Streamlit is not responding on port ${PORT}"
    echo "[start_up]   Tunnel may not work correctly. Check Streamlit logs: $ERR_LOG"
fi

# Extract HTTPS URL from logs
HTTPS_URL=$(grep -i "trycloudflare.com" "$TUNNEL_LOG" 2>/dev/null | grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' | head -1 || true)

# If still no URL, wait a bit more and check again
if [ -z "$HTTPS_URL" ]; then
    echo "[start_up] Waiting additional 5 seconds for tunnel URL..."
    sleep 5
    HTTPS_URL=$(grep -i "trycloudflare.com" "$TUNNEL_LOG" 2>/dev/null | grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' | head -1 || true)
fi

if [ -n "$HTTPS_URL" ]; then
    echo "[start_up] ========================================"
    echo "[start_up] ✓ HTTPS TUNNEL ACTIVE"
    echo "[start_up] ========================================"
    echo "[start_up] Your application is available at:"
    echo "[start_up]   $HTTPS_URL"
    echo "[start_up] ========================================"
    echo
    echo "$HTTPS_URL" > "$LOG_DIR/https_url.txt"
    echo "[start_up] HTTPS URL saved to: $LOG_DIR/https_url.txt"
else
    echo "[start_up] ⚠ WARNING: Could not extract HTTPS URL from logs"
    echo "[start_up]   Check tunnel logs: tail -f $TUNNEL_LOG"
    echo "[start_up]   Look for lines containing 'trycloudflare.com'"
fi
echo

# ============================================================================
# Summary
# ============================================================================

echo "[start_up] ========================================"
echo "[start_up] Startup Complete"
echo "[start_up] ========================================"
echo "[start_up] Streamlit PID: $APP_PID"
echo "[start_up] Cloudflare Tunnel PID: $TUNNEL_PID"
if [ -n "$HTTPS_URL" ]; then
    echo "[start_up] HTTPS URL: $HTTPS_URL"
fi
echo "[start_up]"
echo "[start_up] Logs:"
echo "[start_up]   - Streamlit: $OUT_LOG | $ERR_LOG"
echo "[start_up]   - Tunnel: $TUNNEL_LOG"
echo "[start_up]   - HTTPS URL: $LOG_DIR/https_url.txt"
echo "[start_up]"
echo "[start_up] To stop: ./stop.sh"
echo "[start_up] ========================================"


