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

# Verify critical packages
echo "[start_up] Verifying psycopg installation..."
uv run python -c "import psycopg; print('psycopg OK')" || {
    echo "[start_up] ERROR: psycopg not found, installing directly..."
    uv add psycopg[binary]
}

# Verify AWS Secrets Manager access
echo "[start_up] Verifying AWS Secrets Manager access..."
uv run python -c "
import os
import sys
try:
    import boto3
    from botocore.exceptions import ClientError
    
    region = os.getenv('AWS_REGION', 'eu-central-1')
    secret_name = os.getenv('OPENROUTER_SECRET_NAME', 'med-sim/openrouter')
    
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

# Verify database connection (if DATABASE_URL can be obtained)
echo "[start_up] Verifying database configuration..."
uv run python -c "
import os
import sys
os.environ['ENVIRONMENT'] = 'prod'
os.environ['AWS_REGION'] = os.getenv('AWS_REGION', 'eu-central-1')
os.environ['POSTGRES_SECRET_NAME'] = os.getenv('POSTGRES_SECRET_NAME', 'med-sim/postgres')

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

echo "[start_up] Starting Streamlit app on ${BIND_ADDR}:${PORT}..."
nohup uv run streamlit run app.py \
  --server.address "$BIND_ADDR" \
  --server.port "$PORT" \
  >> "$OUT_LOG" 2>> "$ERR_LOG" &

APP_PID=$!
echo "$APP_PID" > "$PID_FILE"
echo "[start_up] App started with PID ${APP_PID}. Logs: $OUT_LOG | $ERR_LOG"


