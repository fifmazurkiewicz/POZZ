#!/bin/bash
#
# Check Cloudflare Tunnel and Streamlit Status
# =============================================

echo "=========================================="
echo "Checking POZZ Application Status"
echo "=========================================="
echo

# Check Streamlit
echo "1. Checking Streamlit..."
if [ -f "app.pid" ]; then
    STREAMLIT_PID=$(cat app.pid 2>/dev/null)
    if [ -n "$STREAMLIT_PID" ] && ps -p "$STREAMLIT_PID" > /dev/null 2>&1; then
        echo "   ✓ Streamlit is running (PID: $STREAMLIT_PID)"
        
        # Check if port 8501 is listening
        if netstat -tlnp 2>/dev/null | grep -q ":8501" || ss -tlnp 2>/dev/null | grep -q ":8501"; then
            echo "   ✓ Port 8501 is listening"
        else
            echo "   ✗ Port 8501 is NOT listening"
        fi
        
        # Test if Streamlit responds
        if curl -s http://localhost:8501 > /dev/null 2>&1; then
            echo "   ✓ Streamlit responds on http://localhost:8501"
        else
            echo "   ✗ Streamlit does NOT respond on http://localhost:8501"
        fi
    else
        echo "   ✗ Streamlit is NOT running (PID file exists but process not found)"
    fi
else
    echo "   ✗ Streamlit PID file not found (app.pid)"
fi
echo

# Check Cloudflare Tunnel
echo "2. Checking Cloudflare Tunnel..."
if [ -f "cloudflared.pid" ]; then
    TUNNEL_PID=$(cat cloudflared.pid 2>/dev/null)
    if [ -n "$TUNNEL_PID" ] && ps -p "$TUNNEL_PID" > /dev/null 2>&1; then
        echo "   ✓ Cloudflare Tunnel is running (PID: $TUNNEL_PID)"
    else
    echo "   ✗ Cloudflare Tunnel is NOT running (PID file exists but process not found)"
    fi
else
    echo "   ✗ Cloudflare Tunnel PID file not found (cloudflared.pid)"
fi

# Check for any cloudflared processes
if pgrep -f "cloudflared tunnel" > /dev/null; then
    echo "   ℹ Found cloudflared processes:"
    ps aux | grep "cloudflared tunnel" | grep -v grep
else
    echo "   ✗ No cloudflared tunnel processes found"
fi
echo

# Check logs
echo "3. Recent logs..."
if [ -f "logs/cloudflared.log" ]; then
    echo "   Last 10 lines from cloudflared.log:"
    tail -10 logs/cloudflared.log | sed 's/^/   /'
else
    echo "   ✗ No cloudflared.log file found"
fi
echo

if [ -f "logs/app.err.log" ]; then
    echo "   Last 5 lines from app.err.log:"
    tail -5 logs/app.err.log | sed 's/^/   /'
else
    echo "   ℹ No app.err.log file found"
fi
echo

# Check HTTPS URL
echo "4. HTTPS URL..."
if [ -f "logs/https_url.txt" ]; then
    HTTPS_URL=$(cat logs/https_url.txt 2>/dev/null)
    echo "   URL from file: $HTTPS_URL"
else
    echo "   ✗ No HTTPS URL file found"
    
    # Try to extract from logs
    if [ -f "logs/cloudflared.log" ]; then
        URL_FROM_LOG=$(grep -i "trycloudflare.com" logs/cloudflared.log 2>/dev/null | grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' | tail -1)
        if [ -n "$URL_FROM_LOG" ]; then
            echo "   URL from logs: $URL_FROM_LOG"
        fi
    fi
fi
echo

echo "=========================================="
echo "Recommendations:"
echo "=========================================="
echo

# Recommendations
if ! ps -p "$STREAMLIT_PID" > /dev/null 2>&1; then
    echo "• Restart Streamlit: ./start_up.sh"
fi

if ! pgrep -f "cloudflared tunnel" > /dev/null; then
    echo "• Restart Cloudflare Tunnel: ./start_up.sh"
fi

if ! curl -s http://localhost:8501 > /dev/null 2>&1; then
    echo "• Streamlit may not be responding. Check logs: tail -f logs/app.err.log"
fi

echo "• Check all logs: tail -f logs/*.log"
echo "• Restart everything: ./stop.sh && ./start_up.sh"
echo

