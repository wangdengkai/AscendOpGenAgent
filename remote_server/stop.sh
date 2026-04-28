#!/bin/bash
# 停止远程服务器

set -e

PID_FILE="server.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "❌ PID file not found. Server may not be running."
    echo "   Try: ps aux | grep app.py"
    exit 1
fi

PID=$(cat "$PID_FILE")

echo "Stopping server (PID: $PID)..."

# 尝试优雅停止
if ps -p $PID > /dev/null 2>&1; then
    kill $PID
    
    # 等待进程退出
    for i in {1..10}; do
        if ! ps -p $PID > /dev/null 2>&1; then
            echo "✅ Server stopped successfully"
            rm -f "$PID_FILE"
            exit 0
        fi
        sleep 1
    done
    
    # 强制杀死
    echo "⚠️  Graceful stop failed, forcing..."
    kill -9 $PID
    sleep 1
fi

echo "✅ Server stopped"
rm -f "$PID_FILE"
