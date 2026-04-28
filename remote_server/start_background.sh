#!/bin/bash
# 后台启动远程服务器（推荐用于生产环境）

set -e

# 加载环境变量
if [ -f ".env" ]; then
    source .env
fi

# 配置
PORT=${PORT:-9002}
HOST=${HOST:-0.0.0.0}
LOG_DIR="logs"
PID_FILE="server.pid"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 检查是否已经在运行
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "⚠️  Server is already running (PID: $OLD_PID)"
        echo "   Stop it first: ./stop.sh"
        exit 1
    else
        echo "🗑️  Removing stale PID file"
        rm -f "$PID_FILE"
    fi
fi

echo "========================================="
echo "Starting AscendC Remote Server (Background)"
echo "========================================="
echo "Host: $HOST"
echo "Port: $PORT"
echo "Log:  $LOG_DIR/server.log"
echo ""

# 后台启动
nohup python3 app.py --host "$HOST" --port "$PORT" \
    > "$LOG_DIR/server.log" 2>&1 &

# 保存 PID
echo $! > "$PID_FILE"

echo "✅ Server started in background"
echo "   PID: $(cat $PID_FILE)"
echo "   Log: tail -f $LOG_DIR/server.log"
echo ""
echo "Waiting for server to be ready..."

# 等待服务器启动
for i in {1..30}; do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "✅ Server is ready!"
        echo "   URL: http://$HOST:$PORT"
        echo "   Health: http://localhost:$PORT/health"
        exit 0
    fi
    sleep 1
done

echo "⚠️  Server may still be starting. Check logs:"
echo "   tail -f $LOG_DIR/server.log"
