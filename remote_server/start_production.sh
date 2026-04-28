#!/bin/bash
# 生产环境启动脚本 - 带看门狗和自动重启

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
MAX_RESTARTS=5
RESTART_DELAY=10

# 创建日志目录
mkdir -p "$LOG_DIR"

# 设置资源限制（防止 OOM）
ulimit -v 8388608  # 虚拟内存限制 8GB
ulimit -m 4194304  # 物理内存限制 4GB

echo "========================================="
echo "AscendC Remote Server (Production Mode)"
echo "========================================="
echo "Host: $HOST"
echo "Port: $PORT"
echo "Max Restarts: $MAX_RESTARTS"
echo "Memory Limit: 4GB"
echo ""

restart_count=0

while [ $restart_count -lt $MAX_RESTARTS ]; do
    if [ $restart_count -gt 0 ]; then
        echo ""
        echo "⚠️  Restarting... ($(($restart_count + 1))/$MAX_RESTARTS)"
        sleep $RESTART_DELAY
    fi
    
    # 清理旧任务（超过 2 小时的）
    if [ -d "/tmp/ascend_tasks" ]; then
        echo "🧹 Cleaning old tasks..."
        find /tmp/ascend_tasks -type d -mmin +120 -exec rm -rf {} + 2>/dev/null || true
    fi
    
    # 启动服务器
    echo "🚀 Starting server..."
    python3 app.py --host "$HOST" --port "$PORT" \
        >> "$LOG_DIR/server.log" 2>&1 &
    
    SERVER_PID=$!
    echo $SERVER_PID > "$PID_FILE"
    echo "   PID: $SERVER_PID"
    
    # 等待启动
    sleep 5
    
    # 检查是否成功启动
    if ps -p $SERVER_PID > /dev/null 2>&1; then
        echo "✅ Server started successfully"
        
        # 监控进程
        wait $SERVER_PID
        EXIT_CODE=$?
        
        echo ""
        echo "⚠️  Server exited with code: $EXIT_CODE"
        
        if [ $EXIT_CODE -eq 137 ]; then
            echo "❌ Killed by OOM Killer (exit code 137)"
            echo "   Solution: Reduce MAX_CONCURRENT_TASKS or increase RAM"
        elif [ $EXIT_CODE -eq 139 ]; then
            echo "❌ Segmentation fault (exit code 139)"
            echo "   Check logs for details: tail -f $LOG_DIR/server.log"
        fi
        
        restart_count=$((restart_count + 1))
    else
        echo "❌ Server failed to start"
        echo "   Check logs: tail -f $LOG_DIR/server.log"
        exit 1
    fi
done

echo ""
echo "❌ Max restarts reached ($MAX_RESTARTS)"
echo "   Please check logs and fix the issue manually"
echo "   Log file: $LOG_DIR/server.log"
exit 1
