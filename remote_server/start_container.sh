#!/bin/bash
# 容器环境启动脚本（带自动重启）

set -e

# 加载环境变量
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# 配置
PORT=${PORT:-9002}
HOST=${HOST:-0.0.0.0}
LOG_DIR="logs"
MAX_RESTARTS=10
RESTART_DELAY=5

# 创建日志目录
mkdir -p "$LOG_DIR"

echo "========================================="
echo "AscendC Remote Server (Container Mode)"
echo "========================================="
echo "Host: $HOST"
echo "Port: $PORT"
echo "Max Restarts: $MAX_RESTARTS"
echo "Memory Limit: $(ulimit -v | awk '{printf "%.1f GB", $1/1048576}')"
echo ""

# 设置资源限制
ulimit -v 6291456  # 虚拟内存限制 6GB

restart_count=0
SERVER_PID=""

cleanup() {
    echo ""
    echo "🛑 Stopping server..."
    if [ -n "$SERVER_PID" ] && ps -p $SERVER_PID > /dev/null 2>&1; then
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
    echo "✅ Server stopped"
    exit 0
}

trap cleanup SIGTERM SIGINT

while [ $restart_count -lt $MAX_RESTARTS ]; do
    if [ $restart_count -gt 0 ]; then
        echo ""
        echo "⚠️  Restarting... ($(($restart_count + 1))/$MAX_RESTARTS)"
        sleep $RESTART_DELAY
    fi
    
    # 清理旧任务（超过 2 小时的）
    if [ -d "/tmp/ascend_tasks" ]; then
        OLD_COUNT=$(find /tmp/ascend_tasks -type d -mmin +120 2>/dev/null | wc -l)
        if [ $OLD_COUNT -gt 0 ]; then
            echo "🧹 Cleaning $OLD_COUNT old tasks..."
            find /tmp/ascend_tasks -type d -mmin +120 -exec rm -rf {} + 2>/dev/null || true
        fi
    fi
    
    # 启动服务器
    echo "🚀 Starting server..."
    python3 app.py --host "$HOST" --port "$PORT" \
        >> "$LOG_DIR/server.log" 2>&1 &
    
    SERVER_PID=$!
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
        
        # 分析退出原因
        if [ $EXIT_CODE -eq 137 ]; then
            echo "❌ Killed by OOM Killer (exit code 137)"
            echo "   💡 Solution: Reduce MAX_CONCURRENT_TASKS or increase memory limit"
        elif [ $EXIT_CODE -eq 139 ]; then
            echo "❌ Segmentation fault (exit code 139)"
            echo "   💡 Check logs: tail -f $LOG_DIR/server.log"
        elif [ $EXIT_CODE -eq 143 ]; then
            echo "ℹ️  Terminated by SIGTERM (normal shutdown)"
            break
        else
            echo "❌ Unexpected exit code: $EXIT_CODE"
        fi
        
        restart_count=$((restart_count + 1))
    else
        echo "❌ Server failed to start"
        echo "   💡 Check logs: tail -f $LOG_DIR/server.log"
        exit 1
    fi
done

echo ""
echo "❌ Max restarts reached ($MAX_RESTARTS)"
echo "   💡 Please check logs and fix the issue manually"
echo "   Log file: $LOG_DIR/server.log"
exit 1
