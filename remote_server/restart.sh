#!/bin/bash
# 重启远程服务器

echo "========================================="
echo "Restarting AscendC Remote Server"
echo "========================================="

# 停止旧服务
if [ -f "server.pid" ]; then
    echo ""
    echo "[1/3] Stopping old server..."
    ./stop.sh
else
    echo "[1/3] No running server found"
fi

# 清理临时文件（可选）
echo ""
echo "[2/3] Cleaning up temporary files..."
if [ -d "/tmp/ascend_tasks" ]; then
    # 只清理超过 1 小时的任务
    find /tmp/ascend_tasks -type d -mmin +60 -exec rm -rf {} + 2>/dev/null || true
    echo "   Cleaned old tasks"
fi

# 启动新服务
echo ""
echo "[3/3] Starting new server..."
./start_background.sh

echo ""
echo "✅ Restart completed!"
