#!/bin/bash
# 诊断服务器被 Kill 的原因

echo "========================================="
echo "Server Crash Diagnosis"
echo "========================================="

echo ""
echo "[1] Check system memory:"
free -h
echo ""

echo "[2] Check OOM killer logs:"
dmesg | grep -i "killed process" | tail -5
echo ""

echo "[3] Check recent dmesg errors:"
dmesg | tail -20
echo ""

echo "[4] Check disk space:"
df -h /tmp
echo ""

echo "[5] Check running python processes:"
ps aux | grep python | grep -v grep
echo ""

echo "[6] Check NPU status:"
if command -v npu-smi &> /dev/null; then
    npu-smi info -t usages
else
    echo "npu-smi not found"
fi
echo ""

echo "[7] Check ulimit (resource limits):"
ulimit -a
echo ""

echo "[8] Check /tmp/ascend_tasks size:"
if [ -d "/tmp/ascend_tasks" ]; then
    du -sh /tmp/ascend_tasks
    echo "Task count: $(ls -1 /tmp/ascend_tasks 2>/dev/null | wc -l)"
else
    echo "/tmp/ascend_tasks does not exist"
fi
echo ""

echo "[9] Check server logs for errors:"
if [ -f "logs/server.log" ]; then
    echo "Last 50 lines of server.log:"
    tail -50 logs/server.log
elif [ -f "nohup.out" ]; then
    echo "Last 50 lines of nohup.out:"
    tail -50 nohup.out
fi
echo ""

echo "========================================="
echo "Diagnosis Complete"
echo "========================================="
echo ""
echo "Common causes:"
echo "1. OOM (Out of Memory) - Increase RAM or reduce MAX_CONCURRENT_TASKS"
echo "2. NPU resource leak - Check if tasks properly release NPU"
echo "3. Disk full in /tmp - Clean old tasks"
echo "4. Too many concurrent tasks - Reduce MAX_CONCURRENT_TASKS"
echo ""
echo "Recommended fixes:"
echo "- Set MAX_CONCURRENT_TASKS=2 in .env"
echo "- Add cleanup cron job for /tmp/ascend_tasks"
echo "- Monitor with: watch -n 5 'free -h && df -h /tmp'"
