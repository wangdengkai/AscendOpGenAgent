# 服务器稳定性问题解决方案

## 🔍 **问题诊断**

你的服务器被 `Killed`，这是典型的 **OOM (Out Of Memory)** 错误。

### **立即执行诊断**

```bash
cd /home/root/AscendOpGenAgent/remote_server
chmod +x diagnose.sh
./diagnose.sh
```

这会检查：
- ✅ 系统内存使用情况
- ✅ OOM Killer 日志
- ✅ 磁盘空间
- ✅ NPU 状态
- ✅ 临时文件大小

---

## ✅ **解决方案（按推荐顺序）**

### **方案 1: 使用 systemd service（最稳定）⭐ 推荐**

#### **步骤 1: 安装服务**

```bash
# 复制 service 文件到 systemd 目录
sudo cp ascendc-server.service /etc/systemd/system/

# 重新加载 systemd
sudo systemctl daemon-reload

# 启动服务
sudo systemctl start ascendc-server

# 设置开机自启
sudo systemctl enable ascendc-server

# 查看状态
sudo systemctl status ascendc-server
```

#### **步骤 2: 查看日志**

```bash
# 实时日志
sudo journalctl -u ascendc-server -f

# 最近 100 行
sudo journalctl -u ascendc-server -n 100

# 今天的日志
sudo journalctl -u ascendc-server --since today
```

#### **步骤 3: 管理服务**

```bash
# 停止服务
sudo systemctl stop ascendc-server

# 重启服务
sudo systemctl restart ascendc-server

# 查看状态
sudo systemctl status ascendc-server
```

**优点**:
- ✅ 自动重启（崩溃后 10 秒自动恢复）
- ✅ 资源限制（最多 4GB 内存）
- ✅ 开机自启
- ✅ 集中日志管理
- ✅ 系统级监控

---

### **方案 2: 使用生产启动脚本（简单）**

```bash
# 赋予执行权限
chmod +x start_production.sh

# 后台运行
nohup ./start_production.sh > logs/startup.log 2>&1 &

# 查看日志
tail -f logs/server.log
```

**特点**:
- ✅ 自动重启（最多 5 次）
- ✅ 内存限制（4GB）
- ✅ 自动清理旧任务
- ✅ 退出代码诊断

---

### **方案 3: 修改 .env 配置文件**

创建或编辑 `.env` 文件：

```bash
cat > .env << 'EOF'
# 服务器配置
PORT=9002
HOST=0.0.0.0
TASKS_DIR=/tmp/ascend_tasks

# ⭐ 关键：限制并发任务数（防止 OOM）
MAX_CONCURRENT_TASKS=2

# 可选：工作进程数
WORKERS=1
EOF
```

**然后重启服务**。

---

## 🔧 **根本原因分析**

### **1. 内存不足（最常见）**

**症状**: `Killed` 或 exit code 137

**解决**:
```bash
# 检查内存
free -h

# 减少并发任务数
echo "MAX_CONCURRENT_TASKS=2" >> .env

# 或者增加 swap
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

### **2. 临时文件积累**

**症状**: `/tmp` 分区满

**解决**:
```bash
# 清理超过 2 小时的任务
find /tmp/ascend_tasks -type d -mmin +120 -exec rm -rf {} +

# 添加定时清理（crontab）
crontab -e
# 添加这行：每小时清理一次
0 * * * * find /tmp/ascend_tasks -type d -mmin +120 -exec rm -rf {} + 2>/dev/null || true
```

---

### **3. NPU 资源泄漏**

**症状**: NPU 显存持续增长

**解决**:
```bash
# 检查 NPU 状态
npu-smi info -t usages

# 重启服务释放 NPU
sudo systemctl restart ascendc-server
```

---

### **4. 子进程未正确清理**

**症状**: 大量僵尸进程

**解决**:
```bash
# 检查子进程
ps aux | grep python

# 杀死所有相关进程
pkill -f "python3 app.py"
```

---

## 📊 **监控建议**

### **实时监控脚本**

创建 `monitor.sh`:

```bash
#!/bin/bash
while true; do
    echo "=== $(date) ==="
    echo "Memory:"
    free -h | grep Mem
    echo "Disk /tmp:"
    df -h /tmp | tail -1
    echo "Server PID:"
    if [ -f server.pid ]; then
        cat server.pid
        ps -p $(cat server.pid) -o pid,vsz,rss,stat || echo "Not running"
    else
        echo "No PID file"
    fi
    echo ""
    sleep 30
done
```

运行:
```bash
chmod +x monitor.sh
./monitor.sh
```

---

### **设置告警**

```bash
# 内存使用超过 80% 时告警
cat > check_memory.sh << 'EOF'
#!/bin/bash
USED=$(free | grep Mem | awk '{print $3/$2 * 100}')
if (( $(echo "$USED > 80" | bc -l) )); then
    echo "⚠️  Memory usage: ${USED}%" | mail -s "Server Alert" admin@example.com
fi
EOF

chmod +x check_memory.sh
crontab -e
# 每 5 分钟检查一次
*/5 * * * * /path/to/check_memory.sh
```

---

## 🎯 **最佳实践配置**

### **.env 推荐配置**

```bash
# 服务器地址
PORT=9002
HOST=0.0.0.0

# 任务目录
TASKS_DIR=/tmp/ascend_tasks

# ⭐ 并发控制（关键！）
MAX_CONCURRENT_TASKS=2

# 工作进程
WORKERS=1
```

### **systemd 推荐配置**

已包含在 `ascendc-server.service` 中：
- MemoryMax=4G
- Restart=on-failure
- RestartSec=10

---

## 🚀 **快速修复步骤**

```bash
# 1. 停止当前服务
pkill -f "python3 app.py"

# 2. 清理临时文件
rm -rf /tmp/ascend_tasks/*

# 3. 创建 .env 配置文件
cat > .env << 'EOF'
PORT=9002
HOST=0.0.0.0
MAX_CONCURRENT_TASKS=2
EOF

# 4. 使用 systemd 启动（推荐）
sudo cp ascendc-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start ascendc-server
sudo systemctl enable ascendc-server

# 5. 验证
sudo systemctl status ascendc-server
curl http://localhost:9002/health
```

---

## 📝 **常见问题**

### **Q: 为什么会被 Killed？**
A: Linux OOM Killer 检测到内存不足，强制杀死进程以保护系统。

### **Q: 如何确认是 OOM？**
A: 运行 `dmesg | grep -i "killed process"` 查看 OOM 日志。

### **Q: MAX_CONCURRENT_TASKS 应该设为多少？**
A: 
- 8GB RAM → 设置为 2
- 16GB RAM → 设置为 4
- 32GB+ RAM → 设置为 8

### **Q: 如何永久解决？**
A: 使用 systemd service + 合理的 MAX_CONCURRENT_TASKS + 定时清理任务。

---

## 📞 **获取帮助**

如果问题依然存在：

```bash
# 收集诊断信息
./diagnose.sh > diagnosis.txt

# 查看完整日志
sudo journalctl -u ascendc-server --no-pager > full_log.txt

# 联系管理员，提供以上两个文件
```
