# 服务器快速启动指南

## 🚀 **在服务器上执行以下步骤**

### **步骤 1: 拉取最新代码**

```bash
cd /home/root/AscendOpGenAgent
git pull origin add_mcp
```

---

### **步骤 2: 运行诊断（可选但推荐）**

```bash
cd remote_server
chmod +x diagnose.sh
./diagnose.sh
```

这会告诉你当前系统状态，帮助确认问题原因。

---

### **步骤 3: 创建 .env 配置文件**

```bash
cd remote_server
cat > .env << 'EOF'
# 服务器配置
PORT=9002
HOST=0.0.0.0
TASKS_DIR=/tmp/ascend_tasks

# ⭐ 关键：限制并发数防止 OOM
MAX_CONCURRENT_TASKS=2

# 工作进程数
WORKERS=1
EOF
```

---

### **步骤 4: 使用 systemd 启动（推荐）**

```bash
# 安装 systemd service
sudo cp ascendc-server.service /etc/systemd/system/

# 重新加载 systemd 配置
sudo systemctl daemon-reload

# 启动服务
sudo systemctl start ascendc-server

# 设置开机自启
sudo systemctl enable ascendc-server

# 查看状态
sudo systemctl status ascendc-server
```

---

### **步骤 5: 验证服务**

```bash
# 检查健康状态
curl http://localhost:9002/health

# 应该看到类似输出:
# {"status":"healthy","timestamp":"...","npu_scheduler":{...}}
```

---

### **步骤 6: 查看日志**

```bash
# 实时日志
sudo journalctl -u ascendc-server -f

# 最近 50 行
sudo journalctl -u ascendc-server -n 50

# 今天的日志
sudo journalctl -u ascendc-server --since today
```

---

## 📊 **常用管理命令**

### **查看服务状态**
```bash
sudo systemctl status ascendc-server
```

### **重启服务**
```bash
sudo systemctl restart ascendc-server
```

### **停止服务**
```bash
sudo systemctl stop ascendc-server
```

### **查看资源使用**
```bash
# 内存使用
free -h

# NPU 状态
npu-smi info -t usages

# 磁盘空间
df -h /tmp

# 临时文件大小
du -sh /tmp/ascend_tasks
```

---

## 🔧 **如果服务还是被 Kill**

### **方案 1: 进一步减少并发**

```bash
# 编辑 .env
echo "MAX_CONCURRENT_TASKS=1" > .env

# 重启服务
sudo systemctl restart ascendc-server
```

---

### **方案 2: 增加 Swap**

```bash
# 创建 4GB swap 文件
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 验证
free -h
```

永久生效（添加到 /etc/fstab）:
```bash
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

### **方案 3: 查看详细错误**

```bash
# 检查 OOM killer 日志
dmesg | grep -i "killed process" | tail -5

# 检查系统日志
sudo journalctl -u ascendc-server --no-pager | tail -100

# 保存完整日志用于分析
sudo journalctl -u ascendc-server --since "1 hour ago" > crash_log.txt
```

---

## 📝 **监控脚本**

创建一个简单的监控脚本 `monitor.sh`:

```bash
#!/bin/bash
echo "=== Server Monitor ==="
echo "Time: $(date)"
echo ""
echo "Memory:"
free -h | grep Mem
echo ""
echo "Disk /tmp:"
df -h /tmp | tail -1
echo ""
echo "Server Status:"
sudo systemctl is-active ascendc-server
echo ""
echo "NPU Usage:"
npu-smi info -t usages 2>/dev/null | head -20 || echo "N/A"
echo ""
echo "Task Count: $(ls -1 /tmp/ascend_tasks 2>/dev/null | wc -l)"
```

使用方法:
```bash
chmod +x monitor.sh
./monitor.sh

# 或者持续监控
watch -n 10 ./monitor.sh
```

---

## ✅ **验证清单**

启动后检查以下项目：

- [ ] 服务正在运行: `sudo systemctl status ascendc-server`
- [ ] 健康检查通过: `curl http://localhost:9002/health`
- [ ] 内存使用正常: `free -h` (使用率 < 80%)
- [ ] NPU 可用: `npu-smi info`
- [ ] 磁盘空间充足: `df -h /tmp` (使用率 < 90%)
- [ ] 日志无错误: `sudo journalctl -u ascendc-server -n 50`

---

## 🆘 **获取帮助**

如果遇到问题：

1. 运行诊断脚本: `./diagnose.sh`
2. 查看详细文档: `cat STABILITY_FIX.md`
3. 收集日志: `sudo journalctl -u ascendc-server > logs.txt`
4. 联系管理员并提供 logs.txt

---

**祝使用顺利！** 🎉
