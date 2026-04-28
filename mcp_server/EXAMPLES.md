# MCP Client 快速示例

## 🚀 最常用的命令

### 1. 完整评估（一键完成所有步骤）

```bash
# Windows
mcp_client.bat full-eval ^
  --task-name ELU ^
  --model tasks/elu_migration/model.py ^
  --kernel-dir tasks/elu_migration/kernel/

# Linux/Mac
./mcp_client.sh full-eval \
  --task-name ELU \
  --model tasks/elu_migration/model.py \
  --kernel-dir tasks/elu_migration/kernel/

# Python
python mcp_client.py full-eval \
  --task-name ELU \
  --model tasks/elu_migration/model.py \
  --kernel-dir tasks/elu_migration/kernel/
```

**这会自动完成**:
1. ✅ 上传代码
2. ✅ 编译 Kernel
3. ✅ 验证精度
4. ✅ 性能测试
5. ✅ 保存 JSON 结果

---

### 2. 分步执行

```bash
# Step 1: 上传
python mcp_client.py upload \
  --task-name ReLU \
  --model tasks/relu/model.py \
  --kernel-dir tasks/relu/kernel/

# 输出: task_id=abc123-def456

# Step 2: 编译
python mcp_client.py build --task-id abc123-def456

# Step 3: 验证
python mcp_client.py verify --task-id abc123-def456

# Step 4: 性能测试
python mcp_client.py benchmark --task-id abc123-def456 --impl ascendc

# Step 5: 下载结果
python mcp_client.py download --task-id abc123-def456 --output ./results/
```

---

### 3. 查询状态

```bash
python mcp_client.py status --task-id abc123-def456
```

输出:
```
📊 Task Status:
   Task Name: ELU
   Created At: 2026-04-24T10:30:00
   SoC Version: Ascend910B2
   Allocated NPU: 0
   Files:
     ✓ model_py
     ✓ model_new_ascendc
     ✓ kernel_build
```

---

### 4. 对比不同实现的性能

```bash
# 测试 Reference
python mcp_client.py benchmark --task-id abc123 --impl reference

# 测试 TileLang
python mcp_client.py benchmark --task-id abc123 --impl tilelang

# 测试 AscendC
python mcp_client.py benchmark --task-id abc123 --impl ascendc
```

---

## 💡 实用技巧

### 技巧 1: 设置服务器地址

```bash
# 临时设置
export REMOTE_SERVER_URL=http://192.168.1.100:9002

# 或者每次指定
python mcp_client.py --server-url http://192.168.1.100:9002 ...
```

---

### 技巧 2: 跳过性能测试（加快速度）

```bash
python mcp_client.py full-eval \
  --task-name ELU \
  --model ... \
  --kernel-dir ... \
  --no-benchmark
```

---

### 技巧 3: 指定 NPU 设备

```bash
# 使用 NPU 0
python mcp_client.py full-eval ... --npu-id 0

# 使用 NPU 1
python mcp_client.py full-eval ... --npu-id 1
```

---

### 技巧 4: 批量处理

创建 `batch_eval.sh`:
```bash
#!/bin/bash

for task in ELU ReLU GELU; do
  echo "Evaluating $task..."
  python mcp_client.py full-eval \
    --task-name $task \
    --model tasks/${task,,}_migration/model.py \
    --kernel-dir tasks/${task,,}_migration/kernel/ \
    --no-benchmark
done
```

运行:
```bash
chmod +x batch_eval.sh
./batch_eval.sh
```

---

## 📖 查看更多

详细文档: [CLIENT_USAGE.md](CLIENT_USAGE.md)

获取帮助:
```bash
python mcp_client.py --help
python mcp_client.py full-eval --help
```
