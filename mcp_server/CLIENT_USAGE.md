# AscendC MCP Client 使用指南

## 📖 简介

`mcp_client.py` 是一个简单易用的命令行工具，让你无需编写代码即可调用 AscendC 远程评估服务。

---

## 🚀 快速开始

### 1. 安装依赖

```bash
cd mcp_server
pip install httpx
```

### 2. 配置服务器地址

```bash
# 方式 1: 环境变量（推荐）
export REMOTE_SERVER_URL=http://your-server:9002

# 方式 2: 命令行参数
python mcp_client.py --server-url http://your-server:9002 ...
```

### 3. 运行完整评估（推荐）

```bash
python mcp_client.py full-eval \
  --task-name ELU \
  --model tasks/elu_migration/model.py \
  --kernel-dir tasks/elu_migration/kernel/
```

---

## 📋 可用命令

### 1️⃣ **full-eval** - 完整评估流程 ⭐ 推荐

一键完成：上传 → 编译 → 验证 → 性能测试

```bash
python mcp_client.py full-eval \
  --task-name ELU \
  --model tasks/elu_migration/model.py \
  --kernel-dir tasks/elu_migration/kernel/ \
  --soc-version Ascend910B2 \
  --npu-id 0 \
  --impl ascendc
```

**参数说明**:
- `--task-name`: 任务名称（必需）
- `--model`: model.py 文件路径（必需）
- `--kernel-dir`: kernel 目录路径（必需）
- `--soc-version`: SoC 版本（可选，自动检测）
- `--npu-id`: NPU 设备 ID（可选，自动分配）
- `--no-clean`: 不清理后编译
- `--no-benchmark`: 不运行性能测试
- `--impl`: 性能测试的实现类型 (reference/tilelang/ascendc)

**输出示例**:
```
================================================================================
🚀 Starting Full Evaluation Pipeline
================================================================================

[1/4] Uploading task...
📤 Uploading task: ELU
✅ Upload successful!
   Task ID: 550e8400-e29b-41d4-a716-446655440000
   SoC Version: Ascend910B2
   Allocated NPU: 0

[2/4] Building kernel...
🔨 Building kernel: 550e8400-e29b-41d4-a716-446655440000
✅ Build successful!

[3/4] Verifying accuracy...
✓ Verifying accuracy: 550e8400-e29b-41d4-a716-446655440000
✅ Verification PASSED!

[4/4] Running benchmark...
⚡ Running benchmark: 550e8400-e29b-41d4-a716-446655440000 (ascendc)
✅ Benchmark completed!
   ascendc      OK            0.987       0.980       0.950       1.050       0.035

================================================================================
✅ Full Evaluation Completed!
================================================================================

Task ID: 550e8400-e29b-41d4-a716-446655440000
Upload: ✓
Build: ✓
Verify: ✓
Benchmark: ✓

💡 详细结果已保存到: 550e8400-e29b-41d4-a716-446655440000_evaluation.json
```

---

### 2️⃣ **upload** - 上传任务

只上传代码，不执行后续步骤。

```bash
python mcp_client.py upload \
  --task-name ReLU \
  --model tasks/relu/model.py \
  --kernel-dir tasks/relu/kernel/ \
  --soc-version Ascend910B2 \
  --npu-id 1
```

**输出**:
```
📤 Uploading task: ReLU
✅ Upload successful!
   Task ID: abc123-def456-ghi789
   SoC Version: Ascend910B2
   Allocated NPU: 1

💡 提示: 使用以下 task_id 进行后续操作:
   task_id=abc123-def456-ghi789
```

---

### 3️⃣ **build** - 编译 Kernel

```bash
python mcp_client.py build \
  --task-id abc123-def456-ghi789 \
  --soc-version Ascend910B2
```

---

### 4️⃣ **verify** - 验证精度

```bash
python mcp_client.py verify \
  --task-id abc123-def456-ghi789
```

**输出**:
```
✓ Verifying accuracy: abc123-def456-ghi789
✅ Verification PASSED!
```

---

### 5️⃣ **benchmark** - 性能测试

```bash
python mcp_client.py benchmark \
  --task-id abc123-def456-ghi789 \
  --impl ascendc \
  --warmup 5 \
  --repeat 10 \
  --seed 0
```

**参数**:
- `--impl`: reference | tilelang | ascendc
- `--warmup`: 预热次数（默认 5）
- `--repeat`: 重复次数（默认 10）
- `--seed`: 随机种子（默认 0）

---

### 6️⃣ **status** - 查询任务状态

```bash
python mcp_client.py status \
  --task-id abc123-def456-ghi789
```

**输出**:
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

### 7️⃣ **download** - 下载结果

```bash
python mcp_client.py download \
  --task-id abc123-def456-ghi789 \
  --output ./results/
```

**输出**:
```
📥 Downloading results: abc123-def456-ghi789
✅ Results saved to: ./results/abc123-def456-ghi789_results.zip

💡 结果已保存到: ./results/abc123-def456-ghi789_results.zip
```

---

### 8️⃣ **exec** - 执行自定义命令

```bash
# 简单命令
python mcp_client.py exec \
  --task-id abc123-def456-ghi789 \
  --command "ls -la"

# 复杂命令
python mcp_client.py exec \
  --task-id abc123-def456-ghi789 \
  --command "python custom_script.py --arg1 value1" \
  --timeout 600
```

---

## 🔧 高级用法

### Python API 调用

如果你需要在 Python 脚本中使用：

```python
from mcp_client import AscendCMCPClient

# 创建客户端
with AscendCMCPClient(server_url="http://localhost:9002") as client:
    
    # 方式 1: 完整评估
    result = client.full_evaluation(
        task_name="ELU",
        model_py_path="tasks/elu/model.py",
        kernel_dir="tasks/elu/kernel/",
        run_benchmark=True
    )
    
    # 方式 2: 分步执行
    upload_result = client.upload_task(
        task_name="ELU",
        model_py_path="tasks/elu/model.py",
        kernel_dir="tasks/elu/kernel/"
    )
    task_id = upload_result["task_id"]
    
    build_result = client.build_kernel(task_id=task_id)
    verify_result = client.verify_accuracy(task_id=task_id)
    benchmark_result = client.benchmark_performance(task_id=task_id)
    
    # 下载结果
    zip_path = client.download_results(task_id, output_dir="./results/")
```

---

## 📝 实用示例

### 示例 1: 批量评估多个算子

```bash
#!/bin/bash
# batch_eval.sh

TASKS=(
  "tasks/elu_migration:ELU"
  "tasks/relu_migration:ReLU"
  "tasks/gelu_migration:GELU"
)

for task in "${TASKS[@]}"; do
  IFS=':' read -r dir name <<< "$task"
  
  echo "=========================================="
  echo "Evaluating: $name"
  echo "=========================================="
  
  python mcp_client.py full-eval \
    --task-name "$name" \
    --model "$dir/model.py" \
    --kernel-dir "$dir/kernel/" \
    --no-benchmark  # 跳过性能测试加快速度
  
  echo ""
done
```

---

### 示例 2: 对比不同实现的性能

```bash
#!/bin/bash
# compare_impls.sh

TASK_ID=$1

echo "Comparing implementations for task: $TASK_ID"
echo ""

for impl in reference tilelang ascendc; do
  echo "--- Testing $impl ---"
  python mcp_client.py benchmark \
    --task-id "$TASK_ID" \
    --impl "$impl" \
    --warmup 10 \
    --repeat 20
  
  echo ""
done
```

**用法**:
```bash
./compare_impls.sh abc123-def456-ghi789
```

---

### 示例 3: 监控任务状态

```bash
#!/bin/bash
# monitor_task.sh

TASK_ID=$1

while true; do
  python mcp_client.py status --task-id "$TASK_ID"
  
  # 检查是否完成
  STATUS=$(python mcp_client.py status --task-id "$TASK_ID" 2>&1)
  if echo "$STATUS" | grep -q "kernel_build.*✓"; then
    echo "✅ Task completed!"
    break
  fi
  
  sleep 10
done
```

---

### 示例 4: 自动化工作流

```python
#!/usr/bin/env python3
# auto_pipeline.py

from mcp_client import AscendCMCPClient
import sys

def main():
    client = AscendCMCPClient()
    
    try:
        # 1. 上传并评估
        result = client.full_evaluation(
            task_name=sys.argv[1],
            model_py_path=f"tasks/{sys.argv[1]}/model.py",
            kernel_dir=f"tasks/{sys.argv[1]}/kernel/",
            run_benchmark=True
        )
        
        # 2. 检查结果
        if not result.get("steps", {}).get("verify", {}).get("passed"):
            print("❌ Verification failed!")
            sys.exit(1)
        
        # 3. 下载结果
        task_id = result["task_id"]
        zip_path = client.download_results(task_id, output_dir="./evaluations/")
        
        print(f"✅ All done! Results: {zip_path}")
    
    finally:
        client.close()

if __name__ == "__main__":
    main()
```

**用法**:
```bash
python auto_pipeline.py ELU
```

---

## ❓ 常见问题

### Q1: 如何指定远程服务器地址？

```bash
# 方式 1: 环境变量（永久生效）
export REMOTE_SERVER_URL=http://192.168.1.100:9002

# 方式 2: 命令行参数（单次生效）
python mcp_client.py --server-url http://192.168.1.100:9002 ...
```

---

### Q2: 如何查看详细的 JSON 结果？

完整评估会自动保存 JSON 文件：
```
{task_id}_evaluation.json
```

也可以手动下载：
```bash
python mcp_client.py download --task-id <task_id> --output ./
```

---

### Q3: 编译失败怎么办？

查看错误信息：
```bash
python mcp_client.py build --task-id <task_id>
```

输出会显示编译日志，根据错误信息修复代码后重新上传。

---

### Q4: 如何清理旧任务？

任务会在服务器上保留 1 小时（可配置），之后自动清理。

如需立即清理，可以 SSH 到服务器：
```bash
ssh user@server
rm -rf /tmp/ascend_tasks/<task_id>
```

---

### Q5: 支持哪些 SoC 版本？

常见版本：
- `Ascend910B1`
- `Ascend910B2` (默认)
- `Ascend910B3`

如果不指定，服务器会自动检测。

---

## 🎯 最佳实践

### 1. 使用完整评估流程

```bash
# ✅ 推荐
python mcp_client.py full-eval --task-name ELU --model ... --kernel-dir ...

# ❌ 不推荐（需要手动管理 task_id）
python mcp_client.py upload ...
python mcp_client.py build --task-id ???
```

---

### 2. 保存评估结果

```bash
# 完整评估会自动保存 JSON
python mcp_client.py full-eval ...

# 手动下载所有文件
python mcp_client.py download --task-id <id> --output ./results/
```

---

### 3. 并行评估多个任务

```bash
# 终端 1
python mcp_client.py full-eval --task-name ELU ... --npu-id 0

# 终端 2
python mcp_client.py full-eval --task-name ReLU ... --npu-id 1

# 终端 3
python mcp_client.py full-eval --task-name GELU ... --npu-id 2
```

---

### 4. 先验证再性能测试

```bash
# 快速验证（跳过性能测试）
python mcp_client.py full-eval ... --no-benchmark

# 确认通过后再跑性能测试
python mcp_client.py benchmark --task-id <id> --impl ascendc
```

---

## 📚 相关文档

- [MCP Server README](README.md) - MCP 服务器详细说明
- [API_USAGE.md](../remote_server/API_USAGE.md) - REST API 完整文档
- [QUICKSTART.md](../MCP_QUICKSTART.md) - 快速启动指南

---

## 🆘 获取帮助

```bash
# 查看所有命令
python mcp_client.py --help

# 查看具体命令的帮助
python mcp_client.py full-eval --help
python mcp_client.py upload --help
```

---

**祝使用愉快！** 🎉
