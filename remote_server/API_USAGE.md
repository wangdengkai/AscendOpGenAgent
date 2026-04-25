# AscendC Remote Server API 使用指南

本文档提供直接使用 HTTP API（无需 MCP）的完整示例。

## 📋 目录

- [API 概览](#api-概览)
- [快速开始](#快速开始)
- [详细示例](#详细示例)
- [错误处理](#错误处理)
- [最佳实践](#最佳实践)

## API 概览

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/upload_task` | POST | 上传算子任务 |
| `/api/build` | POST | 编译 Kernel |
| `/api/verify` | POST | 精度验证 |
| `/api/benchmark` | POST | 性能测试 |
| `/api/execute_command` | POST | 执行自定义命令 |
| `/api/task_status/{task_id}` | GET | 查询任务状态 |
| `/api/download_results/{task_id}` | GET | 下载结果 |
| `/ws/task/{task_id}` | WebSocket | 实时进度推送 |

**基础 URL**: `http://your-server:8080`

**交互式文档**: 
- Swagger UI: `http://your-server:8080/docs`
- ReDoc: `http://your-server:8080/redoc`

## 快速开始

### 1. 检查服务器状态

```bash
curl http://localhost:8080/docs
```

应该返回 HTML 页面（Swagger UI）。

### 2. 完整工作流示例

```bash
# Step 1: 上传任务
TASK_ID=$(curl -s -X POST "http://localhost:8080/api/upload_task" \
  -H "Content-Type: application/json" \
  -d '{
    "task_name": "test_elu",
    "model_py": "import torch\n\nclass Model(torch.nn.Module):\n    def forward(self, x):\n        return torch.nn.functional.elu(x)\n\ndef get_input_groups():\n    return [[torch.randn(128, 128, dtype=torch.float16)]]\n\ndef get_init_inputs():\n    return []",
    "kernel_files": {
      "elu_kernel.cpp": "#include <ascendc.h>\n// ... kernel code",
      "pybind11.cpp": "#include <pybind11/pybind11.h>\nPYBIND11_MODULE(elu_ext, m) {}"
    },
    "npu_id": 0
  }' | jq -r '.task_id')

echo "Task ID: $TASK_ID"

# Step 2: 编译
curl -X POST "http://localhost:8080/api/build" \
  -H "Content-Type: application/json" \
  -d "{\"task_id\": \"$TASK_ID\"}"

# Step 3: 验证
curl -X POST "http://localhost:8080/api/verify" \
  -H "Content-Type: application/json" \
  -d "{\"task_id\": \"$TASK_ID\"}"

# Step 4: 性能测试
curl -X POST "http://localhost:8080/api/benchmark" \
  -H "Content-Type: application/json" \
  -d "{
    \"task_id\": \"$TASK_ID\",
    \"warmup\": 5,
    \"repeat\": 10
  }"

# Step 5: 下载结果
curl -O "http://localhost:8080/api/download_results/$TASK_ID"
```

## 详细示例

### 示例 1: 上传任务

#### 基本上传

```bash
curl -X POST "http://localhost:8080/api/upload_task" \
  -H "Content-Type: application/json" \
  -d '{
    "task_name": "31_ELU",
    "model_py": "import torch\n...",
    "kernel_files": {
      "elu_kernel.cpp": "#include ...",
      "pybind11.cpp": "PYBIND11_MODULE..."
    }
  }'
```

#### 指定 SoC 版本

```bash
curl -X POST "http://localhost:8080/api/upload_task" \
  -H "Content-Type: application/json" \
  -d '{
    "task_name": "matmul",
    "model_py": "...",
    "kernel_files": {...},
    "soc_version": "Ascend910B3",
    "npu_id": 2,
    "clean_build": true
  }'
```

**响应:**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "uploaded",
  "message": "Task uploaded successfully",
  "soc_version": "Ascend910B3",
  "auto_detected": false
}
```

### 示例 2: 编译 Kernel

```bash
curl -X POST "http://localhost:8080/api/build" \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "clean": true
  }'
```

**成功响应:**
```json
{
  "success": true,
  "logs": "[build_ascendc] Running: cmake...\n[build_ascendc] Build completed",
  "error": null
}
```

**失败响应:**
```json
{
  "success": false,
  "logs": "Error: syntax error in line 42",
  "error": "Compilation failed"
}
```

### 示例 3: 精度验证

```bash
curl -X POST "http://localhost:8080/api/verify" \
  -H "Content-Type: application/json" \
  -d '{"task_id": "550e8400-e29b-41d4-a716-446655440000"}'
```

**通过:**
```json
{
  "passed": true,
  "output": "Status: PASS\nOperator: test_elu\n...",
  "comparison": "case[0]: matched"
}
```

**失败:**
```json
{
  "passed": false,
  "comparison": "case[0]: max_abs_diff=0.05, mismatch_ratio=2.5%",
  "error": "Accuracy below threshold"
}
```

### 示例 4: 性能测试

```bash
curl -X POST "http://localhost:8080/api/benchmark" \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "impl": "ascendc",
    "warmup": 5,
    "repeat": 10,
    "seed": 0
  }'
```

**响应:**
```json
{
  "success": true,
  "output": "Performance Report\n...\nascendc OK 1.234 1.200 1.100 1.350 0.080"
}
```

### 示例 5: 执行自定义命令

#### 简单命令

```bash
curl -X POST "http://localhost:8080/api/execute_command" \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "command": "ls -la kernel/"
  }'
```

#### 上传并执行脚本

```bash
curl -X POST "http://localhost:8080/api/execute_command" \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "command": "python custom_check.py --threshold 0.95",
    "scripts": {
      "custom_check.py": "#!/usr/bin/env python3\nimport sys\nprint('Custom check passed')"
    },
    "timeout": 60,
    "env_vars": {
      "DEBUG_MODE": "1"
    }
  }'
```

### 示例 6: 查询任务状态

```bash
curl "http://localhost:8080/api/task_status/550e8400-e29b-41d4-a716-446655440000"
```

**响应:**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "task_name": "31_ELU",
  "created_at": "2024-04-24T10:30:00",
  "files": {
    "model_py": true,
    "model_new_ascendc": true,
    "kernel_build": true
  }
}
```

### 示例 7: 下载结果

```bash
curl -O "http://localhost:8080/api/download_results/550e8400-e29b-41d4-a716-446655440000"
```

会下载 `550e8400-e29b-41d4-a716-446655440000_results.zip` 文件。

### 示例 8: WebSocket 实时进度

使用 Python 示例：

```python
import asyncio
import websockets

async def watch_progress(task_id: str):
    uri = f"ws://localhost:8080/ws/task/{task_id}"
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            print(f"Progress: {message}")
            
            # 解析 JSON 检查是否完成
            import json
            data = json.loads(message)
            if data.get("status") in ["completed", "failed"]:
                break

asyncio.run(watch_progress("550e8400-e29b-41d4-a716-446655440000"))
```

## 错误处理

### HTTP 错误码

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 400 | 请求参数错误 |
| 404 | 任务不存在 |
| 413 | 文件过大 |
| 500 | 服务器内部错误 |

### 错误响应格式

```json
{
  "detail": "Error message describing the problem"
}
```

### 常见错误及解决

#### 1. 任务不存在

```bash
curl "http://localhost:8080/api/task_status/invalid-id"
# {"detail":"Task not found"}
```

**解决**: 检查 task_id 是否正确

#### 2. 文件过大

```bash
# {"detail":"Total size 250000000 exceeds limit 209715200"}
```

**解决**: 减小文件大小或分割任务

#### 3. 编译失败

```json
{
  "success": false,
  "logs": "Error: undefined reference to `SomeFunction'",
  "error": "Compilation failed"
}
```

**解决**: 检查 kernel 代码，查看 logs 中的详细错误

## 最佳实践

### 1. 使用 jq 格式化输出

```bash
curl ... | jq .
```

### 2. 保存 task_id 到变量

```bash
TASK_ID=$(curl -s ... | jq -r '.task_id')
echo "Task ID: $TASK_ID"
```

### 3. 检查每一步的结果

```bash
# 编译后检查
RESULT=$(curl -s -X POST .../api/build ...)
if echo "$RESULT" | jq -e '.success' > /dev/null; then
    echo "Build successful"
else
    echo "Build failed:"
    echo "$RESULT" | jq -r '.error'
    exit 1
fi
```

### 4. 使用超时

```bash
curl --max-time 600 ...  # 10分钟超时
```

### 5. 批量处理

```bash
#!/bin/bash
for task_dir in tasks/*/; do
    task_name=$(basename "$task_dir")
    echo "Processing $task_name..."
    
    # 上传、编译、验证...
done
```

## Python 示例

```python
import requests
import json

BASE_URL = "http://localhost:8080"

def upload_task(task_name: str, model_py: str, kernel_files: dict):
    """上传任务"""
    response = requests.post(
        f"{BASE_URL}/api/upload_task",
        json={
            "task_name": task_name,
            "model_py": model_py,
            "kernel_files": kernel_files,
            "npu_id": 0
        }
    )
    response.raise_for_status()
    return response.json()["task_id"]

def build_kernel(task_id: str):
    """编译 kernel"""
    response = requests.post(
        f"{BASE_URL}/api/build",
        json={"task_id": task_id, "clean": True}
    )
    return response.json()

def verify_accuracy(task_id: str):
    """验证精度"""
    response = requests.post(
        f"{BASE_URL}/api/verify",
        json={"task_id": task_id}
    )
    return response.json()

def benchmark(task_id: str):
    """性能测试"""
    response = requests.post(
        f"{BASE_URL}/api/benchmark",
        json={
            "task_id": task_id,
            "warmup": 5,
            "repeat": 10
        }
    )
    return response.json()

# 使用示例
task_id = upload_task(
    "test_elu",
    open("model.py").read(),
    {
        "elu_kernel.cpp": open("kernel/elu_kernel.cpp").read(),
        "pybind11.cpp": open("kernel/pybind11.cpp").read()
    }
)

print(f"Task ID: {task_id}")

build_result = build_kernel(task_id)
if not build_result["success"]:
    print(f"Build failed: {build_result['error']}")
    exit(1)

verify_result = verify_accuracy(task_id)
if not verify_result["passed"]:
    print(f"Verification failed: {verify_result['comparison']}")
    exit(1)

perf_result = benchmark(task_id)
print(f"Performance: {perf_result['output']}")
```

## 常见问题

### Q: 如何知道服务器是否正常运行？

```bash
curl http://localhost:8080/docs
```

### Q: 任务会保留多久？

默认 1 小时（3600秒），之后自动清理。

### Q: 可以同时运行多个任务吗？

可以，每个任务有独立的 task_id。

### Q: 如何查看详细的编译日志？

`/api/build` 返回的 `logs` 字段包含完整输出。

---

更多问题？查看 [Swagger UI](http://localhost:8080/docs) 或提交 Issue。
