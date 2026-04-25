# AscendC FastMCP 远程评估系统

基于 FastMCP 的 AscendC 算子远程编译、验证和性能测试系统。

## 📋 目录

- [架构概述](#架构概述)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [使用示例](#使用示例)
- [API 参考](#api-参考)
- [故障排除](#故障排除)

## 架构概述

```
┌─────────────┐     FastMCP      ┌──────────────┐    HTTP API    ┌─────────────────┐
│ Claude Code │ ◄──────────────► │  FastMCP     │ ◄────────────► │ Remote Server   │
│ (Local)     │     (stdio/SSE)  │  Server      │   (REST API)   │ (Ascend NPU)    │
└─────────────┘                  └──────────────┘                 └─────────────────┘
```

### 组件说明

- **Claude Code**: AI 编程助手，通过 MCP 协议调用工具
- **FastMCP Server**: 本地 MCP 服务，提供工具接口
- **Remote Server**: 远程 FastAPI 服务，执行实际编译和测试
- **Ascend NPU**: 华为昇腾AI处理器

## 快速开始

### 1. 远程服务器部署

在具有 Ascend NPU 的服务器上：

```bash
cd remote_server

# Linux/Mac
chmod +x start.sh
./start.sh

# Windows
start.bat
```

默认监听 `http://0.0.0.0:8080`

### 2. 本地 MCP Server 配置

```bash
cd mcp_server

# 安装依赖
pip install -r requirements.txt

# 设置远程服务器地址
export REMOTE_SERVER_URL=http://your-remote-server:8080

# 启动 MCP Server
python server.py
```

### 3. Claude Code 配置

在 Claude Code 项目中添加 MCP 配置（`.claude/settings.json`）：

```json
{
  "mcpServers": {
    "ascendc-evaluator": {
      "command": "python",
      "args": ["e:/huawei/project/fork/AscendOpGenAgent/mcp_server/server.py"],
      "env": {
        "REMOTE_SERVER_URL": "http://your-remote-server:8080"
      }
    }
  }
}
```

## 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `REMOTE_SERVER_URL` | 远程服务器地址 | `http://localhost:8080` |
| `TASKS_DIR` | 任务存储目录（远程） | `/tmp/ascend_tasks` |

### 配置文件

复制 `config.example.yaml` 为 `config.yaml` 并修改：

```yaml
remote_server:
  url: "http://your-server:8080"
  
defaults:
  soc_version: "Ascend910B2"
  npu_id: 0
```

## 使用示例

### 示例 1: 完整评估流程

```markdown
生成ascendC算子，npu=0，算子描述文件为 benchmarks/NPUKernelBench/level1/31_ELU.py，输出到 tasks/31_ELU/

然后调用 MCP 工具：

@upload_and_evaluate
task_name="31_ELU"
model_py_path="tasks/31_ELU/model.py"
kernel_dir="tasks/31_ELU/kernel"
# soc_version 不指定，服务器自动检测（推荐）
npu_id=0
```

**注意**: `soc_version` 参数现在是可选的，服务器会自动检测 SoC 版本。如果需要手动指定，可以添加 `soc_version="Ascend910B3"`。

### 示例 2: 分步执行

```markdown
# Step 1: 上传任务
@upload_and_evaluate
task_name="matmul"
model_py_path="tasks/matmul/model.py"
kernel_dir="tasks/matmul/kernel"
enable_benchmark=false  # 只做编译和验证

# 获得 task_id 后

# Step 2: 单独编译
@remote_build_kernel
task_id="abc-123-def"
clean=true

# Step 3: 精度验证
@remote_verify_accuracy
task_id="abc-123-def"

# Step 4: 性能测试
@remote_benchmark
task_id="abc-123-def"
warmup=5
repeat=10
```

### 示例 3: 自定义命令

```markdown
# 执行自定义验证脚本
@execute_custom_command
task_id="abc-123-def"
command="python my_validator.py --threshold 0.95"
scripts={
    "my_validator.py": """
import sys
print('Running custom validation...')
# Your custom logic here
"""
}

# 运行性能剖析
@execute_custom_command
task_id="abc-123-def"
command="nsys profile python benchmark.py"
timeout=600
```

### 示例 4: 查询任务状态

```markdown
@check_task_status
task_id="abc-123-def"
```

## API 参考

### MCP 工具

#### upload_and_evaluate

上传算子代码并执行完整评估流程。

**参数:**
- `task_name` (str, 必需): 任务名称
- `model_py_path` (str, 必需): model.py 文件路径
- `kernel_dir` (str, 必需): kernel 目录路径
- `soc_version` (str, 可选): SoC 版本，默认 "Ascend910B2"
- `npu_id` (int, 可选): NPU 设备 ID，默认 0
- `clean_build` (bool, 可选): 是否清理后编译，默认 True
- `enable_benchmark` (bool, 可选): 是否执行性能测试，默认 True

**返回:**
```json
{
  "status": "success",
  "task_id": "abc-123",
  "verification": {"passed": true, ...},
  "performance": {...}
}
```

#### remote_build_kernel

编译 AscendC kernel。

**参数:**
- `task_id` (str, 必需)
- `soc_version` (str, 可选)
- `clean` (bool, 可选)

#### remote_verify_accuracy

验证算子精度。

**参数:**
- `task_id` (str, 必需)

#### remote_benchmark

性能基准测试。

**参数:**
- `task_id` (str, 必需)
- `impl` (str, 可选): "reference"/"tilelang"/"ascendc"
- `warmup` (int, 可选): 预热次数
- `repeat` (int, 可选): 重复次数
- `seed` (int, 可选): 随机种子

#### execute_custom_command

执行自定义命令。

**参数:**
- `task_id` (str, 必需)
- `command` (str, 必需)
- `scripts` (dict, 可选): {文件名: 内容}
- `timeout` (int, 可选): 超时秒数
- `env_vars` (dict, 可选): 额外环境变量

#### check_task_status

查询任务状态。

**参数:**
- `task_id` (str, 必需)

### REST API

远程服务器提供的 HTTP API：

- `POST /api/upload_task` - 上传任务
- `POST /api/build` - 编译
- `POST /api/verify` - 验证
- `POST /api/benchmark` - 性能测试
- `POST /api/execute_command` - 执行自定义命令
- `GET /api/task_status/{task_id}` - 查询状态
- `GET /api/download_results/{task_id}` - 下载结果
- `WS /ws/task/{task_id}` - WebSocket 实时进度

## 故障排除

### 问题 1: 连接远程服务器失败

**错误信息:** `Connection refused` 或 `Timeout`

**解决方案:**
1. 检查远程服务器是否启动：`curl http://your-server:8080/api/task_status/test`
2. 检查防火墙设置
3. 确认 `REMOTE_SERVER_URL` 配置正确

### 问题 2: 编译失败

**错误信息:** `BUILD_COMPILATION_ERROR`

**解决方案:**
1. 查看返回的 logs 字段
2. 检查 kernel 代码语法
3. 确认 CANN 环境正确配置
4. 验证 pybind11.cpp 存在且正确

### 问题 3: 精度验证失败

**错误信息:** `VERIFY_ACCURACY_LOW`

**解决方案:**
1. 查看 comparison 字段了解差异
2. 检查算法实现是否正确
3. 确认数据类型和形状匹配
4. 调整容差参数（atol/rtol）

### 问题 4: 文件过大

**错误信息:** `File size exceeds limit`

**解决方案:**
1. 单个文件限制 50MB
2. 总大小限制 200MB
3. 移除不必要的文件
4. 压缩大型数据文件

### 问题 5: 任务超时

**错误信息:** `Command timed out`

**解决方案:**
1. 增加 timeout 参数
2. 检查是否有死循环
3. 优化代码性能
4. 分批处理大规模任务

## 最佳实践

### 1. 任务管理

- 使用有意义的 task_name
- 定期清理过期任务
- 保留重要的 task_id 用于追溯

### 2. 错误处理

- 检查返回的 status 字段
- 阅读 error.suggestion 获取解决建议
- 查看详细日志定位问题

### 3. 性能优化

- 合理设置 warmup 和 repeat
- 选择合适的 NPU 设备
- 使用 clean_build 确保干净环境

### 4. 安全考虑

- 不要执行危险命令（rm -rf, sudo 等）
- 限制文件大小
- 使用 HTTPS（生产环境）
- 配置 API 认证

## 开发指南

### 添加新工具

1. 在 `mcp_server/server.py` 中添加 `@mcp.tool()` 装饰的函数
2. 在 `remote_server/app.py` 中添加对应的 API 端点
3. 更新本文档

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 测试远程 API
curl -X POST http://localhost:8080/api/upload_task \
  -H "Content-Type: application/json" \
  -d '{"task_name": "test", ...}'
```

## 许可证

本项目采用 Apache 2.0 许可证。详见 [LICENSE](../LICENSE) 文件。

## 支持

如有问题，请提交 Issue 或联系维护团队。
