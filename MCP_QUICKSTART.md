# AscendC FastMCP 快速启动指南

## 🚀 5分钟快速开始

### 第一步：启动远程服务器（在 Ascend NPU 服务器上）

```bash
cd remote_server

# Linux/Mac
chmod +x start.sh
./start.sh

# Windows
start.bat
```

服务器将在 `http://0.0.0.0:8080` 启动。

### 第二步：配置本地 MCP Server

```bash
cd mcp_server

# 安装依赖
pip install -r requirements.txt

# 设置远程服务器地址（替换为你的服务器IP）
export REMOTE_SERVER_URL=http://your-server-ip:8080

# 测试连接
curl $REMOTE_SERVER_URL/api/task_status/test
```

### 第三步：配置 Claude Code

在项目根目录创建 `.claude/settings.json`：

```json
{
  "mcpServers": {
    "ascendc-evaluator": {
      "command": "python",
      "args": ["e:/huawei/project/fork/AscendOpGenAgent/mcp_server/server.py"],
      "env": {
        "REMOTE_SERVER_URL": "http://your-server-ip:8080"
      }
    }
  }
}
```

### 第四步：测试使用

在 Claude Code 中输入：

```markdown
@check_task_status
task_id="test"
```

如果返回任务状态信息，说明配置成功！

## 📝 完整使用示例

### 示例：评估一个算子

1. **生成算子代码**（使用 ascend-kernel-developer agent）

2. **调用 MCP 工具进行评估**：

```markdown
@upload_and_evaluate
task_name="31_ELU"
model_py_path="tasks/31_ELU/model.py"
kernel_dir="tasks/31_ELU/kernel"
soc_version="Ascend910B2"
npu_id=0
enable_benchmark=true
```

3. **查看结果**：

返回的 JSON 包含：
- `status`: "success" 或 "failed"
- `task_id`: 任务ID，用于后续查询
- `verification`: 精度验证结果
- `performance`: 性能测试结果
- `download_url`: 下载完整结果的URL

## 🔧 常见问题

### Q1: 如何知道远程服务器是否正常运行？

```bash
curl http://your-server:8080/docs
```

应该看到 FastAPI 的 Swagger UI 文档页面。

### Q2: 如何查看可用的 MCP 工具？

启动 MCP Server 后会列出所有可用工具：
- upload_and_evaluate
- remote_build_kernel
- remote_verify_accuracy
- remote_benchmark
- execute_custom_command
- check_task_status

### Q3: 任务文件保存在哪里？

默认在远程服务器的 `/tmp/ascend_tasks/` 目录。
可以通过 `TASKS_DIR` 环境变量修改。

### Q4: 如何清理过期任务？

服务器每小时自动清理一次。也可以手动删除：

```bash
rm -rf /tmp/ascend_tasks/*
```

### Q5: 支持哪些 SoC 版本？

- Ascend910B1
- Ascend910B2（默认）
- Ascend910B3

通过 `soc_version` 参数指定。

## 🎯 下一步

- 阅读 [完整文档](mcp_server/README.md)
- 查看 [API 参考](mcp_server/README.md#api-参考)
- 了解 [故障排除](mcp_server/README.md#故障排除)

## 💡 提示

1. **首次使用建议**：先用小算子测试整个流程
2. **调试技巧**：查看返回的 logs 和 error 字段
3. **性能优化**：合理设置 warmup 和 repeat 参数
4. **资源管理**：定期清理不需要的任务

---

有问题？查看 [完整文档](mcp_server/README.md) 或提交 Issue。
