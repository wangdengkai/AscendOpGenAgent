# MCP Client 使用指南

## 🎯 概述

MCP Client 是一个**零代码**的命令行工具，让你可以像使用普通命令一样调用远程评估服务。

**无需编写任何 Python 代码**，只需传入参数即可！

---

## 📦 文件说明

```
mcp_server/
├── mcp_client.py          # 核心客户端脚本（Python）
├── mcp_client.bat         # Windows 快捷启动
├── mcp_client.sh          # Linux/Mac 快捷启动
├── CLIENT_USAGE.md        # 详细使用文档（543行）
├── EXAMPLES.md            # 快速示例（169行）
└── README_CLIENT.md       # 本文件（快速入门）
```

---

## ⚡ 30秒快速上手

### 步骤 1: 安装依赖

```bash
cd mcp_server
pip install httpx
```

### 步骤 2: 运行完整评估

```bash
# Windows
mcp_client.bat full-eval ^
  --task-name ELU ^
  --model tasks/elu/model.py ^
  --kernel-dir tasks/elu/kernel/

# Linux/Mac
./mcp_client.sh full-eval \
  --task-name ELU \
  --model tasks/elu/model.py \
  --kernel-dir tasks/elu/kernel/

# 或直接使用 Python
python mcp_client.py full-eval \
  --task-name ELU \
  --model tasks/elu/model.py \
  --kernel-dir tasks/elu/kernel/
```

### 步骤 3: 查看结果

自动保存的文件：
- `{task_id}_evaluation.json` - 详细的 JSON 结果
- 控制台输出 - 实时进度和摘要

---

## 🚀 常用命令速查

| 命令 | 用途 | 示例 |
|------|------|------|
| `full-eval` | 完整评估 ⭐ | `python mcp_client.py full-eval --task-name ELU --model ... --kernel-dir ...` |
| `upload` | 上传任务 | `python mcp_client.py upload --task-name ELU --model ... --kernel-dir ...` |
| `build` | 编译 Kernel | `python mcp_client.py build --task-id <id>` |
| `verify` | 验证精度 | `python mcp_client.py verify --task-id <id>` |
| `benchmark` | 性能测试 | `python mcp_client.py benchmark --task-id <id> --impl ascendc` |
| `status` | 查询状态 | `python mcp_client.py status --task-id <id>` |
| `download` | 下载结果 | `python mcp_client.py download --task-id <id> --output ./` |
| `exec` | 执行命令 | `python mcp_client.py exec --task-id <id> --command "ls -la"` |

---

## 💡 核心优势

### ✅ 零代码
不需要写任何 Python 代码，直接传参即可。

### ✅ 一键完成
`full-eval` 命令自动完成上传→编译→验证→测试全流程。

### ✅ 友好提示
每一步都有清晰的进度提示和错误信息。

### ✅ 自动保存
评估结果自动保存为 JSON 文件，方便后续分析。

### ✅ 跨平台
支持 Windows (`.bat`)、Linux/Mac (`.sh`)、Python 三种方式。

---

## 📖 学习路径

### 新手路线
1. 阅读本文件（5分钟）
2. 查看 [EXAMPLES.md](EXAMPLES.md) 中的常用示例（10分钟）
3. 尝试运行第一个 `full-eval` 命令
4. 遇到问题查看 [CLIENT_USAGE.md](CLIENT_USAGE.md) 的 FAQ

### 进阶路线
1. 学习分步执行（upload/build/verify/benchmark）
2. 掌握批量处理技巧
3. 使用 Python API 集成到自己的工作流
4. 自定义命令执行复杂任务

---

## 🔧 配置服务器地址

### 方法 1: 环境变量（推荐）

```bash
# Windows (PowerShell)
$env:REMOTE_SERVER_URL="http://your-server:9002"

# Linux/Mac
export REMOTE_SERVER_URL=http://your-server:9002

# 永久设置（添加到 ~/.bashrc 或 ~/.zshrc）
echo 'export REMOTE_SERVER_URL=http://your-server:9002' >> ~/.bashrc
source ~/.bashrc
```

### 方法 2: 命令行参数

```bash
python mcp_client.py --server-url http://your-server:9002 full-eval ...
```

---

## 📊 输出示例

### 完整评估输出

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

## ❓ 常见问题

### Q: 如何查看所有可用命令？

```bash
python mcp_client.py --help
```

### Q: 如何查看具体命令的参数？

```bash
python mcp_client.py full-eval --help
```

### Q: 编译失败怎么办？

查看错误输出，修复代码后重新上传：
```bash
python mcp_client.py upload --task-name ELU --model ... --kernel-dir ...
```

### Q: 如何并行运行多个任务？

打开多个终端，分别指定不同的 NPU：
```bash
# 终端 1
python mcp_client.py full-eval ... --npu-id 0

# 终端 2
python mcp_client.py full-eval ... --npu-id 1
```

### Q: 结果保存在哪里？

- JSON 结果：当前目录下的 `{task_id}_evaluation.json`
- 完整文件：使用 `download` 命令下载到指定目录

---

## 🎓 下一步

- 📖 详细文档：[CLIENT_USAGE.md](CLIENT_USAGE.md)
- 💡 实用示例：[EXAMPLES.md](EXAMPLES.md)
- 🔌 API 文档：[../remote_server/API_USAGE.md](../remote_server/API_USAGE.md)

---

## 🆘 获取帮助

```bash
# 查看所有命令
python mcp_client.py --help

# 查看具体命令的帮助
python mcp_client.py full-eval --help
python mcp_client.py upload --help

# 查看示例
cat EXAMPLES.md
```

---

**开始使用吧！** 🚀

最简单的起步：
```bash
python mcp_client.py full-eval --task-name YOUR_TASK --model model.py --kernel-dir kernel/
```
