#!/bin/bash
# AscendC MCP Client - Linux/Mac 快捷启动脚本

# 设置服务器地址（可修改）
export REMOTE_SERVER_URL="${REMOTE_SERVER_URL:-http://localhost:9002}"

echo "========================================"
echo "AscendC MCP Client"
echo "Server: $REMOTE_SERVER_URL"
echo "========================================"
echo ""

# 检查参数
if [ $# -eq 0 ]; then
    echo "用法: ./mcp_client.sh <command> [options]"
    echo ""
    echo "可用命令:"
    echo "  full-eval    完整评估流程（推荐）"
    echo "  upload       上传任务"
    echo "  build        编译 Kernel"
    echo "  verify       验证精度"
    echo "  benchmark    性能测试"
    echo "  status       查询状态"
    echo "  download     下载结果"
    echo "  exec         执行命令"
    echo ""
    echo "示例:"
    echo "  ./mcp_client.sh full-eval --task-name ELU --model model.py --kernel-dir kernel/"
    echo "  ./mcp_client.sh upload --task-name ReLU --model model.py --kernel-dir kernel/"
    echo "  ./mcp_client.sh status --task-id abc123-def456"
    echo ""
    echo "使用 './mcp_client.sh <command> --help' 查看详细信息"
    exit 1
fi

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 运行 Python 脚本
python "$SCRIPT_DIR/mcp_client.py" --server-url "$REMOTE_SERVER_URL" "$@"
