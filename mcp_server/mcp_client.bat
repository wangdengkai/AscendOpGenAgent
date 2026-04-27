@echo off
REM AscendC MCP Client - Windows 快捷启动脚本

setlocal

REM 设置服务器地址（可修改）
if not defined REMOTE_SERVER_URL (
    set REMOTE_SERVER_URL=http://localhost:9002
)

echo ========================================
echo AscendC MCP Client
echo Server: %REMOTE_SERVER_URL%
echo ========================================
echo.

REM 检查参数
if "%1"=="" goto usage

python "%~dp0mcp_client.py" --server-url %REMOTE_SERVER_URL% %*
goto end

:usage
echo 用法: mcp_client.bat ^<command^> [options]
echo.
echo 可用命令:
echo   full-eval    完整评估流程（推荐）
echo   upload       上传任务
echo   build        编译 Kernel
echo   verify       验证精度
echo   benchmark    性能测试
echo   status       查询状态
echo   download     下载结果
echo   exec         执行命令
echo.
echo 示例:
echo   mcp_client.bat full-eval --task-name ELU --model model.py --kernel-dir kernel/
echo   mcp_client.bat upload --task-name ReLU --model model.py --kernel-dir kernel/
echo   mcp_client.bat status --task-id abc123-def456
echo.
echo 使用 "mcp_client.bat ^<command^> --help" 查看详细信息

:end
endlocal
