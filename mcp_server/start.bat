@echo off
REM Start the local MCP server (Windows)

echo =========================================
echo AscendC MCP Server
echo =========================================

REM Check Python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is required but not installed.
    exit /b 1
)

echo Python version:
python --version

REM Install dependencies
echo.
echo Installing dependencies...
pip install -r requirements.txt

REM Set remote server URL
if not defined REMOTE_SERVER_URL set REMOTE_SERVER_URL=http://localhost:8080
echo Remote Server URL: %REMOTE_SERVER_URL%

REM Start MCP server
echo.
echo Starting MCP Server...
echo Configure Claude Code to use this server.
echo =========================================

python server.py
