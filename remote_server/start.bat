@echo off
REM Deploy and start the remote evaluation server (Windows)

echo =========================================
echo AscendC Remote Evaluation Server Setup
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

REM Create tasks directory
if not defined TASKS_DIR set TASKS_DIR=%TEMP%\ascend_tasks
if not exist "%TASKS_DIR%" mkdir "%TASKS_DIR%"
echo Tasks directory: %TASKS_DIR%

REM Set environment variables
set TASKS_DIR=%TASKS_DIR%

REM Start server
echo.
echo Starting server on port 8080...
echo Server URL: http://0.0.0.0:8080
echo.
echo Press Ctrl+C to stop the server
echo =========================================

python app.py
