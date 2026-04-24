#!/usr/bin/env python3
"""
AscendC Remote Evaluation Server
提供算子编译、验证和性能测试的 HTTP API
"""

import os
import sys
import json
import uuid
import shutil
import asyncio
import signal
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import uvicorn

# 添加项目根目录到路径
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = FastAPI(title="AscendC Remote Evaluator", version="1.0.0")

# ==================== 配置 ====================

TASKS_DIR = Path(os.environ.get("TASKS_DIR", "/tmp/ascend_tasks"))
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_TOTAL_SIZE = 200 * 1024 * 1024  # 200MB
TASK_TTL = 3600  # 任务保留时间（秒）

# 确保任务目录存在
TASKS_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 数据模型 ====================

class TaskUpload(BaseModel):
    """任务上传请求"""
    task_name: str
    model_py: str
    kernel_files: Dict[str, str]
    soc_version: str = "Ascend910B2"
    npu_id: int = 0
    clean_build: bool = True

class BuildRequest(BaseModel):
    """编译请求"""
    task_id: str
    soc_version: Optional[str] = None
    clean: bool = True

class VerifyRequest(BaseModel):
    """验证请求"""
    task_id: str

class BenchmarkRequest(BaseModel):
    """性能测试请求"""
    task_id: str
    impl: str = "ascendc"
    warmup: int = 5
    repeat: int = 10
    seed: int = 0

class CustomCommandRequest(BaseModel):
    """自定义命令请求"""
    task_id: str
    command: str
    scripts: Dict[str, str] = {}
    timeout: int = 300
    env_vars: Dict[str, str] = {}

# ==================== 工具函数 ====================

def detect_cann_path() -> Optional[Path]:
    """智能检测 CANN 安装路径"""
    for env_var in ["ASCEND_INSTALL_PATH", "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"]:
        path = os.environ.get(env_var)
        if path and Path(path).exists():
            return Path(path).resolve()
    
    candidates = [
        Path("/usr/local/Ascend/ascend-toolkit/latest"),
        Path("/usr/local/Ascend/ascend-cann-toolkit/latest"),
        Path.home() / "Ascend" / "ascend-toolkit" / "latest",
    ]
    
    for path in candidates:
        if path.exists():
            return path.resolve()
    
    return None

def find_model_class_in_module(module_path: Path, hints: List[str] = None) -> Optional[str]:
    """检查模块中是否有模型类"""
    try:
        import importlib.util
        import inspect
        import torch.nn as nn
        
        spec = importlib.util.spec_from_file_location("temp_module", module_path)
        if not spec or not spec.loader:
            return None
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hints is None:
            hints = ["ModelNew", "Model", "Net"]
        
        for name in hints:
            cls = getattr(module, name, None)
            if cls and inspect.isclass(cls) and issubclass(cls, nn.Module):
                return name
        
        for name, obj in vars(module).items():
            if inspect.isclass(obj) and issubclass(obj, nn.Module) and obj is not nn.Module:
                return name
        
        return None
    except Exception:
        return None

async def run_command_with_timeout(
    cmd: List[str], 
    timeout: int, 
    cwd: Path,
    env: Dict[str, str] = None
) -> Dict[str, Any]:
    """运行命令并严格控制超时"""
    process = None
    try:
        process_env = os.environ.copy()
        if env:
            process_env.update(env)
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
            env=process_env,
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None
        )
        
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout
        )
        
        return {
            "success": process.returncode == 0,
            "stdout": stdout.decode('utf-8', errors='replace'),
            "stderr": stderr.decode('utf-8', errors='replace'),
            "returncode": process.returncode
        }
        
    except asyncio.TimeoutError:
        if process and process.pid:
            try:
                if hasattr(os, 'killpg'):
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                else:
                    process.kill()
            except (ProcessLookupError, OSError):
                pass
        
        return {
            "success": False,
            "error": f"Command timed out after {timeout}s",
            "stdout": "",
            "stderr": ""
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "stdout": "",
            "stderr": ""
        }

def cleanup_old_tasks():
    """清理过期任务"""
    import time
    current_time = time.time()
    
    for task_dir in TASKS_DIR.iterdir():
        if task_dir.is_dir():
            created_at_file = task_dir / ".created_at"
            if created_at_file.exists():
                try:
                    created_at = float(created_at_file.read_text())
                    if current_time - created_at > TASK_TTL:
                        shutil.rmtree(task_dir)
                        print(f"Cleaned up expired task: {task_dir.name}")
                except (ValueError, IOError):
                    pass

# ==================== API 端点 ====================

@app.post("/api/upload_task")
async def upload_task(data: TaskUpload):
    """接收上传的算子代码"""
    try:
        # 检查文件大小
        total_size = len(data.model_py.encode()) + sum(
            len(content.encode()) for content in data.kernel_files.values()
        )
        
        if total_size > MAX_TOTAL_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Total size {total_size} exceeds limit {MAX_TOTAL_SIZE}"
            )
        
        for filename, content in data.kernel_files.items():
            if len(content.encode()) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File {filename} exceeds size limit {MAX_FILE_SIZE}"
                )
        
        # 创建任务目录
        task_id = str(uuid.uuid4())
        task_dir = TASKS_DIR / task_id
        task_dir.mkdir(parents=True)
        
        # 保存创建时间
        (task_dir / ".created_at").write_text(str(datetime.now().timestamp()))
        
        # 写入 model.py
        (task_dir / "model.py").write_text(data.model_py)
        
        # 写入 kernel 文件
        if data.kernel_files:
            kernel_dir = task_dir / "kernel"
            kernel_dir.mkdir(exist_ok=True)
            for filename, content in data.kernel_files.items():
                (kernel_dir / filename).write_text(content)
        
        # 复制工具脚本
        utils_dir = task_dir / "utils"
        utils_dir.mkdir(exist_ok=True)
        
        for script in ["build_ascendc.py", "verification_ascendc.py", "performance.py"]:
            src = PROJECT_ROOT / "utils" / script
            if src.exists():
                shutil.copy2(src, utils_dir / script)
        
        # 保存配置
        config = {
            "task_name": data.task_name,
            "soc_version": data.soc_version,
            "npu_id": data.npu_id,
            "clean_build": data.clean_build,
            "created_at": datetime.now().isoformat()
        }
        (task_dir / "config.json").write_text(json.dumps(config, indent=2))
        
        return {
            "task_id": task_id,
            "status": "uploaded",
            "message": "Task uploaded successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/build")
async def build_kernel(request: BuildRequest):
    """编译 AscendC kernel"""
    task_dir = TASKS_DIR / request.task_id
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail="Task not found")
    
    config_file = task_dir / "config.json"
    if config_file.exists():
        config = json.loads(config_file.read_text())
    else:
        config = {}
    
    soc_version = request.soc_version or config.get("soc_version", "Ascend910B2")
    clean = request.clean
    
    try:
        # 设置环境变量
        env = os.environ.copy()
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(config.get("npu_id", 0))
        
        cann_path = detect_cann_path()
        if cann_path:
            env["ASCEND_HOME_PATH"] = str(cann_path)
        
        # 执行编译
        cmd = [
            sys.executable,
            str(task_dir / "utils" / "build_ascendc.py"),
            str(task_dir),
            "-v", soc_version
        ]
        
        if clean:
            cmd.append("--clean")
        
        result = await run_command_with_timeout(
            cmd,
            timeout=300,
            cwd=task_dir,
            env=env
        )
        
        return {
            "success": result["success"],
            "logs": result.get("stdout", "") + result.get("stderr", ""),
            "error": result.get("error") or (result.get("stderr") if not result["success"] else None)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "logs": ""
        }

@app.post("/api/verify")
async def verify_accuracy(request: VerifyRequest):
    """验证算子精度"""
    task_dir = TASKS_DIR / request.task_id
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail="Task not found")
    
    config_file = task_dir / "config.json"
    config = json.loads(config_file.read_text()) if config_file.exists() else {}
    
    try:
        env = os.environ.copy()
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(config.get("npu_id", 0))
        
        cmd = [
            sys.executable,
            str(task_dir / "utils" / "verification_ascendc.py"),
            str(task_dir)
        ]
        
        result = await run_command_with_timeout(
            cmd,
            timeout=120,
            cwd=task_dir,
            env=env
        )
        
        # 解析输出判断是否通过
        output = result.get("stdout", "")
        passed = "Result: pass" in output.lower()
        
        return {
            "passed": passed,
            "output": output,
            "error": result.get("error") or (result.get("stderr") if not passed else None),
            "comparison": output
        }
        
    except Exception as e:
        return {
            "passed": False,
            "error": str(e),
            "output": "",
            "comparison": ""
        }

@app.post("/api/benchmark")
async def benchmark_performance(request: BenchmarkRequest):
    """性能基准测试"""
    task_dir = TASKS_DIR / request.task_id
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail="Task not found")
    
    config_file = task_dir / "config.json"
    config = json.loads(config_file.read_text()) if config_file.exists() else {}
    
    try:
        env = os.environ.copy()
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(config.get("npu_id", 0))
        
        cmd = [
            sys.executable,
            str(task_dir / "utils" / "performance.py"),
            str(task_dir),
            request.impl,
            str(request.warmup),
            str(request.repeat),
            str(request.seed)
        ]
        
        result = await run_command_with_timeout(
            cmd,
            timeout=180,
            cwd=task_dir,
            env=env
        )
        
        return {
            "success": result["success"],
            "output": result.get("stdout", ""),
            "error": result.get("error") or (result.get("stderr") if not result["success"] else None)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "output": ""
        }

@app.post("/api/execute_command")
async def execute_custom_command(request: CustomCommandRequest):
    """执行自定义命令"""
    task_dir = TASKS_DIR / request.task_id
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail="Task not found")
    
    try:
        # 上传自定义脚本
        for filename, content in request.scripts.items():
            script_path = task_dir / filename
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text(content)
        
        # 准备环境变量
        env = os.environ.copy()
        config_file = task_dir / "config.json"
        if config_file.exists():
            config = json.loads(config_file.read_text())
            env["ASCEND_RT_VISIBLE_DEVICES"] = str(config.get("npu_id", 0))
        
        env.update(request.env_vars)
        
        # 安全检查
        dangerous_patterns = ['rm -rf /', 'mkfs', 'dd if=', '> /dev/', 'sudo']
        cmd_lower = request.command.lower()
        for pattern in dangerous_patterns:
            if pattern in cmd_lower:
                return {
                    "success": False,
                    "error": "Command contains unsafe operations",
                    "stdout": "",
                    "stderr": "Unsafe command blocked"
                }
        
        # 执行命令
        result = await run_command_with_timeout(
            request.command.split() if isinstance(request.command, str) else request.command,
            timeout=request.timeout,
            cwd=task_dir,
            env=env
        )
        
        return {
            "success": result["success"],
            "returncode": result.get("returncode", -1),
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "command": request.command
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "stdout": "",
            "stderr": ""
        }

@app.get("/api/task_status/{task_id}")
async def get_task_status(task_id: str):
    """查询任务状态"""
    task_dir = TASKS_DIR / task_id
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail="Task not found")
    
    config_file = task_dir / "config.json"
    if config_file.exists():
        config = json.loads(config_file.read_text())
    else:
        config = {}
    
    # 检查各阶段产物
    status = {
        "task_id": task_id,
        "task_name": config.get("task_name", ""),
        "created_at": config.get("created_at", ""),
        "files": {
            "model_py": (task_dir / "model.py").exists(),
            "model_new_ascendc": (task_dir / "model_new_ascendc.py").exists(),
            "kernel_build": (task_dir / "kernel" / "build").exists(),
        }
    }
    
    return status

@app.get("/api/download_results/{task_id}")
async def download_results(task_id: str):
    """打包下载所有结果"""
    from fastapi.responses import FileResponse
    
    task_dir = TASKS_DIR / task_id
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail="Task not found")
    
    # 创建 zip 文件
    import zipfile
    zip_path = task_dir / "results.zip"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file in task_dir.rglob("*"):
            if file.is_file() and file.name != "results.zip":
                arcname = file.relative_to(task_dir)
                zf.write(file, arcname)
    
    return FileResponse(
        zip_path,
        media_type='application/zip',
        filename=f"{task_id}_results.zip"
    )

@app.websocket("/ws/task/{task_id}")
async def task_progress_websocket(websocket: WebSocket, task_id: str):
    """WebSocket 实时推送任务进度"""
    await websocket.accept()
    
    try:
        while True:
            task_dir = TASKS_DIR / task_id
            if not task_dir.exists():
                await websocket.send_json({
                    "task_id": task_id,
                    "status": "not_found",
                    "message": "Task not found"
                })
                break
            
            # 简单状态检查
            has_build = (task_dir / "kernel" / "build").exists()
            has_ascendc = (task_dir / "model_new_ascendc.py").exists()
            
            status = "uploaded"
            if has_build:
                status = "built"
            if has_ascendc:
                status = "verified"
            
            await websocket.send_json({
                "task_id": task_id,
                "status": status,
                "timestamp": datetime.now().isoformat()
            })
            
            if status in ["verified", "completed"]:
                break
            
            await asyncio.sleep(2)
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({
            "task_id": task_id,
            "status": "error",
            "error": str(e)
        })

@app.on_event("startup")
async def startup_event():
    """启动时清理过期任务"""
    cleanup_old_tasks()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
