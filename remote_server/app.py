#!/usr/bin/env python3
"""
AscendC Remote Evaluation Server - Robust Version
提供 AscendC 算子的编译、验证和性能测试服务

特性:
- 结构化日志记录
- 统一的异常处理
- 请求限流
- 异步文件操作
- 资源自动清理
"""

import os
import sys
import json
import uuid
import shutil
import asyncio
import signal
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

# Third-party imports
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator
import uvicorn
import aiofiles
import structlog
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ==================== 配置 ====================

TASKS_DIR = Path(os.environ.get("TASKS_DIR", "/tmp/ascend_tasks"))
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_TOTAL_SIZE = 200 * 1024 * 1024  # 200MB
TASK_TTL = 3600  # 任务保留时间（秒）
MAX_CONCURRENT_TASKS = int(os.environ.get("MAX_CONCURRENT_TASKS", "10"))

# 确保任务目录存在
TASKS_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 日志配置 ====================

def setup_logging():
    """配置结构化日志"""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
            structlog.dev.ConsoleRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False
    )

setup_logging()
logger = structlog.get_logger()

# ==================== 限流器 ====================

limiter = Limiter(key_func=get_remote_address)

# ==================== 数据模型 ====================

class TaskUpload(BaseModel):
    """任务上传请求模型"""
    task_name: str = Field(..., min_length=1, max_length=100, description="任务名称")
    model_py: str = Field(..., min_length=1, description="model.py 内容")
    kernel_files: Dict[str, str] = Field(..., description="Kernel 文件字典 {filename: content}")
    soc_version: Optional[str] = Field(None, description="SoC 版本，不指定则自动检测")
    npu_id: Optional[int] = Field(None, ge=-1, le=7, description="NPU 设备 ID，None 表示自动分配")
    clean_build: bool = Field(True, description="是否清理后重新编译")
    
    @field_validator('kernel_files')
    @classmethod
    def validate_kernel_files(cls, v):
        if not v:
            raise ValueError("kernel_files cannot be empty")
        if 'pybind11.cpp' not in v:
            raise ValueError("kernel_files must contain 'pybind11.cpp'")
        return v

class BuildRequest(BaseModel):
    """编译请求模型"""
    task_id: str = Field(..., description="任务 ID")
    soc_version: Optional[str] = Field(None, description="SoC 版本")
    clean: bool = Field(True, description="是否清理后编译")

class VerifyRequest(BaseModel):
    """验证请求模型"""
    task_id: str = Field(..., description="任务 ID")

class BenchmarkRequest(BaseModel):
    """性能测试请求模型"""
    task_id: str = Field(..., description="任务 ID")
    impl: str = Field("ascendc", pattern="^(reference|tilelang|ascendc)$", description="实现类型")
    warmup: int = Field(5, ge=0, le=100, description="预热次数")
    repeat: int = Field(10, ge=1, le=1000, description="重复次数")
    seed: int = Field(0, ge=0, description="随机种子")

class CustomCommandRequest(BaseModel):
    """自定义命令请求模型"""
    task_id: str = Field(..., description="任务 ID")
    command: str = Field(..., min_length=1, max_length=1000, description="要执行的命令")
    scripts: Dict[str, str] = Field(default_factory=dict, description="需要上传的脚本")
    timeout: int = Field(300, ge=1, le=3600, description="超时时间（秒）")
    env_vars: Dict[str, str] = Field(default_factory=dict, description="额外环境变量")

# ==================== 异常处理 ====================

class TaskError(Exception):
    """任务执行错误基类"""
    def __init__(self, message: str, error_code: str = "UNKNOWN_ERROR", details: dict = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

class TaskNotFoundError(TaskError):
    """任务未找到错误"""
    def __init__(self, task_id: str):
        super().__init__(
            message=f"Task '{task_id}' not found",
            error_code="TASK_NOT_FOUND",
            details={"task_id": task_id}
        )

class ValidationError(TaskError):
    """验证错误"""
    def __init__(self, message: str, details: dict = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=details or {}
        )

class BuildError(TaskError):
    """编译错误"""
    def __init__(self, message: str, logs: str = ""):
        super().__init__(
            message=message,
            error_code="BUILD_ERROR",
            details={"logs": logs}
        )

async def task_error_handler(request: Request, exc: TaskError):
    """TaskError 异常处理器"""
    logger.error(
        "task_error",
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details,
        path=request.url.path
    )
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "details": exc.details
            }
        }
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """请求验证异常处理器"""
    logger.warning(
        "validation_error",
        errors=exc.errors(),
        path=request.url.path
    )
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid request parameters",
                "details": exc.errors()
            }
        }
    )

async def generic_exception_handler(request: Request, exc: Exception):
    """通用异常处理器"""
    logger.exception(
        "unexpected_error",
        exception_type=type(exc).__name__,
        message=str(exc),
        path=request.url.path
    )
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "details": {"type": type(exc).__name__}
            }
        }
    )

# ==================== NPU 调度器 ====================

import threading

class NPUScheduler:
    """NPU 设备调度器"""
    
    def __init__(self, num_npus: int = None):
        if num_npus is None:
            try:
                import subprocess
                result = subprocess.run(
                    ["npu-smi", "info", "-l"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'Total Count' in line:
                            num_npus = int(line.split(':')[-1].strip())
                            break
            except Exception as e:
                logger.warning("npusmi_detection_failed", error=str(e))
            
            if num_npus is None:
                num_npus = int(os.environ.get("NUM_NPUS", "8"))
        
        self.num_npus = num_npus
        self.lock = threading.Lock()
        self.npu_load = {i: 0 for i in range(num_npus)}
        logger.info("npu_scheduler_initialized", num_npus=num_npus)
    
    def allocate_npu(self, preferred_npu: int = None) -> int:
        """分配 NPU 设备"""
        with self.lock:
            if preferred_npu is not None and 0 <= preferred_npu < self.num_npus:
                if self.npu_load[preferred_npu] < 2:
                    self.npu_load[preferred_npu] += 1
                    logger.info("npu_allocated_preferred", npu_id=preferred_npu, load=self.npu_load[preferred_npu])
                    return preferred_npu
            
            min_load_npu = min(self.npu_load, key=self.npu_load.get)
            self.npu_load[min_load_npu] += 1
            logger.info("npu_allocated_auto", npu_id=min_load_npu, load=self.npu_load[min_load_npu])
            return min_load_npu
    
    def release_npu(self, npu_id: int):
        """释放 NPU 设备"""
        with self.lock:
            if 0 <= npu_id < self.num_npus:
                self.npu_load[npu_id] = max(0, self.npu_load[npu_id] - 1)
                logger.info("npu_released", npu_id=npu_id, load=self.npu_load[npu_id])
    
    def get_status(self) -> dict:
        """获取调度器状态"""
        with self.lock:
            return {
                "num_npus": self.num_npus,
                "npu_load": dict(self.npu_load),
                "total_tasks": sum(self.npu_load.values())
            }

npu_scheduler = NPUScheduler()

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

def detect_soc_version() -> str:
    """自动检测 SoC 版本"""
    env_soc = os.environ.get("ASCEND_SOC_VERSION")
    if env_soc:
        logger.info("soc_detected_from_env", soc_version=env_soc)
        return env_soc
    
    try:
        import subprocess
        result = subprocess.run(
            ["npu-smi", "info", "-t", "device-info"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            output = result.stdout
            if "Ascend910B3" in output:
                logger.info("soc_detected_from_npusmi", soc_version="Ascend910B3")
                return "Ascend910B3"
            elif "Ascend910B2" in output:
                logger.info("soc_detected_from_npusmi", soc_version="Ascend910B2")
                return "Ascend910B2"
            elif "Ascend910B1" in output:
                logger.info("soc_detected_from_npusmi", soc_version="Ascend910B1")
                return "Ascend910B1"
    except Exception as e:
        logger.warning("npusmi_detection_failed", error=str(e))
    
    default_soc = "Ascend910B2"
    logger.info("soc_using_default", soc_version=default_soc)
    return default_soc

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
        
        logger.info("command_executing", cmd=" ".join(cmd), timeout=timeout)
        
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
        
        result = {
            "success": process.returncode == 0,
            "stdout": stdout.decode('utf-8', errors='replace'),
            "stderr": stderr.decode('utf-8', errors='replace'),
            "returncode": process.returncode
        }
        
        logger.info(
            "command_completed",
            success=result["success"],
            returncode=result["returncode"]
        )
        
        return result
        
    except asyncio.TimeoutError:
        if process and process.pid:
            try:
                if hasattr(os, 'killpg'):
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                else:
                    process.kill()
                logger.warning("command_killed_timeout", timeout=timeout)
            except (ProcessLookupError, OSError) as e:
                logger.error("command_kill_failed", error=str(e))
        
        return {
            "success": False,
            "error": f"Command timed out after {timeout}s",
            "stdout": "",
            "stderr": ""
        }
    except Exception as e:
        logger.exception("command_execution_failed", error=str(e))
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
    cleaned = 0
    
    for task_dir in TASKS_DIR.iterdir():
        if task_dir.is_dir():
            created_at_file = task_dir / ".created_at"
            if created_at_file.exists():
                try:
                    created_at = float(created_at_file.read_text(encoding='utf-8'))
                    if current_time - created_at > TASK_TTL:
                        shutil.rmtree(task_dir)
                        cleaned += 1
                        logger.info("task_cleaned", task_id=task_dir.name)
                except (ValueError, IOError) as e:
                    logger.warning("task_cleanup_failed", task_id=task_dir.name, error=str(e))
    
    if cleaned > 0:
        logger.info("cleanup_completed", cleaned_tasks=cleaned)

# ==================== 应用生命周期 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时清理过期任务
    logger.info("application_startup")
    cleanup_old_tasks()
    yield
    # 关闭时的清理工作
    logger.info("application_shutdown")

# ==================== FastAPI 应用 ====================

app = FastAPI(
    title="AscendC Remote Evaluator",
    version="1.0.0",
    description="""
# AscendC 算子远程评估 API

提供 AscendC 算子的编译、验证和性能测试服务。

## 主要功能

- **任务上传**: 上传算子代码到远程服务器
- **Kernel 编译**: 编译 AscendC kernel
- **精度验证**: 验证算子输出正确性
- **性能测试**: 基准性能测试
- **自定义命令**: 执行自定义脚本和命令
- **状态查询**: 实时查询任务状态
- **结果下载**: 打包下载所有结果

## 认证

当前版本无需认证。生产环境建议添加 API Key。
    """,
    contact={
        "name": "AscendOpGenAgent Team",
        "url": "https://github.com/your-repo/AscendOpGenAgent",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    lifespan=lifespan
)

# 注册异常处理器
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_exception_handler(TaskError, task_error_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# ==================== API 端点 ====================

@app.post("/api/upload_task", 
          summary="上传算子任务",
          description="上传算子代码到远程服务器，创建新的评估任务。")
@limiter.limit("10/minute")
async def upload_task(request: Request, data: TaskUpload):
    """接收上传的算子代码"""
    task_id = None
    try:
        logger.info(
            "upload_started",
            task_name=data.task_name,
            model_size=len(data.model_py),
            kernel_files=list(data.kernel_files.keys()),
            client_ip=request.client.host if request.client else "unknown"
        )
        
        # 检查文件大小
        total_size = len(data.model_py.encode()) + sum(
            len(content.encode()) for content in data.kernel_files.values()
        )
        
        logger.info("upload_size_check", total_size=total_size, max_size=MAX_TOTAL_SIZE)
        
        if total_size > MAX_TOTAL_SIZE:
            raise ValidationError(
                f"Total size {total_size} exceeds limit {MAX_TOTAL_SIZE}",
                details={"total_size": total_size, "max_size": MAX_TOTAL_SIZE}
            )
        
        for filename, content in data.kernel_files.items():
            if len(content.encode()) > MAX_FILE_SIZE:
                raise ValidationError(
                    f"File {filename} exceeds size limit {MAX_FILE_SIZE}",
                    details={"filename": filename, "size": len(content.encode())}
                )
        
        # 自动检测 SoC 版本（如果未指定）
        soc_version = data.soc_version if data.soc_version else detect_soc_version()
        
        # 使用 NPU 调度器分配设备（支持并发）
        allocated_npu = npu_scheduler.allocate_npu(data.npu_id)
        
        # 创建任务目录
        task_id = str(uuid.uuid4())
        task_dir = TASKS_DIR / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存创建时间
        async with aiofiles.open(task_dir / ".created_at", 'w', encoding='utf-8') as f:
            await f.write(str(datetime.now().timestamp()))
        
        # 写入 model.py（强制使用 UTF-8 编码）
        async with aiofiles.open(task_dir / "model.py", 'w', encoding='utf-8') as f:
            await f.write(data.model_py)
        
        # 写入 model_new_ascendc.py (if provided)
        if data.model_new_ascendc:
            async with aiofiles.open(task_dir / "model_new_ascendc.py", 'w', encoding='utf-8') as f:
                await f.write(data.model_new_ascendc)
        
        # 写入 kernel 文件（强制使用 UTF-8 编码）
        if data.kernel_files:
            kernel_dir = task_dir / "kernel"
            kernel_dir.mkdir(exist_ok=True)
            for filename, content in data.kernel_files.items():
                async with aiofiles.open(kernel_dir / filename, 'w', encoding='utf-8') as f:
                    await f.write(content)
        
        # 复制工具脚本
        utils_dir = task_dir / "utils"
        utils_dir.mkdir(exist_ok=True)
        
        for script in ["build_ascendc.py", "verification_ascendc.py", "performance.py"]:
            src = PROJECT_ROOT / "utils" / script
            if src.exists():
                shutil.copy2(src, utils_dir / script)
        
        # 保存配置（使用 UTF-8 编码）
        config = {
            "task_id": task_id,
            "task_name": data.task_name,
            "soc_version": soc_version,
            "npu_id": allocated_npu,
            "preferred_npu": data.npu_id,
            "clean_build": data.clean_build,
            "created_at": datetime.now().isoformat(),
            "detected_soc": soc_version != data.soc_version
        }
        
        async with aiofiles.open(task_dir / "config.json", 'w', encoding='utf-8') as f:
            await f.write(json.dumps(config, indent=2, ensure_ascii=False))
        
        logger.info("upload_completed", task_id=task_id, allocated_npu=allocated_npu)
        
        return {
            "success": True,
            "task_id": task_id,
            "status": "uploaded",
            "message": "Task uploaded successfully",
            "soc_version": soc_version,
            "allocated_npu": allocated_npu,
            "auto_detected_soc": soc_version != data.soc_version
        }
        
    except ValidationError:
        raise
    except Exception as e:
        # 如果发生错误，清理已创建的任务目录
        if task_id:
            task_dir = TASKS_DIR / task_id
            if task_dir.exists():
                shutil.rmtree(task_dir, ignore_errors=True)
                logger.info("upload_rollback", task_id=task_id)
        
        logger.exception("upload_failed", error=str(e))
        raise

@app.post("/api/build",
          summary="编译 AscendC Kernel",
          description="编译上传的 AscendC kernel 代码。")
@limiter.limit("5/minute")
async def build_kernel(request: Request, req: BuildRequest):
    """编译 AscendC kernel"""
    try:
        logger.info("build_started", task_id=req.task_id)
        
        task_dir = TASKS_DIR / req.task_id
        if not task_dir.exists():
            raise TaskNotFoundError(req.task_id)
        
        config_file = task_dir / "config.json"
        if config_file.exists():
            async with aiofiles.open(config_file, 'r', encoding='utf-8') as f:
                config = json.loads(await f.read())
        else:
            config = {}
        
        soc_version = req.soc_version or config.get("soc_version", "Ascend910B2")
        clean = req.clean
        
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
        
        if not result["success"]:
            logger.warning("build_failed", task_id=req.task_id, error=result.get("error"))
            raise BuildError(
                "Compilation failed",
                logs=result.get("stderr", "") + result.get("stdout", "")
            )
        
        logger.info("build_completed", task_id=req.task_id)
        
        return {
            "success": True,
            "task_id": req.task_id,
            "logs": result.get("stdout", ""),
            "error": None
        }
        
    except (TaskNotFoundError, BuildError):
        raise
    except Exception as e:
        logger.exception("build_unexpected_error", task_id=req.task_id, error=str(e))
        raise

@app.post("/api/verify",
          summary="验证算子精度",
          description="验证 AscendC 实现与 PyTorch 参考实现的精度对比。")
@limiter.limit("5/minute")
async def verify_accuracy(request: Request, req: VerifyRequest):
    """验证算子精度"""
    try:
        logger.info("verify_started", task_id=req.task_id)
        
        task_dir = TASKS_DIR / req.task_id
        if not task_dir.exists():
            raise TaskNotFoundError(req.task_id)
        
        config_file = task_dir / "config.json"
        if config_file.exists():
            async with aiofiles.open(config_file, 'r', encoding='utf-8') as f:
                config = json.loads(await f.read())
        else:
            config = {}
        
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
        
        if not passed:
            logger.warning("verify_failed", task_id=req.task_id)
        
        logger.info("verify_completed", task_id=req.task_id, passed=passed)
        
        return {
            "success": True,
            "task_id": req.task_id,
            "passed": passed,
            "output": output,
            "error": result.get("error") if not passed else None,
            "comparison": output
        }
        
    except TaskNotFoundError:
        raise
    except Exception as e:
        logger.exception("verify_unexpected_error", task_id=req.task_id, error=str(e))
        raise

@app.post("/api/benchmark",
          summary="性能基准测试",
          description="执行性能基准测试，测量算子延迟。")
@limiter.limit("3/minute")
async def benchmark_performance(request: Request, req: BenchmarkRequest):
    """性能基准测试"""
    try:
        logger.info("benchmark_started", task_id=req.task_id, impl=req.impl)
        
        task_dir = TASKS_DIR / req.task_id
        if not task_dir.exists():
            raise TaskNotFoundError(req.task_id)
        
        config_file = task_dir / "config.json"
        if config_file.exists():
            async with aiofiles.open(config_file, 'r', encoding='utf-8') as f:
                config = json.loads(await f.read())
        else:
            config = {}
        
        env = os.environ.copy()
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(config.get("npu_id", 0))
        
        cmd = [
            sys.executable,
            str(task_dir / "utils" / "performance.py"),
            str(task_dir),
            req.impl,
            str(req.warmup),
            str(req.repeat),
            str(req.seed)
        ]
        
        result = await run_command_with_timeout(
            cmd,
            timeout=180,
            cwd=task_dir,
            env=env
        )
        
        if not result["success"]:
            logger.warning("benchmark_failed", task_id=req.task_id)
        
        logger.info("benchmark_completed", task_id=req.task_id, success=result["success"])
        
        return {
            "success": result["success"],
            "task_id": req.task_id,
            "output": result.get("stdout", ""),
            "error": result.get("error") if not result["success"] else None
        }
        
    except TaskNotFoundError:
        raise
    except Exception as e:
        logger.exception("benchmark_unexpected_error", task_id=req.task_id, error=str(e))
        raise

@app.post("/api/execute_command",
          summary="执行自定义命令",
          description="在任务目录中执行自定义命令，支持上传脚本。")
@limiter.limit("5/minute")
async def execute_custom_command(request: Request, req: CustomCommandRequest):
    """执行自定义命令"""
    try:
        logger.info(
            "custom_command_started",
            task_id=req.task_id,
            command=req.command[:100]  # 只记录前100字符
        )
        
        task_dir = TASKS_DIR / req.task_id
        if not task_dir.exists():
            raise TaskNotFoundError(req.task_id)
        
        # 上传自定义脚本（强制使用 UTF-8 编码）
        for filename, content in req.scripts.items():
            script_path = task_dir / filename
            script_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(script_path, 'w', encoding='utf-8') as f:
                await f.write(content)
            logger.info("script_uploaded", filename=filename)
        
        # 准备环境变量
        env = os.environ.copy()
        config_file = task_dir / "config.json"
        if config_file.exists():
            async with aiofiles.open(config_file, 'r', encoding='utf-8') as f:
                config = json.loads(await f.read())
            env["ASCEND_RT_VISIBLE_DEVICES"] = str(config.get("npu_id", 0))
        
        env.update(req.env_vars)
        
        # 安全检查
        dangerous_patterns = ['rm -rf /', 'mkfs', 'dd if=', '> /dev/', 'sudo']
        cmd_lower = req.command.lower()
        for pattern in dangerous_patterns:
            if pattern in cmd_lower:
                logger.warning("dangerous_command_blocked", command=req.command)
                raise ValidationError(
                    "Command contains unsafe operations",
                    details={"blocked_patterns": [p for p in dangerous_patterns if p in cmd_lower]}
                )
        
        # 执行命令
        result = await run_command_with_timeout(
            req.command.split() if isinstance(req.command, str) else req.command,
            timeout=req.timeout,
            cwd=task_dir,
            env=env
        )
        
        logger.info(
            "custom_command_completed",
            task_id=req.task_id,
            success=result["success"],
            returncode=result.get("returncode")
        )
        
        return {
            "success": result["success"],
            "task_id": req.task_id,
            "returncode": result.get("returncode", -1),
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "command": req.command,
            "error": result.get("error") if not result["success"] else None
        }
        
    except (TaskNotFoundError, ValidationError):
        raise
    except Exception as e:
        logger.exception("custom_command_unexpected_error", task_id=req.task_id, error=str(e))
        raise

@app.get("/api/task_status/{task_id}",
         summary="查询任务状态",
         description="查询任务的当前状态和文件信息。")
@limiter.limit("30/minute")
async def get_task_status(request: Request, task_id: str):
    """查询任务状态"""
    try:
        logger.info("status_checked", task_id=task_id)
        
        task_dir = TASKS_DIR / task_id
        if not task_dir.exists():
            raise TaskNotFoundError(task_id)
        
        config_file = task_dir / "config.json"
        if config_file.exists():
            async with aiofiles.open(config_file, 'r', encoding='utf-8') as f:
                config = json.loads(await f.read())
        else:
            config = {}
        
        # 检查各阶段产物
        status = {
            "success": True,
            "task_id": task_id,
            "task_name": config.get("task_name", ""),
            "created_at": config.get("created_at", ""),
            "soc_version": config.get("soc_version", ""),
            "allocated_npu": config.get("npu_id"),
            "files": {
                "model_py": (task_dir / "model.py").exists(),
                "model_new_ascendc": (task_dir / "model_new_ascendc.py").exists(),
                "kernel_build": (task_dir / "kernel" / "build").exists(),
            }
        }
        
        return status
        
    except TaskNotFoundError:
        raise
    except Exception as e:
        logger.exception("status_check_failed", task_id=task_id, error=str(e))
        raise

@app.get("/api/npu_status",
         summary="查询 NPU 调度器状态",
         description="返回当前 NPU 设备的负载情况，用于监控和调试。")
@limiter.limit("30/minute")
async def get_npu_status(request: Request):
    """查询 NPU 调度器状态"""
    try:
        status = npu_scheduler.get_status()
        logger.info("npu_status_queried", status=status)
        return {
            "success": True,
            **status
        }
    except Exception as e:
        logger.exception("npu_status_query_failed", error=str(e))
        raise

@app.get("/api/download_results/{task_id}",
         summary="下载任务结果",
         description="打包下载任务的所有结果文件。")
@limiter.limit("10/minute")
async def download_results(request: Request, task_id: str):
    """打包下载所有结果"""
    try:
        logger.info("download_requested", task_id=task_id)
        
        task_dir = TASKS_DIR / task_id
        if not task_dir.exists():
            raise TaskNotFoundError(task_id)
        
        # 创建 zip 文件
        import zipfile
        zip_path = task_dir / "results.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file in task_dir.rglob("*"):
                if file.is_file() and file.name != "results.zip":
                    arcname = file.relative_to(task_dir)
                    zf.write(file, arcname)
        
        logger.info("download_ready", task_id=task_id, zip_size=zip_path.stat().st_size)
        
        return FileResponse(
            zip_path,
            media_type='application/zip',
            filename=f"{task_id}_results.zip"
        )
        
    except TaskNotFoundError:
        raise
    except Exception as e:
        logger.exception("download_failed", task_id=task_id, error=str(e))
        raise

@app.websocket("/ws/task/{task_id}")
async def task_progress_websocket(websocket, task_id: str):
    """WebSocket 实时推送任务进度"""
    await websocket.accept()
    
    try:
        while True:
            task_dir = TASKS_DIR / task_id
            if not task_dir.exists():
                await websocket.send_json({
                    "success": False,
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
                "success": True,
                "task_id": task_id,
                "status": status,
                "timestamp": datetime.now().isoformat()
            })
            
            if status in ["verified", "completed"]:
                break
            
            await asyncio.sleep(2)
            
    except Exception as e:
        logger.exception("websocket_error", task_id=task_id, error=str(e))
        try:
            await websocket.send_json({
                "success": False,
                "task_id": task_id,
                "status": "error",
                "error": str(e)
            })
        except:
            pass

@app.get("/health",
         summary="健康检查",
         description="服务健康检查端点。")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "npu_scheduler": npu_scheduler.get_status()
    }

# ==================== 启动入口 ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AscendC Remote Evaluation Server")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"),
                        help="Host to bind to")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")),
                        help="Port to listen on")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "1")),
                        help="Number of worker processes")
    
    args = parser.parse_args()
    
    logger.info(
        "server_starting",
        host=args.host,
        port=args.port,
        workers=args.workers,
        tasks_dir=str(TASKS_DIR),
        max_file_size=MAX_FILE_SIZE,
        max_total_size=MAX_TOTAL_SIZE
    )
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )
