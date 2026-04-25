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

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时清理过期任务
    cleanup_old_tasks()
    yield
    # 关闭时的清理工作（如果需要）

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

## 快速开始

### 1. 上传任务

```bash
curl -X POST "http://localhost:8080/api/upload_task" \\
  -H "Content-Type: application/json" \\
  -d '{
    "task_name": "31_ELU",
    "model_py": "import torch...",
    "kernel_files": {
      "elu_kernel.cpp": "#include ...",
      "pybind11.cpp": "PYBIND11_MODULE..."
    }
    # npu_id 可选，不指定则服务器自动分配
  }'
```

### 2. 编译 Kernel

```bash
curl -X POST "http://localhost:8080/api/build" \\
  -H "Content-Type: application/json" \\
  -d '{"task_id": "your-task-id"}'
```

### 3. 验证精度

```bash
curl -X POST "http://localhost:8080/api/verify" \\
  -H "Content-Type: application/json" \\
  -d '{"task_id": "your-task-id"}'
```

### 4. 性能测试

```bash
curl -X POST "http://localhost:8080/api/benchmark" \\
  -H "Content-Type: application/json" \\
  -d '{
    "task_id": "your-task-id",
    "warmup": 5,
    "repeat": 10
  }'
```

## 错误处理

所有 API 返回统一的错误格式：

```json
{
  "detail": "Error message"
}
```

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
    lifespan=lifespan  # 使用新的 lifespan 方式
)

# ==================== 配置 ====================

TASKS_DIR = Path(os.environ.get("TASKS_DIR", "/tmp/ascend_tasks"))
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_TOTAL_SIZE = 200 * 1024 * 1024  # 200MB
TASK_TTL = 3600  # 任务保留时间（秒）

# NPU 调度器
import threading

class NPUScheduler:
    """
    NPU 设备调度器
    
    功能：
    1. 自动分配空闲 NPU 设备
    2. 跟踪设备使用状态
    3. 支持负载均衡
    """
    
    def __init__(self, num_npus: int = None):
        """
        初始化调度器
        
        Args:
            num_npus: NPU 数量，None 则自动检测
        """
        if num_npus is None:
            # 尝试自动检测 NPU 数量
            try:
                import subprocess
                result = subprocess.run(
                    ["npu-smi", "info", "-l"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    # 解析输出获取 NPU 数量
                    for line in result.stdout.split('\n'):
                        if 'Total Count' in line:
                            num_npus = int(line.split(':')[-1].strip())
                            break
            except Exception:
                pass
            
            # 默认值
            if num_npus is None:
                num_npus = int(os.environ.get("NUM_NPUS", "8"))
        
        self.num_npus = num_npus
        self.lock = threading.Lock()
        # 记录每个 NPU 上的活跃任务数
        self.npu_load = {i: 0 for i in range(num_npus)}
        print(f"NPU Scheduler initialized with {num_npus} devices")
    
    def allocate_npu(self, preferred_npu: int = None) -> int:
        """
        分配 NPU 设备
        
        Args:
            preferred_npu: 首选 NPU ID（可选）
            
        Returns:
            分配的 NPU ID
        """
        with self.lock:
            # 如果指定了首选 NPU 且负载较低，优先使用
            if preferred_npu is not None and 0 <= preferred_npu < self.num_npus:
                if self.npu_load[preferred_npu] < 2:  # 负载阈值
                    self.npu_load[preferred_npu] += 1
                    print(f"Allocated preferred NPU {preferred_npu} (load: {self.npu_load[preferred_npu]})")
                    return preferred_npu
            
            # 选择负载最低的 NPU
            min_load_npu = min(self.npu_load, key=self.npu_load.get)
            self.npu_load[min_load_npu] += 1
            print(f"Allocated NPU {min_load_npu} (load: {self.npu_load[min_load_npu]})")
            return min_load_npu
    
    def release_npu(self, npu_id: int):
        """
        释放 NPU 设备
        
        Args:
            npu_id: 要释放的 NPU ID
        """
        with self.lock:
            if 0 <= npu_id < self.num_npus:
                self.npu_load[npu_id] = max(0, self.npu_load[npu_id] - 1)
                print(f"Released NPU {npu_id} (load: {self.npu_load[npu_id]})")
    
    def get_status(self) -> dict:
        """获取调度器状态"""
        with self.lock:
            return {
                "num_npus": self.num_npus,
                "npu_load": dict(self.npu_load),
                "total_tasks": sum(self.npu_load.values())
            }

# 创建全局 NPU 调度器实例
npu_scheduler = NPUScheduler()

# 确保任务目录存在
TASKS_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 数据模型 ====================

class TaskUpload(BaseModel):
    """任务上传请求"""
    task_name: str
    model_py: str
    model_new_ascendc: Optional[str] = None  # Add this field
    kernel_files: Dict[str, str]
    soc_version: Optional[str] = None  # None 表示自动检测
    npu_id: Optional[int] = None  # None 表示服务器自动分配
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

def detect_soc_version() -> str:
    """
    自动检测 SoC 版本
    
    通过以下方式检测：
    1. 环境变量 ASCEND_SOC_VERSION
    2. npu-smi info 命令
    3. 默认值 Ascend910B2
    """
    # 1. 检查环境变量
    env_soc = os.environ.get("ASCEND_SOC_VERSION")
    if env_soc:
        print(f"Detected SoC from environment: {env_soc}")
        return env_soc
    
    # 2. 尝试通过 npu-smi 检测
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
            # 解析 npu-smi 输出，查找 SoC 信息
            if "Ascend910B3" in output:
                print("Detected SoC: Ascend910B3 (from npu-smi)")
                return "Ascend910B3"
            elif "Ascend910B2" in output:
                print("Detected SoC: Ascend910B2 (from npu-smi)")
                return "Ascend910B2"
            elif "Ascend910B1" in output:
                print("Detected SoC: Ascend910B1 (from npu-smi)")
                return "Ascend910B1"
    except Exception as e:
        print(f"Failed to detect SoC via npu-smi: {e}")
    
    # 3. 使用默认值
    default_soc = "Ascend910B2"
    print(f"Using default SoC version: {default_soc}")
    return default_soc

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
                    created_at = float(created_at_file.read_text(encoding='utf-8'))
                    if current_time - created_at > TASK_TTL:
                        shutil.rmtree(task_dir)
                        print(f"Cleaned up expired task: {task_dir.name}")
                except (ValueError, IOError):
                    pass

# ==================== API 端点 ====================

@app.post("/api/upload_task", 
          summary="上传算子任务",
          description="""
上传算子代码到远程服务器，创建新的评估任务。

**请求示例:**

```bash
curl -X POST "http://localhost:8080/api/upload_task" \\
  -H "Content-Type: application/json" \\
  -d '{
    "task_name": "31_ELU",
    "model_py": "import torch\\ntorch.nn.functional.elu(x)",
    "kernel_files": {
      "elu_kernel.cpp": "#include <ascendc.h>...",
      "pybind11.cpp": "PYBIND11_MODULE(elu_ext, m)..."
    },
    "soc_version": "Ascend910B2",
    "npu_id": 0,
    "clean_build": true
  }'
```

**响应示例:**

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "uploaded",
  "message": "Task uploaded successfully",
  "soc_version": "Ascend910B2",
  "auto_detected": false
}
```

**注意:**
- `soc_version` 可选，不指定则服务器自动检测
- `kernel_files` 必须包含 `pybind11.cpp`
- 单个文件最大 50MB，总大小最大 200MB
          """)
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
        
        # 自动检测 SoC 版本（如果未指定）
        soc_version = data.soc_version if data.soc_version else detect_soc_version()
        
        # 使用 NPU 调度器分配设备（支持并发）
        allocated_npu = npu_scheduler.allocate_npu(data.npu_id if data.npu_id >= 0 else None)
        
        # 创建任务目录
        task_id = str(uuid.uuid4())
        task_dir = TASKS_DIR / task_id
        task_dir.mkdir(parents=True)
        
        # 保存创建时间
        (task_dir / ".created_at").write_text(str(datetime.now().timestamp()), encoding='utf-8')
        
        # 写入 model.py（强制使用 UTF-8 编码）
        (task_dir / "model.py").write_text(data.model_py, encoding='utf-8')
        
        # 写入 model_new_ascendc.py (if provided)
        if data.model_new_ascendc:
            (task_dir / "model_new_ascendc.py").write_text(data.model_new_ascendc, encoding='utf-8')
        
        # 写入 kernel 文件（强制使用 UTF-8 编码）
        if data.kernel_files:
            kernel_dir = task_dir / "kernel"
            kernel_dir.mkdir(exist_ok=True)
            for filename, content in data.kernel_files.items():
                (kernel_dir / filename).write_text(content, encoding='utf-8')
        
        # 复制工具脚本
        utils_dir = task_dir / "utils"
        utils_dir.mkdir(exist_ok=True)
        
        for script in ["build_ascendc.py", "verification_ascendc.py", "performance.py"]:
            src = PROJECT_ROOT / "utils" / script
            if src.exists():
                shutil.copy2(src, utils_dir / script)
        
        # 保存配置（使用 UTF-8 编码）
        config = {
            "task_name": data.task_name,
            "soc_version": soc_version,
            "npu_id": allocated_npu,  # 使用调度器分配的 NPU
            "preferred_npu": data.npu_id,  # 记录用户首选
            "clean_build": data.clean_build,
            "created_at": datetime.now().isoformat(),
            "detected_soc": soc_version != data.soc_version  # 标记是否自动检测
        }
        (task_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
        
        return {
            "task_id": task_id,
            "status": "uploaded",
            "message": "Task uploaded successfully",
            "soc_version": soc_version,
            "allocated_npu": allocated_npu,
            "auto_detected": soc_version != data.soc_version
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/build",
          summary="编译 AscendC Kernel",
          description="""
编译上传的 AscendC kernel 代码。

**请求示例:**

```bash
curl -X POST "http://localhost:8080/api/build" \\
  -H "Content-Type: application/json" \\
  -d '{
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "soc_version": "Ascend910B2",
    "clean": true
  }'
```

**响应示例 (成功):**

```json
{
  "success": true,
  "logs": "[build_ascendc] Running: cmake...\nBuild completed",
  "error": null
}
```

**响应示例 (失败):**

```json
{
  "success": false,
  "logs": "Error: syntax error in elu_kernel.cpp:42",
  "error": "Compilation failed"
}
```

**注意:**
- 编译超时时间为 300 秒
- `clean=true` 会删除之前的 build 目录
          """)
async def build_kernel(request: BuildRequest):
    """编译 AscendC kernel"""
    task_dir = TASKS_DIR / request.task_id
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail="Task not found")
    
    config_file = task_dir / "config.json"
    if config_file.exists():
        config = json.loads(config_file.read_text(encoding='utf-8'))
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

@app.post("/api/verify",
          summary="验证算子精度",
          description="""
验证 AscendC 实现与 PyTorch 参考实现的精度对比。

**请求示例:**

```bash
curl -X POST "http://localhost:8080/api/verify" \\
  -H "Content-Type: application/json" \\
  -d '{"task_id": "550e8400-e29b-41d4-a716-446655440000"}'
```

**响应示例 (通过):**

```json
{
  "passed": true,
  "output": "================================================================\nStatus    : PASS\n...",
  "error": null,
  "comparison": "case[0]: matched"
}
```

**响应示例 (失败):**

```json
{
  "passed": false,
  "output": "...",
  "error": "Accuracy verification failed",
  "comparison": "case[0]: max_abs_diff=0.05, mismatch_ratio=2.5%"
}
```

**注意:**
- 默认容差: atol=1e-2, rtol=1e-2
- int8 类型自动使用更宽松容差: atol=1.5
- 验证超时时间为 120 秒
          """)
async def verify_accuracy(request: VerifyRequest):
    """验证算子精度"""
    task_dir = TASKS_DIR / request.task_id
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail="Task not found")
    
    config_file = task_dir / "config.json"
    config = json.loads(config_file.read_text(encoding='utf-8')) if config_file.exists() else {}
    
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

@app.post("/api/benchmark",
          summary="性能基准测试",
          description="""
执行性能基准测试，测量算子延迟。

**请求示例:**

```bash
curl -X POST "http://localhost:8080/api/benchmark" \\
  -H "Content-Type: application/json" \\
  -d '{
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "impl": "ascendc",
    "warmup": 5,
    "repeat": 10,
    "seed": 0
  }'
```

**响应示例:**

```json
{
  "success": true,
  "output": "================================================================\nPerformance Report\n...\nascendc      OK       1.234      1.200 ...",
  "error": null
}
```

**参数说明:**
- `impl`: 测试的实现类型 (reference/tilelang/ascendc)
- `warmup`: 预热次数，默认 5
- `repeat`: 重复次数，默认 10
- `seed`: 随机种子，默认 0

**注意:**
- 性能测试超时时间为 180 秒
- 建议先通过精度验证再进行性能测试
          """)
async def benchmark_performance(request: BenchmarkRequest):
    """性能基准测试"""
    task_dir = TASKS_DIR / request.task_id
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail="Task not found")
    
    config_file = task_dir / "config.json"
    config = json.loads(config_file.read_text(encoding='utf-8')) if config_file.exists() else {}
    
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

@app.post("/api/execute_command",
          summary="执行自定义命令",
          description="""
在任务目录中执行自定义命令，支持上传脚本。

**请求示例 1: 执行简单命令**

```bash
curl -X POST "http://localhost:8080/api/execute_command" \\
  -H "Content-Type: application/json" \\
  -d '{
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "command": "python my_script.py --arg1 value1",
    "timeout": 300
  }'
```

**请求示例 2: 上传并执行脚本**

```bash
curl -X POST "http://localhost:8080/api/execute_command" \\
  -H "Content-Type: application/json" \\
  -d '{
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "command": "python custom_validator.py",
    "scripts": {
      "custom_validator.py": "import sys\\nprint(\'Validating...\')"
    },
    "env_vars": {"DEBUG": "1"}
  }'
```

**响应示例:**

```json
{
  "success": true,
  "returncode": 0,
  "stdout": "Validation passed",
  "stderr": "",
  "command": "python custom_validator.py"
}
```

**安全限制:**
- 禁止危险命令: rm -rf /, sudo, mkfs, dd 等
- 命令在任务目录中执行
- 超时后自动终止

**注意:**
- 脚本会被上传到任务目录
- 环境变量会与系统环境变量合并
          """)
async def execute_custom_command(request: CustomCommandRequest):
    """执行自定义命令"""
    task_dir = TASKS_DIR / request.task_id
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail="Task not found")
    
    try:
        # 上传自定义脚本（强制使用 UTF-8 编码）
        for filename, content in request.scripts.items():
            script_path = task_dir / filename
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text(content, encoding='utf-8')
        
        # 准备环境变量
        env = os.environ.copy()
        config_file = task_dir / "config.json"
        if config_file.exists():
            config = json.loads(config_file.read_text(encoding='utf-8'))
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
        config = json.loads(config_file.read_text(encoding='utf-8'))
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

@app.get("/api/npu_status")
async def get_npu_status():
    """
    查询 NPU 调度器状态
    
    返回当前 NPU 设备的负载情况，用于监控和调试。
    
    **响应示例:**
    
    ```json
    {
      "num_npus": 8,
      "npu_load": {
        "0": 2,
        "1": 1,
        "2": 0,
        "3": 3,
        "4": 0,
        "5": 1,
        "6": 0,
        "7": 0
      },
      "total_tasks": 7
    }
    ```
    """
    return npu_scheduler.get_status()

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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AscendC Remote Evaluation Server")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"),
                        help="Host to bind to")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")),
                        help="Port to listen on")
    
    args = parser.parse_args()
    
    print(f"Starting server on {args.host}:{args.port}...")
    print(f"Server URL: http://{args.host}:{args.port}")
    print()
    
    uvicorn.run(app, host=args.host, port=args.port)
