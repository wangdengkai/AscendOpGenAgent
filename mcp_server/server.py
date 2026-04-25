#!/usr/bin/env python3
"""
AscendC Remote Evaluator - FastMCP Server
提供 MCP 工具供 Claude Code 调用，实现远程算子评估
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional

# 添加项目根目录
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from fastmcp import FastMCP
    import httpx
except ImportError:
    print("Error: fastmcp and httpx are required. Install with: pip install fastmcp httpx")
    sys.exit(1)

# ==================== 配置 ====================

REMOTE_SERVER_URL = os.environ.get(
    "REMOTE_SERVER_URL",
    "http://localhost:8080"
)

# 创建 MCP 实例
mcp = FastMCP("AscendC Remote Evaluator")

# ==================== 工具函数 ====================

async def call_remote_api(endpoint: str, data: dict = None, timeout: int = 600):
    """调用远程 API（带重试）"""
    from tenacity import retry, stop_after_attempt, wait_exponential
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def _call():
        async with httpx.AsyncClient(timeout=timeout) as client:
            if data:
                response = await client.post(
                    f"{REMOTE_SERVER_URL}{endpoint}",
                    json=data
                )
            else:
                response = await client.get(f"{REMOTE_SERVER_URL}{endpoint}")
            
            response.raise_for_status()
            return response.json()
    
    return await _call()

def read_file_content(file_path: str) -> str:
    """读取文件内容"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return path.read_text(encoding='utf-8')

def read_directory_files(dir_path: str, pattern: str = "*") -> Dict[str, str]:
    """读取目录下所有文件"""
    path = Path(dir_path)
    if not path.exists():
        return {}
    
    files = {}
    for file in path.glob(pattern):
        if file.is_file():
            files[file.name] = file.read_text(encoding='utf-8')
    
    return files

# ==================== MCP 工具 ====================

@mcp.tool()
async def upload_and_evaluate(
    task_name: str,
    model_py_path: str,
    kernel_dir: str,
    soc_version: Optional[str] = None,  # 改为可选，None 表示自动检测
    npu_id: Optional[int] = None,  # None 表示服务器自动分配
    clean_build: bool = True,
    enable_benchmark: bool = True
) -> Dict:
    """
    上传算子代码到远程服务器并执行完整评估流程
    
    ## 参数说明
    
    ### 必需参数
    - **task_name**: 任务名称（如 "31_ELU"）
    - **model_py_path**: model.py 文件路径
    - **kernel_dir**: kernel 目录路径
    
    ### 可选参数
    - **soc_version**: SoC 版本（可选，不指定则服务器自动检测）
      - 可选值: "Ascend910B1", "Ascend910B2", "Ascend910B3"
      - 默认: None (自动检测)
    - **npu_id**: NPU 设备 ID（可选，不指定则服务器自动分配）
      - None: 服务器自动选择负载最低的 NPU（推荐）
      - 0-7: 偏好使用指定 NPU（服务器可能根据负载调整）
      - 默认: None
    - **clean_build**: 是否清理后重新编译，默认 True
    - **enable_benchmark**: 是否执行性能测试，默认 True
    
    ## 返回结果
    
    成功时返回包含验证和性能结果的字典。
    失败时返回错误信息和解决建议。
    
    ## 使用示例
    
    ### 示例 1: 完全自动（推荐）
    ```python
    result = await upload_and_evaluate(
        task_name="31_ELU",
        model_py_path="tasks/31_ELU/model.py",
        kernel_dir="tasks/31_ELU/kernel"
        # npu_id 不指定，服务器自动分配最优设备
    )
    ```
    
    ### 示例 2: 偏好指定 NPU
    ```python
    result = await upload_and_evaluate(
        task_name="31_ELU",
        model_py_path="tasks/31_ELU/model.py",
        kernel_dir="tasks/31_ELU/kernel",
        npu_id=2  # 偏好使用 NPU 2，但服务器可能根据负载调整
    )
    ```
    """
    try:
        # 读取文件内容
        model_py_content = read_file_content(model_py_path)
        kernel_files = read_directory_files(kernel_dir)
        
        if not kernel_files:
            return {
                "status": "error",
                "error": {
                    "error_code": "NO_KERNEL_FILES",
                    "message": f"No files found in kernel directory: {kernel_dir}",
                    "suggestion": "Ensure kernel directory contains .cpp files and pybind11.cpp"
                }
            }
        
        # Step 1: 上传任务
        upload_data = {
            "task_name": task_name,
            "model_py": model_py_content,
            "kernel_files": kernel_files,
            "soc_version": soc_version,
            "npu_id": npu_id,
            "clean_build": clean_build
        }
        
        upload_result = await call_remote_api("/api/upload_task", upload_data)
        task_id = upload_result["task_id"]
        
        # Step 2: 编译
        build_result = await call_remote_api("/api/build", {
            "task_id": task_id,
            "soc_version": soc_version,
            "clean": clean_build
        })
        
        if not build_result["success"]:
            return {
                "status": "build_failed",
                "task_id": task_id,
                "error": {
                    "error_code": "BUILD_COMPILATION_ERROR",
                    "message": "Kernel compilation failed",
                    "suggestion": "Check AscendC syntax and API usage in kernel files",
                    "details": {
                        "logs": build_result.get("logs", "")
                    }
                }
            }
        
        # Step 3: 精度验证
        verify_result = await call_remote_api("/api/verify", {"task_id": task_id})
        
        if not verify_result["passed"]:
            return {
                "status": "verification_failed",
                "task_id": task_id,
                "error": {
                    "error_code": "VERIFY_ACCURACY_LOW",
                    "message": "Accuracy verification failed",
                    "suggestion": "Check numerical precision and algorithm implementation",
                    "details": {
                        "comparison": verify_result.get("comparison", "")
                    }
                }
            }
        
        # Step 4: 性能测试（可选）
        perf_result = None
        if enable_benchmark:
            perf_result = await call_remote_api("/api/benchmark", {
                "task_id": task_id,
                "impl": "ascendc",
                "warmup": 5,
                "repeat": 10
            })
        
        return {
            "status": "success",
            "task_id": task_id,
            "verification": {
                "passed": verify_result["passed"],
                "output": verify_result.get("output", "")
            },
            "performance": perf_result,
            "download_url": f"{REMOTE_SERVER_URL}/api/download_results/{task_id}"
        }
        
    except FileNotFoundError as e:
        return {
            "status": "error",
            "error": {
                "error_code": "FILE_NOT_FOUND",
                "message": str(e),
                "suggestion": "Check file paths and ensure they exist"
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": {
                "error_code": "UNKNOWN_ERROR",
                "message": str(e),
                "suggestion": "Check server connectivity and try again"
            }
        }

@mcp.tool()
async def remote_build_kernel(
    task_id: str,
    soc_version: str = "Ascend910B2",
    clean: bool = True
) -> Dict:
    """
    在远程服务器上编译 AscendC kernel
    
    ## 参数
    - **task_id**: 任务 ID（从 upload_task 获得）
    - **soc_version**: SoC 版本
    - **clean**: 是否清理后编译
    
    ## 返回
    编译结果，包含 success 状态和日志
    """
    try:
        result = await call_remote_api("/api/build", {
            "task_id": task_id,
            "soc_version": soc_version,
            "clean": clean
        })
        
        return {
            "status": "success" if result["success"] else "failed",
            "task_id": task_id,
            "logs": result.get("logs", ""),
            "error": result.get("error")
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@mcp.tool()
async def remote_verify_accuracy(task_id: str) -> Dict:
    """
    验证算子精度
    
    ## 参数
    - **task_id**: 任务 ID
    
    ## 返回
    验证结果，包含 passed 状态和对比信息
    """
    try:
        result = await call_remote_api("/api/verify", {"task_id": task_id})
        
        return {
            "status": "passed" if result["passed"] else "failed",
            "task_id": task_id,
            "passed": result["passed"],
            "comparison": result.get("comparison", ""),
            "error": result.get("error")
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@mcp.tool()
async def remote_benchmark(
    task_id: str,
    impl: str = "ascendc",
    warmup: int = 5,
    repeat: int = 10,
    seed: int = 0
) -> Dict:
    """
    性能基准测试
    
    ## 参数
    - **task_id**: 任务 ID
    - **impl**: 实现类型（reference/tilelang/ascendc）
    - **warmup**: 预热次数
    - **repeat**: 重复次数
    - **seed**: 随机种子
    
    ## 返回
    性能测试结果
    """
    try:
        result = await call_remote_api("/api/benchmark", {
            "task_id": task_id,
            "impl": impl,
            "warmup": warmup,
            "repeat": repeat,
            "seed": seed
        })
        
        return {
            "status": "success" if result["success"] else "failed",
            "task_id": task_id,
            "output": result.get("output", ""),
            "error": result.get("error")
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@mcp.tool()
async def execute_custom_command(
    task_id: str,
    command: str,
    scripts: Dict[str, str] = None,
    timeout: int = 300,
    env_vars: Dict[str, str] = None
) -> Dict:
    """
    在远程服务器上执行自定义命令
    
    ## 参数
    - **task_id**: 任务 ID
    - **command**: 要执行的命令
    - **scripts**: 需要上传的脚本 {文件名: 内容}
    - **timeout**: 超时时间（秒）
    - **env_vars**: 额外环境变量
    
    ## 返回
    命令执行结果
    
    ## 使用示例
    
    ### 执行自定义验证脚本
    ```python
    result = await execute_custom_command(
        task_id="abc-123",
        command="python my_validator.py --threshold 0.95",
        scripts={
            "my_validator.py": "print('custom validation')..."
        }
    )
    ```
    
    ### 运行性能剖析
    ```python
    result = await execute_custom_command(
        task_id="abc-123",
        command="nsys profile python benchmark.py",
        timeout=600
    )
    ```
    """
    try:
        result = await call_remote_api("/api/execute_command", {
            "task_id": task_id,
            "command": command,
            "scripts": scripts or {},
            "timeout": timeout,
            "env_vars": env_vars or {}
        })
        
        return {
            "status": "success" if result["success"] else "failed",
            "task_id": task_id,
            "returncode": result.get("returncode", -1),
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "error": result.get("error")
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@mcp.tool()
async def check_task_status(task_id: str) -> Dict:
    """
    查询远程任务状态
    
    ## 参数
    - **task_id**: 任务 ID
    
    ## 返回
    任务状态信息
    """
    try:
        result = await call_remote_api(f"/api/task_status/{task_id}")
        return result
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@mcp.resource("task://{task_id}/report")
async def get_task_report(task_id: str) -> str:
    """
    获取任务评估报告
    
    ## 参数
    - **task_id**: 任务 ID
    
    ## 返回
    任务报告文本
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{REMOTE_SERVER_URL}/api/download_results/{task_id}"
            )
            response.raise_for_status()
            return f"Results available for download at: {REMOTE_SERVER_URL}/api/download_results/{task_id}"
    except Exception as e:
        return f"Error fetching report: {str(e)}"

# ==================== 启动 ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AscendC Remote Evaluator MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="sse",
                        help="Transport mode: stdio (for Claude Code) or sse (standalone)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (for SSE mode)")
    parser.add_argument("--port", type=int, default=8089, help="Port to listen on (for SSE mode)")
    
    args = parser.parse_args()
    
    print(f"Starting AscendC Remote Evaluator MCP Server")
    print(f"Remote Server URL: {REMOTE_SERVER_URL}")
    print(f"Transport mode: {args.transport}")
    if args.transport == "sse":
        print(f"Listening on {args.host}:{args.port}")
    print(f"Available tools:")
    print(f"  - upload_and_evaluate")
    print(f"  - remote_build_kernel")
    print(f"  - remote_verify_accuracy")
    print(f"  - remote_benchmark")
    print(f"  - execute_custom_command")
    print(f"  - check_task_status")
    
    if args.transport == "sse":
        # Run with SSE transport for standalone mode
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        # Run with stdio transport for Claude Code integration
        mcp.run(transport="stdio")
