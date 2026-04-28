#!/usr/bin/env python3
"""
AscendC MCP Client - 简易调用脚本

用法:
    python mcp_client.py upload --task-name ELU --model model.py --kernel-dir kernel/
    python mcp_client.py build --task-id <task_id>
    python mcp_client.py verify --task-id <task_id>
    python mcp_client.py benchmark --task-id <task_id>
    python mcp_client.py full-eval --task-name ELU --model model.py --kernel-dir kernel/
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

# 添加项目根目录
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import httpx
except ImportError:
    print("Error: httpx is required. Install with: pip install httpx")
    sys.exit(1)


class AscendCMCPClient:
    """AscendC MCP 客户端"""
    
    def __init__(self, server_url: str = None):
        """
        初始化客户端
        
        Args:
            server_url: 远程服务器地址，默认从环境变量读取或使用 http://localhost:9002
        """
        import os
        self.server_url = server_url or os.environ.get(
            "REMOTE_SERVER_URL",
            "http://localhost:9002"
        )
        self.client = httpx.Client(timeout=600)
    
    def _call_api(self, endpoint: str, data: dict = None, method: str = "POST"):
        """调用 API"""
        url = f"{self.server_url}{endpoint}"
        
        try:
            if method == "POST":
                response = self.client.post(url, json=data)
            else:
                response = self.client.get(url)
            
            response.raise_for_status()
            return response.json()
        
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json().get("detail", str(e))
            print(f"❌ API Error: {error_detail}")
            raise
        except Exception as e:
            print(f"❌ Request Failed: {str(e)}")
            raise
    
    def upload_task(
        self,
        task_name: str,
        model_py_path: str,
        kernel_dir: str,
        soc_version: str = None,
        npu_id: int = None,
        clean_build: bool = True
    ) -> dict:
        """
        上传任务
        
        Args:
            task_name: 任务名称
            model_py_path: model.py 文件路径
            kernel_dir: kernel 目录路径
            soc_version: SoC 版本（可选，自动检测）
            npu_id: NPU 设备 ID（可选，自动分配）
            clean_build: 是否清理后编译
        
        Returns:
            包含 task_id 的响应字典
        """
        print(f"📤 Uploading task: {task_name}")
        
        # 读取文件
        model_path = Path(model_py_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_py_path}")

        model_py = model_path.read_text(encoding='utf-8')

        # 读取 model_new_ascendc.py（如果存在）
        model_ascendc_path = model_path.parent / "model_new_ascendc.py"
        model_new_ascendc = None
        if model_ascendc_path.exists():
            model_new_ascendc = model_ascendc_path.read_text(encoding='utf-8')

        # 读取 kernel 文件
        kernel_path = Path(kernel_dir)
        if not kernel_path.exists():
            raise FileNotFoundError(f"Kernel directory not found: {kernel_dir}")

        kernel_files = {}
        for pattern in ("*.cpp", "*.h"):
            for file in kernel_path.glob(pattern):
                if file.is_file():
                    kernel_files[file.name] = file.read_text(encoding='utf-8')

        if not kernel_files:
            raise ValueError(f"No .cpp files found in {kernel_dir}")

        if "pybind11.cpp" not in kernel_files:
            raise ValueError("pybind11.cpp not found in kernel directory")

        # 构建请求数据
        data = {
            "task_name": task_name,
            "model_py": model_py,
            "kernel_files": kernel_files,
            "clean_build": clean_build
        }

        if model_new_ascendc:
            data["model_new_ascendc"] = model_new_ascendc
        
        if soc_version:
            data["soc_version"] = soc_version
        
        if npu_id is not None:
            data["npu_id"] = npu_id
        
        # 调用 API
        result = self._call_api("/api/upload_task", data)
        
        print(f"✅ Upload successful!")
        print(f"   Task ID: {result['task_id']}")
        print(f"   SoC Version: {result['soc_version']}")
        print(f"   Allocated NPU: {result['allocated_npu']}")
        
        return result
    
    def build_kernel(
        self,
        task_id: str,
        soc_version: str = None,
        clean: bool = True
    ) -> dict:
        """
        编译 Kernel
        
        Args:
            task_id: 任务 ID
            soc_version: SoC 版本（可选）
            clean: 是否清理后编译
        
        Returns:
            编译结果
        """
        print(f"🔨 Building kernel: {task_id}")
        
        data = {
            "task_id": task_id,
            "clean": clean
        }
        
        if soc_version:
            data["soc_version"] = soc_version
        
        result = self._call_api("/api/build", data)
        
        if result.get("success"):
            print(f"✅ Build successful!")
        else:
            print(f"❌ Build failed!")
            print(f"   Error: {result.get('error', 'Unknown')}")
        
        return result
    
    def verify_accuracy(self, task_id: str) -> dict:
        """
        验证精度
        
        Args:
            task_id: 任务 ID
        
        Returns:
            验证结果
        """
        print(f"✓ Verifying accuracy: {task_id}")
        
        data = {"task_id": task_id}
        result = self._call_api("/api/verify", data)
        
        if result.get("success"):
            passed = result.get("passed", False)
            if passed:
                print(f"✅ Verification PASSED!")
            else:
                print(f"❌ Verification FAILED!")
                print(f"   Output: {result.get('output', '')[:500]}")
        else:
            print(f"❌ Verification error: {result.get('error', 'Unknown')}")
        
        return result
    
    def benchmark_performance(
        self,
        task_id: str,
        impl: str = "ascendc",
        warmup: int = 5,
        repeat: int = 10,
        seed: int = 0
    ) -> dict:
        """
        性能测试
        
        Args:
            task_id: 任务 ID
            impl: 实现类型 (reference/tilelang/ascendc)
            warmup: 预热次数
            repeat: 重复次数
            seed: 随机种子
        
        Returns:
            性能测试结果
        """
        print(f"⚡ Running benchmark: {task_id} ({impl})")
        
        data = {
            "task_id": task_id,
            "impl": impl,
            "warmup": warmup,
            "repeat": repeat,
            "seed": seed
        }
        
        result = self._call_api("/api/benchmark", data)
        
        if result.get("success"):
            print(f"✅ Benchmark completed!")
            output = result.get("output", "")
            # 提取关键信息
            if "Mean(ms)" in output:
                for line in output.split('\n'):
                    if impl in line and ('OK' in line or 'ERROR' in line):
                        print(f"   {line.strip()}")
        else:
            print(f"❌ Benchmark failed: {result.get('error', 'Unknown')}")
        
        return result
    
    def get_task_status(self, task_id: str) -> dict:
        """
        查询任务状态
        
        Args:
            task_id: 任务 ID
        
        Returns:
            任务状态
        """
        result = self._call_api(f"/api/task_status/{task_id}", method="GET")
        
        if result.get("success"):
            print(f"📊 Task Status:")
            print(f"   Task Name: {result.get('task_name', 'N/A')}")
            print(f"   Created At: {result.get('created_at', 'N/A')}")
            print(f"   SoC Version: {result.get('soc_version', 'N/A')}")
            print(f"   Allocated NPU: {result.get('allocated_npu', 'N/A')}")
            print(f"   Files:")
            for key, exists in result.get('files', {}).items():
                status = "✓" if exists else "✗"
                print(f"     {status} {key}")
        
        return result
    
    def execute_command(
        self,
        task_id: str,
        command: str,
        scripts: dict = None,
        timeout: int = 300,
        env_vars: dict = None
    ) -> dict:
        """
        执行自定义命令
        
        Args:
            task_id: 任务 ID
            command: 要执行的命令
            scripts: 需要上传的脚本 {filename: content}
            timeout: 超时时间（秒）
            env_vars: 额外环境变量
        
        Returns:
            命令执行结果
        """
        print(f"⚙️  Executing command: {command[:50]}...")
        
        data = {
            "task_id": task_id,
            "command": command,
            "timeout": timeout
        }
        
        if scripts:
            data["scripts"] = scripts
        
        if env_vars:
            data["env_vars"] = env_vars
        
        result = self._call_api("/api/execute_command", data)
        
        if result.get("success"):
            print(f"✅ Command executed successfully!")
            print(f"   Return Code: {result.get('returncode', 0)}")
            if result.get("stdout"):
                print(f"   Output:\n{result['stdout'][:1000]}")
        else:
            print(f"❌ Command failed!")
            print(f"   Error: {result.get('error', 'Unknown')}")
            if result.get("stderr"):
                print(f"   Stderr:\n{result['stderr'][:1000]}")
        
        return result
    
    def download_results(self, task_id: str, output_dir: str = ".") -> str:
        """
        下载任务结果
        
        Args:
            task_id: 任务 ID
            output_dir: 输出目录
        
        Returns:
            下载的 zip 文件路径
        """
        print(f"📥 Downloading results: {task_id}")
        
        url = f"{self.server_url}/api/download_results/{task_id}"
        response = self.client.get(url, timeout=600)
        response.raise_for_status()
        
        # 保存文件
        output_path = Path(output_dir) / f"{task_id}_results.zip"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)
        
        print(f"✅ Results saved to: {output_path}")
        return str(output_path)
    
    def full_evaluation(
        self,
        task_name: str,
        model_py_path: str,
        kernel_dir: str,
        soc_version: str = None,
        npu_id: int = None,
        clean_build: bool = True,
        run_benchmark: bool = True,
        benchmark_impl: str = "ascendc"
    ) -> dict:
        """
        完整评估流程（上传 → 编译 → 验证 → 性能测试）
        
        Args:
            task_name: 任务名称
            model_py_path: model.py 文件路径
            kernel_dir: kernel 目录路径
            soc_version: SoC 版本
            npu_id: NPU 设备 ID
            clean_build: 是否清理后编译
            run_benchmark: 是否运行性能测试
            benchmark_impl: 性能测试的实现类型
        
        Returns:
            完整的评估结果
        """
        print("=" * 80)
        print("🚀 Starting Full Evaluation Pipeline")
        print("=" * 80)
        
        results = {
            "task_name": task_name,
            "steps": {}
        }
        
        try:
            # Step 1: Upload
            print("\n[1/4] Uploading task...")
            upload_result = self.upload_task(
                task_name=task_name,
                model_py_path=model_py_path,
                kernel_dir=kernel_dir,
                soc_version=soc_version,
                npu_id=npu_id,
                clean_build=clean_build
            )
            task_id = upload_result["task_id"]
            results["task_id"] = task_id
            results["steps"]["upload"] = upload_result
            
            # Step 2: Build
            print("\n[2/4] Building kernel...")
            build_result = self.build_kernel(
                task_id=task_id,
                soc_version=soc_version,
                clean=clean_build
            )
            results["steps"]["build"] = build_result
            
            if not build_result.get("success"):
                print("\n❌ Build failed! Stopping pipeline.")
                return results
            
            # Step 3: Verify
            print("\n[3/4] Verifying accuracy...")
            verify_result = self.verify_accuracy(task_id=task_id)
            results["steps"]["verify"] = verify_result
            
            # Step 4: Benchmark (optional)
            if run_benchmark:
                print("\n[4/4] Running benchmark...")
                benchmark_result = self.benchmark_performance(
                    task_id=task_id,
                    impl=benchmark_impl
                )
                results["steps"]["benchmark"] = benchmark_result
            
            print("\n" + "=" * 80)
            print("✅ Full Evaluation Completed!")
            print("=" * 80)
            print(f"\nTask ID: {task_id}")
            print(f"Upload: {'✓' if upload_result.get('success') else '✗'}")
            print(f"Build: {'✓' if build_result.get('success') else '✗'}")
            print(f"Verify: {'✓' if verify_result.get('success') and verify_result.get('passed') else '✗'}")
            if run_benchmark:
                print(f"Benchmark: {'✓' if benchmark_result.get('success') else '✗'}")
            
            return results
        
        except Exception as e:
            print(f"\n❌ Pipeline failed: {str(e)}")
            results["error"] = str(e)
            return results
    
    def close(self):
        """关闭客户端"""
        self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="AscendC MCP Client - 简易调用工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 完整评估（推荐）
  python mcp_client.py full-eval \\
    --task-name ELU \\
    --model tasks/elu_migration/model.py \\
    --kernel-dir tasks/elu_migration/kernel/

  # 分步执行
  python mcp_client.py upload --task-name ELU --model model.py --kernel-dir kernel/
  python mcp_client.py build --task-id <task_id>
  python mcp_client.py verify --task-id <task_id>
  python mcp_client.py benchmark --task-id <task_id>

  # 查询状态
  python mcp_client.py status --task-id <task_id>

  # 下载结果
  python mcp_client.py download --task-id <task_id> --output ./results/

  # 执行自定义命令
  python mcp_client.py exec --task-id <task_id> --command "ls -la"
        """
    )
    
    # 全局参数
    parser.add_argument(
        "--server-url",
        default=None,
        help="远程服务器地址 (默认: http://localhost:9002)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # upload 命令
    upload_parser = subparsers.add_parser("upload", help="上传任务")
    upload_parser.add_argument("--task-name", required=True, help="任务名称")
    upload_parser.add_argument("--model", required=True, help="model.py 文件路径")
    upload_parser.add_argument("--kernel-dir", required=True, help="kernel 目录路径")
    upload_parser.add_argument("--soc-version", default=None, help="SoC 版本")
    upload_parser.add_argument("--npu-id", type=int, default=None, help="NPU 设备 ID")
    upload_parser.add_argument("--no-clean", action="store_true", help="不清理后编译")
    
    # build 命令
    build_parser = subparsers.add_parser("build", help="编译 Kernel")
    build_parser.add_argument("--task-id", required=True, help="任务 ID")
    build_parser.add_argument("--soc-version", default=None, help="SoC 版本")
    build_parser.add_argument("--no-clean", action="store_true", help="不清理后编译")
    
    # verify 命令
    verify_parser = subparsers.add_parser("verify", help="验证精度")
    verify_parser.add_argument("--task-id", required=True, help="任务 ID")
    
    # benchmark 命令
    bench_parser = subparsers.add_parser("benchmark", help="性能测试")
    bench_parser.add_argument("--task-id", required=True, help="任务 ID")
    bench_parser.add_argument("--impl", default="ascendc", 
                             choices=["reference", "tilelang", "ascendc"],
                             help="实现类型")
    bench_parser.add_argument("--warmup", type=int, default=5, help="预热次数")
    bench_parser.add_argument("--repeat", type=int, default=10, help="重复次数")
    bench_parser.add_argument("--seed", type=int, default=0, help="随机种子")
    
    # status 命令
    status_parser = subparsers.add_parser("status", help="查询任务状态")
    status_parser.add_argument("--task-id", required=True, help="任务 ID")
    
    # download 命令
    download_parser = subparsers.add_parser("download", help="下载结果")
    download_parser.add_argument("--task-id", required=True, help="任务 ID")
    download_parser.add_argument("--output", default=".", help="输出目录")
    
    # exec 命令
    exec_parser = subparsers.add_parser("exec", help="执行自定义命令")
    exec_parser.add_argument("--task-id", required=True, help="任务 ID")
    exec_parser.add_argument("--command", required=True, help="要执行的命令")
    exec_parser.add_argument("--timeout", type=int, default=300, help="超时时间（秒）")
    
    # full-eval 命令（推荐）
    full_parser = subparsers.add_parser("full-eval", help="完整评估流程（推荐）")
    full_parser.add_argument("--task-name", required=True, help="任务名称")
    full_parser.add_argument("--model", required=True, help="model.py 文件路径")
    full_parser.add_argument("--kernel-dir", required=True, help="kernel 目录路径")
    full_parser.add_argument("--soc-version", default=None, help="SoC 版本")
    full_parser.add_argument("--npu-id", type=int, default=None, help="NPU 设备 ID")
    full_parser.add_argument("--no-clean", action="store_true", help="不清理后编译")
    full_parser.add_argument("--no-benchmark", action="store_true", help="不运行性能测试")
    full_parser.add_argument("--impl", default="ascendc",
                            choices=["reference", "tilelang", "ascendc"],
                            help="性能测试的实现类型")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # 创建客户端
    client = AscendCMCPClient(server_url=args.server_url)
    
    try:
        if args.command == "upload":
            result = client.upload_task(
                task_name=args.task_name,
                model_py_path=args.model,
                kernel_dir=args.kernel_dir,
                soc_version=args.soc_version,
                npu_id=args.npu_id,
                clean_build=not args.no_clean
            )
            print("\n💡 提示: 使用以下 task_id 进行后续操作:")
            print(f"   task_id={result['task_id']}")
        
        elif args.command == "build":
            result = client.build_kernel(
                task_id=args.task_id,
                soc_version=args.soc_version,
                clean=not args.no_clean
            )
        
        elif args.command == "verify":
            result = client.verify_accuracy(task_id=args.task_id)
        
        elif args.command == "benchmark":
            result = client.benchmark_performance(
                task_id=args.task_id,
                impl=args.impl,
                warmup=args.warmup,
                repeat=args.repeat,
                seed=args.seed
            )
        
        elif args.command == "status":
            result = client.get_task_status(task_id=args.task_id)
        
        elif args.command == "download":
            output_path = client.download_results(
                task_id=args.task_id,
                output_dir=args.output
            )
            print(f"\n💡 结果已保存到: {output_path}")
        
        elif args.command == "exec":
            result = client.execute_command(
                task_id=args.task_id,
                command=args.command,
                timeout=args.timeout
            )
        
        elif args.command == "full-eval":
            result = client.full_evaluation(
                task_name=args.task_name,
                model_py_path=args.model,
                kernel_dir=args.kernel_dir,
                soc_version=args.soc_version,
                npu_id=args.npu_id,
                clean_build=not args.no_clean,
                run_benchmark=not args.no_benchmark,
                benchmark_impl=args.impl
            )
            
            # 保存结果为 JSON
            if result.get("task_id"):
                json_file = f"{result['task_id']}_evaluation.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"\n💡 详细结果已保存到: {json_file}")
        
        # 打印 JSON 结果（如果需要）
        if hasattr(args, 'json') and args.json:
            print("\n📄 JSON Result:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
    
    finally:
        client.close()


if __name__ == "__main__":
    main()
