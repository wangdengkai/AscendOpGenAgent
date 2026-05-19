#!/usr/bin/env python3
"""
ops仓算子 baseline vs evolved 直接对比评估工具。

不依赖 reference.py / custom.py / AscendBackend / PyBind。
直接通过 torch.ops.npu/custom 调用算子，用 call_spec.json 描述调用规格。

核心流程：
1. 从 call_spec.json 读取算子调用规格（函数名、输入 shape/dtype、标量参数）
2. 生成输入张量并保存（确保 baseline 和 evolved 用同一份输入）
3. baseline 子进程：设置 OPP → 调用算子 → profiling → 保存输出
4. evolved 子进程：设置 OPP → 调用算子 → profiling → 保存输出
5. 主进程：加载两组输出 → allclose 精度对比 + 性能对比

用法:
    python evaluate_ops_direct.py {op_name} \\
        --call-spec call_spec.json \\
        --baseline-path /path/to/baseline \\
        --evolved-path /path/to/evolved \\
        --device-id 0 \\
        --task-type vector \\
        --output evaluation_results.json
"""

import argparse
import fcntl
import json
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PERF_SCRIPT = SCRIPT_DIR.parent.parent / "ascendc-evaluation" / "scripts" / "AscendPerformanceTest.py"


def _acquire_eval_lock(lock_path: str, timeout: float = 300) -> int:
    """阻塞获取评估排队锁。返回 fd。"""
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT)
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except (BlockingIOError, OSError):
            if time.monotonic() >= deadline:
                os.close(fd)
                raise TimeoutError(
                    f"Failed to acquire eval lock {lock_path} within {timeout}s"
                )
            time.sleep(1)


def _release_eval_lock(fd: int):
    """释放评估排队锁。"""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    except OSError:
        pass


def detect_vendor_subdir(install_path: str) -> str:
    """检测安装后的 vendors 子目录名。"""
    vendors_dir = os.path.join(install_path, "vendors")
    if os.path.isdir(vendors_dir):
        subdirs = [d for d in os.listdir(vendors_dir)
                    if os.path.isdir(os.path.join(vendors_dir, d))
                    and (d.startswith("custom") or d.startswith("omni_custom"))]
        if subdirs:
            return subdirs[0]
    if os.path.isdir(os.path.join(vendors_dir, "customize")):
        return "customize"
    return "custom_nn"


def generate_inputs(call_spec: dict, output_dir: str) -> int:
    """根据 call_spec 生成输入张量并保存到文件。返回输入数量。"""
    import torch
    os.makedirs(output_dir, exist_ok=True)

    # Fixed seed for reproducible precision evaluation across variants
    torch.manual_seed(42)

    dtype_map = {
        "float16": torch.float16, "float32": torch.float32,
        "bfloat16": torch.bfloat16, "int32": torch.int32,
        "int64": torch.int64, "bool": torch.bool,
    }

    inputs = call_spec.get("inputs", [])
    for i, inp in enumerate(inputs):
        dtype = dtype_map.get(inp["dtype"], torch.float16)
        shape = inp["shape"]
        if dtype in (torch.int32, torch.int64):
            t = torch.randint(0, 10, shape, dtype=dtype)
        elif dtype == torch.bool:
            t = torch.randint(0, 2, shape, dtype=torch.uint8).to(torch.bool)
        else:
            t = torch.randn(shape, dtype=dtype)
        torch.save(t, os.path.join(output_dir, f"input_{i}.pt"))

    # 生成 tensor_kwargs 张量（实际序列长度等）
    tensor_kwargs = call_spec.get("tensor_kwargs", {})
    for kwarg_name, kwarg_spec in tensor_kwargs.items():
        dtype = dtype_map.get(kwarg_spec["dtype"], torch.int64)
        value = kwarg_spec.get("value", None)
        if value is not None:
            t = torch.tensor(value, dtype=dtype)
        else:
            shape = kwarg_spec["shape"]
            if dtype in (torch.int32, torch.int64):
                t = torch.randint(0, 10, shape, dtype=dtype)
            else:
                t = torch.randn(shape, dtype=dtype)
        torch.save(t, os.path.join(output_dir, f"kwarg_{kwarg_name}.pt"))

    # 保存 call_spec 到输入目录（子进程需要读取）
    with open(os.path.join(output_dir, "call_spec.json"), "w") as f:
        json.dump(call_spec, f, indent=2)

    return len(inputs)


def run_single_version(
    call_spec: dict,
    install_path: str,
    input_dir: str,
    device_id: int,
    task_type: str,
    profile_dir: str,
    num_trials: int,
    tag: str,
) -> dict:
    """在独立子进程中评估单个版本（baseline 或 evolved）。

    子进程中：设置 OPP → 加载输入 → 调用算子 → profiling → 保存输出。

    Args:
        call_spec: 算子调用规格
        install_path: OPP 安装路径
        input_dir: 预生成的输入张量目录
        device_id: NPU 设备 ID
        task_type: 算子类型
        profile_dir: profiling 输出目录
        num_trials: profiling 试验次数
        tag: 版本标签 ("baseline" / "evolved")

    Returns:
        dict: 评估结果
    """
    vendor_subdir = detect_vendor_subdir(install_path)
    opp_path = os.path.join(install_path, "vendors", vendor_subdir)
    lib_path = os.path.join(opp_path, "op_api", "lib")

    if not os.path.isdir(opp_path):
        return {
            "tag": tag,
            "error": f"OPP directory not found: {opp_path}",
            "precision_passed": False,
            "time_us": -1,
        }

    os.makedirs(profile_dir, exist_ok=True)

    # 序列化 call_spec 供子进程读取
    spec_json = json.dumps(call_spec)
    n_inputs = len(call_spec.get("inputs", []))
    scalar_args_json = json.dumps(call_spec.get("scalar_args", {}))
    tensor_kwargs_json = json.dumps(call_spec.get("tensor_kwargs", {}))
    op_namespace = call_spec["op_namespace"]
    op_func = call_spec["op_func"]
    is_omni_ops = call_spec.get("is_omni_ops", False)

    eval_script = textwrap.dedent(f"""\
        #!/usr/bin/env python3
        import os, sys, json, logging
        logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

        os.environ["ASCEND_CUSTOM_OPP_PATH"] = {repr(opp_path)}
        existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = {repr(lib_path)} + ":" + existing_ld
        os.environ["ASCEND_DEVICE_ID"] = str({device_id})

        # 添加 AscendPerformanceTest.py 所在目录到 sys.path
        perf_script_dir = {repr(str(PERF_SCRIPT.parent))}
        if perf_script_dir not in sys.path:
            sys.path.insert(0, perf_script_dir)

        from pathlib import Path
        import torch
        import torch_npu
        torch.npu.set_device({device_id})

        # omni-ops需要导入omni_custom_ops
        if {repr(is_omni_ops)}:
            try:
                import omni_custom_ops
                logging.info("omni_custom_ops imported successfully")
            except ImportError as e:
                logging.warning(f"Failed to import omni_custom_ops: {{e}}")
        # Fallback unconditional import for safety (some envs need it regardless)
        try:
            import omni_custom_ops
        except ImportError:
            pass
        from AscendPerformanceTest import AdvancedPerformanceEngine

        # 加载输入张量
        inputs = []
        for i in range({n_inputs}):
            t = torch.load(os.path.join({repr(input_dir)}, f"input_{{i}}.pt"),
                           weights_only=True)
            inputs.append(t.npu())

        scalar_args = json.loads({repr(scalar_args_json)})

        # 加载 tensor_kwargs 张量
        tensor_kwargs = {{}}
        tensor_kwargs_spec = json.loads({repr(tensor_kwargs_json)})
        for kwarg_name in tensor_kwargs_spec.keys():
            kwarg_path = os.path.join({repr(input_dir)}, f"kwarg_{{kwarg_name}}.pt")
            if os.path.exists(kwarg_path):
                t = torch.load(kwarg_path, weights_only=True)
                tensor_kwargs[kwarg_name] = t.npu()

        # 构造 callable
        op_fn = getattr(getattr(torch.ops, {repr(op_namespace)}), {repr(op_func)})
        def model(*args):
            # Handle positional scalar args for omni-ops flash attention
            scalar_args_copy = dict(scalar_args)
            pos_args = []
            for key in ["scale_value", "sparse_block_size"]:
                if key in scalar_args_copy:
                    pos_args.append(scalar_args_copy.pop(key))
            return op_fn(*args, *pos_args, **scalar_args_copy, **tensor_kwargs)

        result = {{
            "tag": {repr(tag)},
            "install_path": {repr(install_path)},
            "precision_passed": True,
            "correctness_message": "",
            "time_us": -1,
            "pipeline": {{}},
            "bottleneck": "unknown",
            "cv_pct": 0.0,
        }}

        try:
            # 性能评估
            engine = AdvancedPerformanceEngine(logging.getLogger("DirectEval"))
            profile_root = Path({repr(profile_dir)})
            median_time, perf_data, output_path, cv_pct = engine.warmup_and_measure(
                model=model, inputs=inputs, device_id={device_id},
                profile_root=profile_root, num_trials={num_trials},
                task_type={repr(task_type)}, model_tag={repr(tag)}
            )
            result["time_us"] = median_time if median_time is not None else -1
            result["cv_pct"] = cv_pct if cv_pct is not None else 0.0

            # 提取 pipeline 信息
            if perf_data and isinstance(perf_data, list):
                pipeline = {{}}
                for row in perf_data:
                    if isinstance(row, dict):
                        for key in ("aiv_mte2_ratio", "aiv_vec_ratio", "aiv_scalar_ratio",
                                    "aiv_mte3_ratio", "aiv_mte2_time(us)", "aiv_vec_time(us)",
                                    "aiv_scalar_time(us)", "aiv_mte3_time(us)",
                                    "aiv_icache_miss_rate", "cube_utilization(%)",
                                    "aic_mac_ratio", "aic_mte1_ratio", "aic_mte2_ratio"):
                            if key in row and row[key] is not None:
                                pipeline[key] = row[key]
                result["pipeline"] = pipeline

            tag_str = result["tag"]
            logging.info(f"[{{tag_str}}] 性能: {{median_time:.2f}}us, cv={{cv_pct:.1f}}%")

            # 保存输出张量（用于精度对比）
            with torch.no_grad():
                output = model(*inputs)
            torch.save(output, os.path.join({repr(profile_dir)}, "{tag}_output.pt"))
            logging.info(f"[{{tag_str}}] 输出已保存")

        except Exception as e:
            import traceback
            result["precision_passed"] = False
            result["correctness_message"] = f"评估异常: {{e}}"
            result["time_us"] = -1
            tag_str = result["tag"]
            logging.error(f"[{{tag_str}}] 评估异常: {{e}}\\n{{traceback.format_exc()}}")

        print("--- EVAL_RESULT_JSON ---")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("--- END_EVAL_RESULT_JSON ---")
    """)

    # 写入临时脚本并执行
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix=f"eval_direct_{tag}_",
        delete=False, dir=profile_dir,
    ) as f:
        f.write(eval_script)
        script_path = f.name

    try:
        logging.info(f"[{tag}] 启动评估子进程: {script_path}")
        env = os.environ.copy()
        env["ASCEND_CUSTOM_OPP_PATH"] = opp_path
        env["LD_LIBRARY_PATH"] = lib_path + ":" + env.get("LD_LIBRARY_PATH", "")
        env["ASCEND_DEVICE_ID"] = str(device_id)

        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=600, env=env,
        )

        stdout = proc.stdout
        if "--- EVAL_RESULT_JSON ---" in stdout:
            json_start = stdout.index("--- EVAL_RESULT_JSON ---") + len("--- EVAL_RESULT_JSON ---")
            json_end = stdout.index("--- END_EVAL_RESULT_JSON ---")
            return json.loads(stdout[json_start:json_end].strip())
        else:
            logging.error(
                f"[{tag}] 子进程未输出结果\n"
                f"stdout (last 1000):\n{stdout[-1000:]}\n"
                f"stderr (last 1000):\n{proc.stderr[-1000:]}"
            )
            return {
                "tag": tag, "error": f"子进程未输出结果: rc={proc.returncode}",
                "precision_passed": False, "time_us": -1,
            }

    except subprocess.TimeoutExpired:
        return {"tag": tag, "error": "评估超时", "precision_passed": False, "time_us": -1}
    except Exception as e:
        return {"tag": tag, "error": str(e), "precision_passed": False, "time_us": -1}


def compare_outputs(baseline_dir: str, evolved_dir: str, rtol: float = 1e-3, atol: float = 1e-3) -> tuple:
    """对比 baseline 和 evolved 的输出张量。

    Returns:
        (precision_passed, message)
    """
    import torch

    baseline_path = os.path.join(baseline_dir, "baseline_output.pt")
    evolved_path = os.path.join(evolved_dir, "evolved_output.pt")

    if not os.path.exists(baseline_path):
        return False, f"Baseline output not found: {baseline_path}"
    if not os.path.exists(evolved_path):
        return False, f"Evolved output not found: {evolved_path}"

    baseline_out = torch.load(baseline_path, weights_only=False, map_location="cpu")
    evolved_out = torch.load(evolved_path, weights_only=False, map_location="cpu")

    # 统一为 list
    if isinstance(baseline_out, torch.Tensor):
        baseline_out = [baseline_out]
    if isinstance(evolved_out, torch.Tensor):
        evolved_out = [evolved_out]
    if isinstance(baseline_out, tuple):
        baseline_out = list(baseline_out)
    if isinstance(evolved_out, tuple):
        evolved_out = list(evolved_out)

    if len(baseline_out) != len(evolved_out):
        return False, f"Output count mismatch: baseline={len(baseline_out)}, evolved={len(evolved_out)}"

    max_diff = 0.0
    for i, (b, e) in enumerate(zip(baseline_out, evolved_out)):
        if not isinstance(b, torch.Tensor) or not isinstance(e, torch.Tensor):
            continue
        bf, ef = b.float(), e.float()
        abs_diff = (bf - ef).abs()
        diff = abs_diff.max().item()
        max_diff = max(max_diff, diff)
        if not torch.allclose(bf, ef, rtol=rtol, atol=atol):
            atol_ok = (abs_diff <= atol).all().item()
            rtol_ok = (abs_diff <= rtol * ef.abs()).all().item()
            return False, (
                f"Output[{i}] mismatch: max_abs_diff={diff:.6f} "
                f"(atol_ok={'Y' if atol_ok else 'N'}, "
                f"rtol_ok={'Y' if rtol_ok else 'N'}, "
                f"rtol={rtol}, atol={atol})"
            )

    return True, f"All outputs match (max_abs_diff={max_diff:.6f})"


def compare_versions(baseline_result: dict, evolved_result: dict) -> dict:
    """计算 baseline 和 evolved 的对比指标。"""
    baseline_time = baseline_result.get("time_us", -1)
    evolved_time = evolved_result.get("time_us", -1)

    comparison = {"compilation_success": True}

    if baseline_time > 0 and evolved_time > 0:
        comparison["speedup"] = baseline_time / evolved_time
        comparison["time_delta_us"] = evolved_time - baseline_time
    else:
        comparison["speedup"] = 0.0
        comparison["time_delta_us"] = 0.0

    baseline_bn = baseline_result.get("bottleneck", "unknown")
    evolved_bn = evolved_result.get("bottleneck", "unknown")
    comparison["bottleneck_change"] = f"{baseline_bn} -> {evolved_bn}"

    evolved_cv = evolved_result.get("cv_pct", 0.0)
    comparison["cv_pct"] = evolved_cv
    if evolved_cv < 5.0:
        comparison["measurement_quality"] = "good"
    elif evolved_cv < 15.0:
        comparison["measurement_quality"] = "acceptable"
    else:
        comparison["measurement_quality"] = "noisy"

    return comparison


def evaluate_ops_direct(
    op_name: str,
    call_spec_path: str,
    baseline_path: str,
    evolved_path: str,
    device_id: int = 0,
    task_type: str = "vector",
    output_path: str = None,
    num_trials: int = 50,
    eval_lock: str = None,
    eval_lock_timeout: float = 300,
    baseline_cache: str = None,
) -> dict:
    """baseline vs evolved 直接对比评估。

    Args:
        op_name: 算子名
        call_spec_path: call_spec.json 路径
        baseline_path: baseline 安装路径
        evolved_path: evolved 安装路径
        device_id: NPU 设备 ID
        task_type: 算子类型
        output_path: 结果输出路径
        num_trials: profiling 试验次数
        eval_lock: 评估排队锁文件路径
        eval_lock_timeout: 评估锁等待超时秒数
        baseline_cache: baseline 评估结果缓存文件路径

    Returns:
        dict: 完整对比结果
    """
    with open(call_spec_path, "r", encoding="utf-8") as f:
        call_spec = json.load(f)

    # 确定工作目录
    work_dir = os.path.dirname(evolved_path) if evolved_path else "."
    input_dir = os.path.join(work_dir, "eval_inputs")
    baseline_profile_dir = os.path.join(work_dir, "baseline_profiling")
    evolved_profile_dir = os.path.join(work_dir, "evolved_profiling")

    # 生成输入张量（一次性，baseline 和 evolved 共享）
    logging.info(f"生成输入张量到: {input_dir}")
    generate_inputs(call_spec, input_dir)

    # 尝试加载 baseline 缓存
    baseline_result = None
    if baseline_cache and os.path.isfile(baseline_cache):
        try:
            with open(baseline_cache, "r", encoding="utf-8") as f:
                cached = json.load(f)
            cached_baseline = cached.get("baseline")
            if cached_baseline and cached_baseline.get("time_us", -1) > 0:
                baseline_result = cached_baseline
                logging.info(f"使用 baseline 缓存 (time_us={baseline_result['time_us']:.2f})")
        except (json.JSONDecodeError, OSError) as e:
            logging.warning(f"Baseline 缓存读取失败: {e}")

    # 获取评估锁
    lock_fd = None
    if eval_lock:
        logging.info(f"等待评估锁: {eval_lock} (超时 {eval_lock_timeout}s)")
        lock_fd = _acquire_eval_lock(eval_lock, eval_lock_timeout)
        logging.info("评估锁已获取")

    try:
        # 评估 baseline（无缓存时）
        if baseline_result is None:
            logging.info(f"评估 baseline: {baseline_path}")
            baseline_result = run_single_version(
                call_spec=call_spec, install_path=baseline_path,
                input_dir=input_dir, device_id=device_id,
                task_type=task_type, profile_dir=baseline_profile_dir,
                num_trials=num_trials, tag="baseline",
            )
        else:
            # 使用缓存时仍需运行 baseline 一次以生成 baseline_output.pt（用于精度对比）
            # 但跳过 profiling（只需要输出张量）
            logging.info(f"Baseline 使用缓存性能数据，仅运行一次获取输出张量")
            baseline_run = run_single_version(
                call_spec=call_spec, install_path=baseline_path,
                input_dir=input_dir, device_id=device_id,
                task_type=task_type, profile_dir=baseline_profile_dir,
                num_trials=3, tag="baseline",
            )
            # 保留缓存的性能数据，仅补充精度相关字段
            if baseline_run.get("error"):
                logging.warning(f"Baseline 输出生成失败: {baseline_run['error']}")

        # 评估 evolved
        logging.info(f"评估 evolved: {evolved_path}")
        evolved_result = run_single_version(
            call_spec=call_spec, install_path=evolved_path,
            input_dir=input_dir, device_id=device_id,
            task_type=task_type, profile_dir=evolved_profile_dir,
            num_trials=num_trials, tag="evolved",
        )
    finally:
        if lock_fd is not None:
            _release_eval_lock(lock_fd)
            logging.info("评估锁已释放")

    # 精度对比（baseline 输出 vs evolved 输出）
    precision_passed, precision_msg = compare_outputs(
        baseline_profile_dir, evolved_profile_dir
    )

    comparison = compare_versions(baseline_result, evolved_result)
    comparison["precision_passed"] = precision_passed
    comparison["precision_message"] = precision_msg

    final_result = {
        "op_name": op_name,
        "call_spec": os.path.basename(call_spec_path),
        "baseline": baseline_result,
        "evolved": evolved_result,
        "comparison": comparison,
    }

    if output_path is None:
        output_path = os.path.join(work_dir, "evaluation_results.json")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    logging.info(f"结果已保存: {output_path}")

    # 打印摘要
    speedup = comparison.get("speedup", 0)
    bt = baseline_result.get("time_us", -1)
    et = evolved_result.get("time_us", -1)
    print(f"\n{'='*60}")
    print(f"评估结果: {op_name}")
    print(f"{'='*60}")
    print(f"  精度: {'PASS' if precision_passed else 'FAIL'} — {precision_msg}")
    print(f"  Baseline: {bt:.2f} us")
    print(f"  Evolved:  {et:.2f} us")
    print(f"  Speedup:  {speedup:.3f}x")
    print(f"  Quality:  {comparison.get('measurement_quality', 'unknown')}")
    print(f"{'='*60}")

    return final_result


def main():
    parser = argparse.ArgumentParser(
        description="Direct baseline vs evolved evaluation for ops repository operators"
    )
    parser.add_argument("op_name", type=str, help="算子名")
    parser.add_argument("--call-spec", required=True, help="call_spec.json 路径")
    parser.add_argument("--baseline-path", required=True, help="baseline 安装路径")
    parser.add_argument("--evolved-path", required=True, help="evolved 安装路径")
    parser.add_argument("--device-id", type=int, default=0, help="NPU 设备 ID")
    parser.add_argument("--task-type", type=str, default="vector",
                        choices=["vector", "cube", "cv-mix", "unknown"])
    parser.add_argument("--output", type=str, default=None, help="结果输出路径")
    parser.add_argument("--num-trials", type=int, default=50, help="profiling 试验次数")
    parser.add_argument("--eval-lock", type=str, default=None, help="评估排队锁文件路径")
    parser.add_argument("--eval-lock-timeout", type=float, default=300)
    parser.add_argument("--baseline-cache", type=str, default=None,
                        help="baseline 评估结果缓存文件路径")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    evaluate_ops_direct(
        op_name=args.op_name,
        call_spec_path=args.call_spec,
        baseline_path=args.baseline_path,
        evolved_path=args.evolved_path,
        device_id=args.device_id,
        task_type=args.task_type,
        output_path=args.output,
        num_trials=args.num_trials,
        eval_lock=args.eval_lock,
        eval_lock_timeout=args.eval_lock_timeout,
        baseline_cache=args.baseline_cache,
    )


if __name__ == "__main__":
    main()
