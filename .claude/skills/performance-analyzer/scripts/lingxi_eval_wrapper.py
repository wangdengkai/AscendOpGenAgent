#!/usr/bin/env python3
"""Lingxi operator evaluation wrapper.

Runs build + verify + benchmark for a task in agent_workdir/agent_workdir/
and outputs structured JSON results. Designed for use by the lingxi-evo agent.

Usage:
    python evolution/lingxi_eval_wrapper.py <task_name> [options]

Examples:
    python evolution/lingxi_eval_wrapper.py matmul_leakyrelu
    python evolution/lingxi_eval_wrapper.py _evo_variant_matmul_leakyrelu --soc Ascend910B3 --device 3
    python evolution/lingxi_eval_wrapper.py matmul_leakyrelu --workdir /path/to/agent_workdir/agent_workdir
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path


def _find_agent_workdir() -> Path:
    """Find agent_workdir from environment or known paths."""
    env_val = os.environ.get("CV_AGENT_WORKDIR")
    if env_val:
        p = Path(env_val) / "agent_workdir"
        if p.is_dir():
            return p
        p = Path(env_val)
        if (p / "utils").is_dir():
            return p

    candidates = [
        Path("/home/z00893531/wzz/agent_workdir/agent_workdir"),
    ]
    for c in candidates:
        if c.is_dir():
            return c

    return Path.cwd()


def run_build(task_name: str, workdir: Path, soc: str) -> dict:
    """Run build_ascendc.py and return result."""
    cmd = [
        sys.executable,
        str(workdir / "utils" / "build_ascendc.py"),
        task_name,
        "-v", soc,
        "--clean",
    ]
    result = subprocess.run(
        cmd, cwd=str(workdir),
        capture_output=True, text=True, timeout=600,
    )
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
        "stderr": result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr,
    }


def run_verify(task_name: str, workdir: Path) -> dict:
    """Run verification_ascendc.py and return result."""
    cmd = [
        sys.executable,
        str(workdir / "utils" / "verification_ascendc.py"),
        task_name,
    ]
    result = subprocess.run(
        cmd, cwd=str(workdir),
        capture_output=True, text=True, timeout=300,
    )
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
        "stderr": result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr,
    }


def _parse_performance_output(stdout: str) -> dict:
    """Parse performance.py text output into structured data."""
    perf = {
        "case_results": [],
        "impl_results": [],
        "overall_mean_ms": None,
        "overall_median_ms": None,
    }

    # Parse the summary table lines like:
    # ascendc      OK          0.322        0.320 ...
    table_pattern = re.compile(
        r"^(\w+)\s+(OK|ERROR)\s+"
        r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
        re.MULTILINE,
    )
    for m in table_pattern.finditer(stdout):
        impl_result = {
            "impl": m.group(1),
            "status": m.group(2),
            "mean_ms": float(m.group(3)),
            "median_ms": float(m.group(4)),
            "min_ms": float(m.group(5)),
            "max_ms": float(m.group(6)),
            "stdev_ms": float(m.group(7)),
        }
        perf["impl_results"].append(impl_result)

    # Parse per-case lines like:
    # case[0] mean=0.401 ms, median=0.397 ms, samples(ms): [0.410, 0.397, ...]
    case_pattern = re.compile(
        r"case\[(\d+)\]\s+mean=([\d.]+)\s*ms,\s*median=([\d.]+)\s*ms",
    )
    current_impl = None
    for line in stdout.splitlines():
        # Track which impl section we're in
        for impl_r in perf["impl_results"]:
            if line.strip().startswith(f"{impl_r['impl']} ->"):
                current_impl = impl_r["impl"]
                break

        m = case_pattern.search(line)
        if m and current_impl:
            case_result = {
                "impl": current_impl,
                "case_index": int(m.group(1)),
                "mean_ms": float(m.group(2)),
                "median_ms": float(m.group(3)),
            }
            perf["case_results"].append(case_result)

    # Extract ascendc-specific overall stats
    for impl_r in perf["impl_results"]:
        if impl_r["impl"] == "ascendc" and impl_r["status"] == "OK":
            perf["overall_mean_ms"] = impl_r["mean_ms"]
            perf["overall_median_ms"] = impl_r["median_ms"]
            break

    return perf


def run_performance(
    task_name: str, workdir: Path,
    impl: str = "ascendc",
    warmup: int = 5, repeat: int = 10, seed: int = 0,
    use_event_timing: bool = False,
    baseline_path: str = None,
    save_baseline_path: str = None,
    device_id: int = 0,
) -> dict:
    """Run performance benchmark and return parsed result.

    Args:
        use_event_timing: If True, use lingxi_perf_bench.py (torch.npu.Event
            device-side timing) instead of performance.py (wall-clock).
        baseline_path: Path to saved baseline JSON (only for lingxi_perf_bench).
        save_baseline_path: Save baseline per-case data (only for lingxi_perf_bench).
        device_id: NPU device ID (only for lingxi_perf_bench).
    """
    if use_event_timing:
        return _run_performance_event(
            task_name, workdir, impl, warmup, repeat, seed,
            baseline_path, save_baseline_path, device_id,
        )
    return _run_performance_legacy(task_name, workdir, impl, warmup, repeat, seed)


def _run_performance_legacy(
    task_name: str, workdir: Path,
    impl: str = "ascendc",
    warmup: int = 5, repeat: int = 10, seed: int = 0,
) -> dict:
    """Legacy: run performance.py with wall-clock timing."""
    cmd = [
        sys.executable,
        str(workdir / "utils" / "performance.py"),
        task_name, impl,
        str(warmup), str(repeat), str(seed),
    ]
    result = subprocess.run(
        cmd, cwd=str(workdir),
        capture_output=True, text=True, timeout=300,
    )
    parsed = {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout,
        "stderr": result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr,
        "timing_method": "perf_counter",
    }
    if result.returncode == 0:
        parsed["performance"] = _parse_performance_output(result.stdout)
    return parsed


def _run_performance_event(
    task_name: str, workdir: Path,
    impl: str = "ascendc",
    warmup: int = 10, repeat: int = 30, seed: int = 0,
    baseline_path: str = None,
    save_baseline_path: str = None,
    device_id: int = 0,
) -> dict:
    """Run lingxi_perf_bench.py with device-side event timing."""
    # Locate lingxi_perf_bench.py relative to this file
    bench_script = Path(__file__).resolve().parent / "lingxi_perf_bench.py"
    if not bench_script.is_file():
        return {"ok": False, "error": f"lingxi_perf_bench.py not found at {bench_script}",
                "timing_method": "npu_event"}

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_output = tmp.name

    cmd = [
        sys.executable, str(bench_script),
        task_name,
        "--impl", impl,
        "--warmup", str(warmup),
        "--repeat", str(repeat),
        "--seed", str(seed),
        "--device", str(device_id),
        "--workdir", str(workdir),
        "--output", tmp_output,
        "--quiet",
    ]
    if baseline_path:
        cmd.extend(["--baseline", baseline_path])
    if save_baseline_path:
        cmd.extend(["--save-baseline", save_baseline_path])

    result = subprocess.run(
        cmd, cwd=str(workdir),
        capture_output=True, text=True, timeout=600,
    )

    parsed = {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stderr": result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr,
        "timing_method": "npu_event",
    }

    tmp_path = Path(tmp_output)
    if tmp_path.is_file():
        try:
            bench_result = json.loads(tmp_path.read_text(encoding="utf-8"))
            parsed["performance"] = bench_result
        except (json.JSONDecodeError, OSError) as exc:
            parsed["performance_parse_error"] = str(exc)
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        parsed["ok"] = False
        parsed["error"] = "lingxi_perf_bench.py produced no output"

    return parsed


def evaluate(
    task_name: str,
    workdir: Path,
    soc: str = "Ascend910B3",
    device_id: int = 3,
    perf_impl: str = "ascendc",
    warmup: int = 5,
    repeat: int = 10,
    seed: int = 0,
    use_event_timing: bool = False,
    baseline_path: str = None,
    save_baseline_path: str = None,
) -> dict:
    """Run full evaluation pipeline: build → verify → benchmark.

    Args:
        use_event_timing: Use lingxi_perf_bench.py (torch.npu.Event) instead of
            performance.py (time.perf_counter). Recommended for accurate results.
        baseline_path: Path to saved baseline JSON for speedup computation.
        save_baseline_path: Save baseline per-case data to this path.
    """
    os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", str(device_id))

    report = {
        "task": task_name,
        "soc": soc,
        "device_id": device_id,
        "build_ok": False,
        "verify_ok": False,
        "performance": None,
        "build_error": None,
        "verify_error": None,
        "perf_error": None,
    }

    # Step 1: Build
    build_result = run_build(task_name, workdir, soc)
    report["build_ok"] = build_result["ok"]
    if not build_result["ok"]:
        report["build_error"] = build_result["stderr"] or build_result["stdout"]
        return report

    # Step 2: Verify
    verify_result = run_verify(task_name, workdir)
    report["verify_ok"] = verify_result["ok"]
    if not verify_result["ok"]:
        report["verify_error"] = verify_result["stdout"] or verify_result["stderr"]
        return report

    # Step 3: Performance
    perf_result = run_performance(
        task_name, workdir,
        impl=perf_impl, warmup=warmup, repeat=repeat, seed=seed,
        use_event_timing=use_event_timing,
        baseline_path=baseline_path,
        save_baseline_path=save_baseline_path,
        device_id=device_id,
    )
    if perf_result["ok"] and "performance" in perf_result:
        report["performance"] = perf_result["performance"]
    else:
        report["perf_error"] = perf_result.get("stderr") or perf_result.get("stdout")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Lingxi operator evaluation wrapper (build + verify + benchmark)",
    )
    parser.add_argument("task", help="Task directory name (under agent_workdir/)")
    parser.add_argument(
        "--workdir", type=str, default=None,
        help="Path to agent_workdir/ directory",
    )
    parser.add_argument("--soc", default="Ascend910B3", help="Ascend SoC version")
    parser.add_argument("--device", type=int, default=3, help="NPU device ID")
    parser.add_argument(
        "--impl", default="ascendc",
        help="Implementation to benchmark (ascendc/reference/all)",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Benchmark warmup iterations")
    parser.add_argument("--repeat", type=int, default=10, help="Benchmark repeat iterations")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--use-event-timing", action="store_true",
        help="Use lingxi_perf_bench.py (torch.npu.Event) for accurate device-side timing",
    )
    parser.add_argument(
        "--baseline", type=str, default=None,
        help="Path to saved baseline JSON for speedup computation (event timing only)",
    )
    parser.add_argument(
        "--save-baseline", type=str, default=None,
        help="Save baseline per-case data to this path (event timing only)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path (default: stdout)",
    )
    args = parser.parse_args()

    workdir = Path(args.workdir) if args.workdir else _find_agent_workdir()
    if not workdir.is_dir():
        print(json.dumps({"error": f"workdir not found: {workdir}"}), file=sys.stderr)
        sys.exit(1)

    report = evaluate(
        task_name=args.task,
        workdir=workdir,
        soc=args.soc,
        device_id=args.device,
        perf_impl=args.impl,
        warmup=args.warmup,
        repeat=args.repeat,
        seed=args.seed,
        use_event_timing=args.use_event_timing,
        baseline_path=args.baseline,
        save_baseline_path=args.save_baseline,
    )

    output_json = json.dumps(report, indent=2, ensure_ascii=False)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_json, encoding="utf-8")
        print(f"Results written to {out_path}")
    else:
        print(output_json)

    # Exit code: 0 if build+verify passed, 1 otherwise
    sys.exit(0 if report["build_ok"] and report["verify_ok"] else 1)


if __name__ == "__main__":
    main()
