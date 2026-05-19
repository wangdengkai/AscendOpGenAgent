#!/usr/bin/env python3
"""对 lingxi-evo 输出目录的 AscendC 实现跑一次 msprof，产出 op_summary_*.csv。

用途：lingxi-evo / lingxi-partial 链路本身只走 torch_npu.profiler
(kernel_details.csv)，拿不到 aiv_*_ratio 这类 pipeline 利用率。该脚本
复用 ascendc-evaluation 的 AdvancedPerformanceEngine.warmup_and_measure，
单次 profile 后 msprof --export=on 落盘 op_summary_*.csv 到
{output_dir}/profiling/，供 analyze_profiling.py 消费并喂给决策树。

用法:
    python3 lingxi_msprof_driver.py --output_dir /path/to/task_dir [--case-idx 0]
                                     [--task-type vector] [--num-trials 20]

退出码：0 正常（含"无数据但已处理"情况），非 0 仅在参数错误或导入失败。
"""
import argparse
import importlib.util
import inspect
import json
import logging
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
ASCENDC_EVAL_SCRIPTS = REPO_ROOT / ".claude" / "skills" / "ascendc-evaluation" / "scripts"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _find_cls(module, preferred):
    import torch.nn as nn
    c = getattr(module, preferred, None)
    if inspect.isclass(c) and issubclass(c, nn.Module):
        return c
    for _, v in vars(module).items():
        if inspect.isclass(v) and issubclass(v, nn.Module) and v is not nn.Module:
            return v
    raise AttributeError(f"no nn.Module class found in {module}")


def _move(v, d):
    import torch
    if isinstance(v, torch.Tensor):
        return v.to(d)
    if isinstance(v, (list, tuple)):
        return type(v)(_move(x, d) for x in v)
    return v


def _clone(v):
    import torch
    if isinstance(v, torch.Tensor):
        return v.clone()
    if isinstance(v, (list, tuple)):
        return type(v)(_clone(x) for x in v)
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output_dir", required=True,
                    help="含 model.py / model_new_ascendc.py / kernel/build 的任务目录")
    ap.add_argument("--case-idx", type=int, default=0,
                    help="用 get_input_groups() 的哪个 case 做 profiling（默认 0）")
    ap.add_argument("--task-type", default="vector",
                    choices=["vector", "cube", "cv-mix", "unknown"])
    ap.add_argument("--num-trials", type=int, default=20)
    ap.add_argument("--device", type=int, default=None,
                    help="NPU device id；未指定时跟随 ASCEND_RT_VISIBLE_DEVICES，否则 0")
    ap.add_argument("--output", help="把 pipeline 汇总 JSON 写到该路径（可选）")
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    if not out_dir.is_dir():
        print(f"[ERROR] output_dir not found: {out_dir}", file=sys.stderr)
        return 2

    model_path = out_dir / "model_new_ascendc.py"
    if not model_path.exists():
        print(f"[ERROR] model_new_ascendc.py not found under {out_dir}", file=sys.stderr)
        return 2

    # 确保 AdvancedPerformanceEngine 可以被导入；该模块依赖与 evaluate.py 相同
    if str(ASCENDC_EVAL_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(ASCENDC_EVAL_SCRIPTS))
    sys.path.insert(0, str(out_dir / "kernel" / "build"))
    sys.path.insert(0, str(out_dir))

    try:
        import torch
        from AscendPerformanceTest import AdvancedPerformanceEngine  # type: ignore
    except Exception as e:
        print(f"[ERROR] failed to import torch or AdvancedPerformanceEngine: {e}",
              file=sys.stderr)
        return 2

    if args.device is not None:
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device)
        device_id = args.device
    else:
        device_id = int(os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0").split(",")[0])

    # 加载 ModelNew 和 inputs
    asc_mod = _load(model_path, "asc_mod")
    cls = _find_cls(asc_mod, "ModelNew")
    ref_mod = _load(out_dir / "model.py", "ref_for_init")
    init_args = getattr(ref_mod, "get_init_inputs", lambda: [])()
    input_groups = _load(out_dir / "model.py", "ref_for_inputs").get_input_groups()
    if not input_groups:
        print("[ERROR] get_input_groups() returned empty list", file=sys.stderr)
        return 2
    case_idx = max(0, min(args.case_idx, len(input_groups) - 1))

    device = torch.device("npu")
    torch.manual_seed(0)
    if hasattr(torch, "npu"):
        torch.npu.manual_seed(0)

    model = cls(*_clone(init_args)).to(device).eval()
    inputs = _move(_clone(input_groups[case_idx]), device)

    profile_root = out_dir / "profiling"
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    engine = AdvancedPerformanceEngine(logging.getLogger("lingxi_msprof"))

    try:
        median_us, perf_data, output_path, cv_pct = engine.warmup_and_measure(
            model, inputs, device_id, profile_root,
            num_trials=args.num_trials,
            task_type=args.task_type,
            model_tag="ModelNew",
        )
    except Exception as e:
        print(f"[WARN] msprof profiling failed: {e}", file=sys.stderr)
        # profiling 失败不视作硬错误 —— 决策树侧会走空 pipeline 回退
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(
                {"error": str(e), "pipeline": None}, indent=2))
        return 0

    # perf_data 是 list[dict]，每行对应 op_summary 里一个 op。decision-tree 侧
    # （wm_ops.refine / synthesize_analysis_from_pipeline）期望扁平 dict，所以
    # 把关键 ratio / time 字段合并成一个 dict，和 evaluate_ops_direct.py 的
    # pipeline 抽取字段保持一致。
    pipeline_flat = {}
    if isinstance(perf_data, list):
        wanted = (
            "aiv_mte2_ratio", "aiv_vec_ratio", "aiv_scalar_ratio",
            "aiv_mte3_ratio", "aiv_mte2_time(us)", "aiv_vec_time(us)",
            "aiv_scalar_time(us)", "aiv_mte3_time(us)",
            "aiv_icache_miss_rate", "cube_utilization(%)",
            "aic_mac_ratio", "aic_mte1_ratio", "aic_mte2_ratio",
        )
        for row in perf_data:
            if not isinstance(row, dict):
                continue
            for k in wanted:
                if k in row and row[k] is not None and k not in pipeline_flat:
                    pipeline_flat[k] = row[k]

    result = {
        "task_dir": str(out_dir),
        "case_idx": case_idx,
        "task_type": args.task_type,
        "device_id": device_id,
        "median_us": median_us,
        "cv_pct": cv_pct,
        "profile_dir": str(output_path),
        "pipeline": pipeline_flat or None,
        "pipeline_rows": perf_data if isinstance(perf_data, list) else None,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
