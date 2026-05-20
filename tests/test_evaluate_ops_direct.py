"""Unit tests for evaluate_ops_direct.py multi-shape extensions.

只测试纯函数（normalize_call_spec / select_shapes_to_run / compute_aggregate /
determine_gating / merge_per_shape_results / _synthesize_legacy_fields）。
不触发 NPU 子进程评估。
"""

import importlib.util
import json
import math
from pathlib import Path

import pytest


WORKTREE_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = WORKTREE_ROOT / ".claude" / "skills" / "ops-evaluation" / "scripts" / "evaluate_ops_direct.py"
SPEC = importlib.util.spec_from_file_location("evaluate_ops_direct_under_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


# ============================================================
# normalize_call_spec
# ============================================================

def test_normalize_legacy_single_shape_auto_wrap():
    """旧 call_spec（顶层 inputs）自动包装成 target_shapes=[default]。"""
    legacy = {
        "op_namespace": "npu",
        "op_func": "npu_add",
        "inputs": [{"name": "x", "shape": [2, 4], "dtype": "float16"}],
        "scalar_args": {"alpha": 1.0},
    }
    norm = MODULE.normalize_call_spec(legacy)
    assert "target_shapes" in norm
    assert len(norm["target_shapes"]) == 1
    assert norm["target_shapes"][0]["name"] == "default"
    assert norm["target_shapes"][0]["inputs"] == legacy["inputs"]
    assert norm["target_shapes"][0]["scalar_args"] == {"alpha": 1.0}
    assert norm["generalization_shapes"] == []
    # 原顶层字段保留
    assert "inputs" in norm


def test_normalize_multi_shape_passthrough():
    multi = {
        "op_namespace": "npu",
        "op_func": "npu_rms_norm",
        "target_shapes": [
            {"name": "T1", "inputs": [{"name": "x", "shape": [2, 4096, 5120], "dtype": "float16"}]},
            {"name": "T2", "inputs": [{"name": "x", "shape": [1, 1024, 5120], "dtype": "float16"}]},
        ],
        "generalization_shapes": [
            {"name": "G1", "inputs": [{"name": "x", "shape": [4, 2048, 5120], "dtype": "float16"}]},
        ],
    }
    norm = MODULE.normalize_call_spec(multi)
    assert len(norm["target_shapes"]) == 2
    assert len(norm["generalization_shapes"]) == 1


def test_normalize_duplicate_name_raises():
    bad = {
        "op_namespace": "npu", "op_func": "x",
        "target_shapes": [
            {"name": "T1", "inputs": []},
            {"name": "T1", "inputs": []},
        ],
    }
    with pytest.raises(ValueError, match="重复"):
        MODULE.normalize_call_spec(bad)


def test_normalize_cross_group_name_collision_raises():
    bad = {
        "op_namespace": "npu", "op_func": "x",
        "target_shapes": [{"name": "S", "inputs": []}],
        "generalization_shapes": [{"name": "S", "inputs": []}],
    }
    with pytest.raises(ValueError, match="重复"):
        MODULE.normalize_call_spec(bad)


def test_normalize_missing_target_raises():
    bad = {"op_namespace": "npu", "op_func": "x"}
    with pytest.raises(ValueError, match="target_shapes"):
        MODULE.normalize_call_spec(bad)


def test_normalize_empty_target_list_raises():
    bad = {"op_namespace": "npu", "op_func": "x", "target_shapes": []}
    with pytest.raises(ValueError, match="不能为空"):
        MODULE.normalize_call_spec(bad)


# ============================================================
# select_shapes_to_run
# ============================================================

def _two_target_one_gen():
    return MODULE.normalize_call_spec({
        "op_namespace": "npu", "op_func": "x",
        "target_shapes": [
            {"name": "T1", "inputs": [{"name": "x", "shape": [2], "dtype": "float16"}]},
            {"name": "T2", "inputs": [{"name": "x", "shape": [4], "dtype": "float16"}]},
        ],
        "generalization_shapes": [
            {"name": "G1", "inputs": [{"name": "x", "shape": [8], "dtype": "float16"}]},
        ],
    })


def test_select_target_only():
    sel = MODULE.select_shapes_to_run(_two_target_one_gen(), MODULE.SHAPES_MODE_TARGET)
    assert len(sel["target"]) == 2
    assert sel["generalization"] == []


def test_select_generalization_only():
    sel = MODULE.select_shapes_to_run(_two_target_one_gen(), MODULE.SHAPES_MODE_GENERALIZATION)
    assert sel["target"] == []
    assert len(sel["generalization"]) == 1


def test_select_all():
    sel = MODULE.select_shapes_to_run(_two_target_one_gen(), MODULE.SHAPES_MODE_ALL)
    assert len(sel["target"]) == 2
    assert len(sel["generalization"]) == 1


def test_select_unknown_mode_raises():
    with pytest.raises(ValueError):
        MODULE.select_shapes_to_run(_two_target_one_gen(), "garbage")


# ============================================================
# compute_aggregate
# ============================================================

def _t(name, speedup, prec=True):
    return {"name": name, "speedup": speedup, "precision_passed": prec}


def test_aggregate_all_meet_target():
    agg = MODULE.compute_aggregate(
        [_t("T1", 1.5), _t("T2", 1.3)], [], target_speedup=1.2)
    assert agg["target_min_speedup"] == 1.3
    assert agg["target_max_speedup"] == 1.5
    assert agg["all_target_meet_target"] is True
    assert agg["all_target_above_baseline"] is True
    assert agg["any_target_regression"] is False


def test_aggregate_partial_passed():
    """所有 ≥ 1.0x 但有未达 1.2x"""
    agg = MODULE.compute_aggregate(
        [_t("T1", 1.5), _t("T2", 1.05)], [], target_speedup=1.2)
    assert agg["all_target_above_baseline"] is True
    assert agg["all_target_meet_target"] is False
    assert agg["any_target_regression"] is False


def test_aggregate_target_regression():
    """任一 target < 1.0x"""
    agg = MODULE.compute_aggregate(
        [_t("T1", 1.5), _t("T2", 0.8)], [], target_speedup=1.2)
    assert agg["all_target_above_baseline"] is False
    assert agg["any_target_regression"] is True


def test_aggregate_precision_failure_treated_as_regression():
    agg = MODULE.compute_aggregate(
        [_t("T1", 1.5, prec=False)], [], target_speedup=1.2)
    assert agg["all_target_above_baseline"] is False
    assert agg["any_target_regression"] is True


def test_aggregate_generalization_geomean():
    agg = MODULE.compute_aggregate(
        [_t("T1", 1.5)],
        [_t("G1", 1.1), _t("G2", 0.95)],
        target_speedup=1.2)
    expected = math.exp((math.log(1.1) + math.log(0.95)) / 2)
    assert abs(agg["generalization_geo_mean_speedup"] - round(expected, 4)) < 1e-3
    assert agg["any_generalization_regression"] is (expected < 1.0)


def test_aggregate_no_target_speedup_threshold():
    agg = MODULE.compute_aggregate(
        [_t("T1", 1.5)], [], target_speedup=None)
    assert agg["all_target_meet_target"] is None


# ============================================================
# determine_gating
# ============================================================

def test_gating_failed_when_compile_fails():
    agg = MODULE.compute_aggregate([_t("T1", 1.5)], [], target_speedup=1.2)
    assert MODULE.determine_gating(agg, precision_ok=True, compile_ok=False) == MODULE.GATING_FAILED


def test_gating_failed_when_precision_fails():
    agg = MODULE.compute_aggregate([_t("T1", 1.5)], [], target_speedup=1.2)
    assert MODULE.determine_gating(agg, precision_ok=False, compile_ok=True) == MODULE.GATING_FAILED


def test_gating_target_regression():
    agg = MODULE.compute_aggregate([_t("T1", 0.8)], [], target_speedup=1.2)
    assert MODULE.determine_gating(agg, precision_ok=True, compile_ok=True) == MODULE.GATING_TARGET_REGRESSION


def test_gating_generalization_regression():
    agg = MODULE.compute_aggregate(
        [_t("T1", 1.5)], [_t("G1", 0.8), _t("G2", 0.85)], target_speedup=1.2)
    assert MODULE.determine_gating(agg, precision_ok=True, compile_ok=True) == MODULE.GATING_GENERALIZATION_REGRESSION


def test_gating_fully_passed():
    agg = MODULE.compute_aggregate(
        [_t("T1", 1.5), _t("T2", 1.3)],
        [_t("G1", 1.05)],
        target_speedup=1.2)
    assert MODULE.determine_gating(agg, precision_ok=True, compile_ok=True) == MODULE.GATING_FULLY_PASSED


def test_gating_partial_passed():
    """没达 1.2x 但全部 ≥ 1.0x"""
    agg = MODULE.compute_aggregate(
        [_t("T1", 1.15), _t("T2", 1.05)], [], target_speedup=1.2)
    assert MODULE.determine_gating(agg, precision_ok=True, compile_ok=True) == MODULE.GATING_PARTIAL_PASSED


def test_gating_target_regression_priority_over_generalization():
    """target_regression 优先级高于 generalization_regression。"""
    agg = MODULE.compute_aggregate(
        [_t("T1", 0.9)], [_t("G1", 0.8)], target_speedup=1.2)
    assert MODULE.determine_gating(agg, precision_ok=True, compile_ok=True) == MODULE.GATING_TARGET_REGRESSION


# ============================================================
# merge_per_shape_results + _synthesize_legacy_fields
# ============================================================

def test_merge_and_legacy_synthesis_target_only():
    baseline = [
        {"name": "T1", "group": "target", "tag": "baseline", "time_us": 100.0,
         "precision_passed": True, "pipeline": {}, "bottleneck": "x", "cv_pct": 1.0},
    ]
    evolved = [
        {"name": "T1", "group": "target", "tag": "evolved", "time_us": 80.0,
         "precision_passed": True, "pipeline": {"aiv_vec_ratio": 0.5},
         "bottleneck": "compute_bound", "cv_pct": 2.0},
    ]
    precision = {"T1": (True, "All outputs match")}
    shape_results, compile_ok = MODULE.merge_per_shape_results(baseline, evolved, precision)
    assert compile_ok is True
    assert len(shape_results["target"]) == 1
    head = shape_results["target"][0]
    assert head["speedup"] == 1.25
    assert head["precision_passed"] is True

    agg = MODULE.compute_aggregate(shape_results["target"], shape_results["generalization"], target_speedup=1.2)
    legacy = MODULE._synthesize_legacy_fields(shape_results, agg)
    assert legacy["baseline"]["time_us"] == 100.0
    assert legacy["evolved"]["time_us"] == 80.0
    # min-of-target 语义 — 只有 1 个 target，min 即 1.25
    assert legacy["comparison"]["speedup"] == 1.25
    assert legacy["comparison"]["precision_passed"] is True


def test_merge_legacy_speedup_uses_min_of_target():
    """有 2 个 target 时，legacy comparison.speedup = min。"""
    baseline = [
        {"name": "T1", "group": "target", "time_us": 100.0, "precision_passed": True,
         "pipeline": {}, "bottleneck": "x", "cv_pct": 1.0, "tag": "baseline"},
        {"name": "T2", "group": "target", "time_us": 200.0, "precision_passed": True,
         "pipeline": {}, "bottleneck": "x", "cv_pct": 1.0, "tag": "baseline"},
    ]
    evolved = [
        {"name": "T1", "group": "target", "time_us": 50.0, "precision_passed": True,
         "pipeline": {}, "bottleneck": "x", "cv_pct": 1.0, "tag": "evolved"},   # 2.0x
        {"name": "T2", "group": "target", "time_us": 180.0, "precision_passed": True,
         "pipeline": {}, "bottleneck": "x", "cv_pct": 1.0, "tag": "evolved"},  # ~1.11x
    ]
    precision = {"T1": (True, "ok"), "T2": (True, "ok")}
    sr, _ = MODULE.merge_per_shape_results(baseline, evolved, precision)
    agg = MODULE.compute_aggregate(sr["target"], [], target_speedup=1.5)
    legacy = MODULE._synthesize_legacy_fields(sr, agg)
    # min(2.0, ~1.11) = ~1.11
    assert abs(legacy["comparison"]["speedup"] - 1.1111) < 0.01


def test_merge_handles_missing_evolved():
    baseline = [{"name": "T1", "group": "target", "time_us": 100.0, "precision_passed": True,
                 "pipeline": {}, "bottleneck": "x", "cv_pct": 0.0, "tag": "baseline"}]
    evolved = []  # evolved 子进程整体崩了
    precision = {}
    sr, compile_ok = MODULE.merge_per_shape_results(baseline, evolved, precision)
    assert sr["target"] == []  # baseline 在 evolved_map 找不到对应项被跳过
