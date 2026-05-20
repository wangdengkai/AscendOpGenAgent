"""Unit tests for wm_ops.py multi-shape gating extensions.

Tests cover:
- compute_utility 加权 w_close_to_target / w_shape_divergence
- select_nodes 过滤 parent_eligible=false 的节点
- REFINE：multi-shape eval 输入 → target_shape_regression / parent_eligible 字段
- REFINE：target_shape_regression=true → 子节点自动注入 P-ShapeSpec-01
"""

import importlib.util
import json
from pathlib import Path

import pytest


WORKTREE_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = WORKTREE_ROOT / "evolution" / "world_model" / "wm_ops.py"
SPEC = importlib.util.spec_from_file_location("wm_ops_under_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


# ============================================================
# compute_utility 多 shape 加权
# ============================================================

def _root_node():
    return {
        "id": "root", "mode": "completed", "status": "completed",
        "score": 1.0, "difficulty": 1, "depth": 0,
    }


def test_utility_close_to_target_bonus_when_parent_partial_passed():
    nodes = {
        "root": _root_node(),
        "n1": {  # 父节点：partial_passed，且接近 target_speedup * 0.85
            "id": "n1", "parent_id": "root", "status": "passed",
            "score": 1.1, "difficulty": 2, "depth": 1,
            "gating": MODULE.GATING_PARTIAL_PASSED,
            "aggregate": {"target_min_speedup": 1.1, "target_max_speedup": 1.1},
        },
        "c1": {
            "id": "c1", "parent_id": "n1", "status": "open",
            "difficulty": 2, "depth": 2, "strategy_combination": [],
        },
    }
    wm = {"target_speedup": 1.2, "decision_tree": {"nodes": nodes}}
    util_with_bonus = MODULE.compute_utility(nodes["c1"], nodes, wm)

    # 同样配置但 parent 没 gating 字段 → 无 bonus
    nodes["n1"].pop("gating")
    nodes["n1"].pop("aggregate")
    util_without_bonus = MODULE.compute_utility(nodes["c1"], nodes, wm)

    assert abs((util_with_bonus - util_without_bonus) - 0.5) < 1e-6


def test_utility_close_to_target_not_applied_when_parent_too_low():
    """parent_score < target * 0.85 → 不加 close_to_target bonus"""
    nodes = {
        "root": _root_node(),
        "n1": {
            "id": "n1", "parent_id": "root", "status": "passed",
            "score": 1.0, "difficulty": 2, "depth": 1,
            "gating": MODULE.GATING_PARTIAL_PASSED,
            "aggregate": {"target_min_speedup": 1.0, "target_max_speedup": 1.0},
        },
        "c1": {
            "id": "c1", "parent_id": "n1", "status": "open",
            "difficulty": 2, "depth": 2, "strategy_combination": [],
        },
    }
    wm = {"target_speedup": 1.2, "decision_tree": {"nodes": nodes}}
    util_low = MODULE.compute_utility(nodes["c1"], nodes, wm)

    nodes["n1"]["score"] = 1.05  # 1.05 >= 1.2 * 0.85 = 1.02
    nodes["n1"]["aggregate"] = {"target_min_speedup": 1.05, "target_max_speedup": 1.05}
    util_close = MODULE.compute_utility(nodes["c1"], nodes, wm)

    # parent_score 提升带来 3.0 × delta 的提升，再加 0.5 的 bonus
    delta_from_parent = 3.0 * (1.05 - 1.0)  # 0.15
    expected_total_delta = delta_from_parent + 0.5
    assert abs((util_close - util_low) - expected_total_delta) < 1e-6


def test_utility_shape_divergence_bonus_when_child_has_shape_spec():
    """parent shape_divergence ≥ 0.20 且 child strategy 含 P-ShapeSpec-01 → +1.0"""
    nodes = {
        "root": _root_node(),
        "n1": {
            "id": "n1", "parent_id": "root", "status": "passed",
            "score": 1.3, "difficulty": 2, "depth": 1,
            "gating": MODULE.GATING_PARTIAL_PASSED,
            "aggregate": {"target_min_speedup": 1.05, "target_max_speedup": 1.5},  # divergence ≈ 0.3
        },
        "c_with_spec": {
            "id": "c_with_spec", "parent_id": "n1", "status": "open",
            "difficulty": 2, "depth": 2,
            "strategy_combination": [MODULE.P_SHAPE_SPEC],
        },
        "c_without_spec": {
            "id": "c_without_spec", "parent_id": "n1", "status": "open",
            "difficulty": 2, "depth": 2,
            "strategy_combination": ["P1"],
        },
    }
    wm = {"target_speedup": 1.5, "decision_tree": {"nodes": nodes}}
    util_with_spec = MODULE.compute_utility(nodes["c_with_spec"], nodes, wm)
    util_without_spec = MODULE.compute_utility(nodes["c_without_spec"], nodes, wm)

    # difference should be exactly 1.0 (shape divergence bonus)
    assert abs((util_with_spec - util_without_spec) - 1.0) < 1e-6


def test_utility_no_bonus_when_divergence_low():
    """parent shape_divergence < 0.20 → 无 shape_divergence bonus 即使 child 带 P-ShapeSpec-01"""
    nodes = {
        "root": _root_node(),
        "n1": {
            "id": "n1", "parent_id": "root", "status": "passed",
            "score": 1.3, "difficulty": 2, "depth": 1,
            "gating": MODULE.GATING_PARTIAL_PASSED,
            "aggregate": {"target_min_speedup": 1.25, "target_max_speedup": 1.30},  # divergence ≈ 0.04
        },
        "c_with_spec": {
            "id": "c_with_spec", "parent_id": "n1", "status": "open",
            "difficulty": 2, "depth": 2,
            "strategy_combination": [MODULE.P_SHAPE_SPEC],
        },
        "c_without_spec": {
            "id": "c_without_spec", "parent_id": "n1", "status": "open",
            "difficulty": 2, "depth": 2,
            "strategy_combination": [],
        },
    }
    wm = {"target_speedup": 1.5, "decision_tree": {"nodes": nodes}}
    util_with_spec = MODULE.compute_utility(nodes["c_with_spec"], nodes, wm)
    util_without_spec = MODULE.compute_utility(nodes["c_without_spec"], nodes, wm)
    assert abs(util_with_spec - util_without_spec) < 1e-6


# ============================================================
# select_nodes parent_eligible 过滤
# ============================================================

def test_select_skips_open_children_of_target_regression_parent():
    """target_shape_regression 的父节点下，open 子节点不进 candidate"""
    nodes = {
        "root": _root_node(),
        "bad": {
            "id": "bad", "parent_id": "root", "status": "passed",
            "score": 0.8, "difficulty": 2, "depth": 1,
            "gating": MODULE.GATING_TARGET_REGRESSION,
            "aggregate": {"any_target_regression": True},
            "parent_eligible": False,
        },
        "bad_c1": {
            "id": "bad_c1", "parent_id": "bad", "status": "open",
            "difficulty": 2, "depth": 2, "strategy_combination": [],
            "mode": "strategy_guided",
        },
        "good": {
            "id": "good", "parent_id": "root", "status": "passed",
            "score": 1.5, "difficulty": 2, "depth": 1,
            "gating": MODULE.GATING_PARTIAL_PASSED,
            "aggregate": {"any_target_regression": False},
            "parent_eligible": True,
        },
        "good_c1": {
            "id": "good_c1", "parent_id": "good", "status": "open",
            "difficulty": 2, "depth": 2, "strategy_combination": [],
            "mode": "strategy_guided",
        },
    }
    wm = {"target_speedup": 1.5, "decision_tree": {"nodes": nodes}}
    selections = MODULE.select_nodes(wm, n=2)
    selected_ids = [s["node_id"] for s in selections]
    assert "bad_c1" not in selected_ids
    assert "good_c1" in selected_ids


def test_select_legacy_parent_without_gating_still_eligible():
    """旧节点没有 gating/aggregate 字段 → SELECT 默认视为 eligible（向后兼容）"""
    nodes = {
        "root": _root_node(),
        "legacy_parent": {
            "id": "legacy_parent", "parent_id": "root", "status": "passed",
            "score": 1.3, "difficulty": 2, "depth": 1,
            # 没有 gating / aggregate / parent_eligible 字段
        },
        "legacy_c1": {
            "id": "legacy_c1", "parent_id": "legacy_parent", "status": "open",
            "difficulty": 2, "depth": 2, "strategy_combination": [],
            "mode": "strategy_guided",
        },
    }
    wm = {"target_speedup": 1.5, "decision_tree": {"nodes": nodes}}
    selections = MODULE.select_nodes(wm, n=1)
    assert any(s["node_id"] == "legacy_c1" for s in selections)


# ============================================================
# REFINE: multi-shape gating 字段写入 + P-ShapeSpec-01 注入
# ============================================================

def _refine_setup(tmp_path, eval_results_per_parallel):
    """构造 wm + results_dir，调用 refine。

    eval_results_per_parallel: list of dict, 每项是某 parallel_index 的
    evaluation_results.json 内容
    """
    # 节点：root + n1 + 一个 open child（refine 后会写到这个 child 上）
    nodes = {
        "root": {
            "id": "root", "parent_id": None, "status": "completed",
            "score": 1.0, "difficulty": 1, "depth": 0, "children": ["n1"],
            "strategy_combination": [], "mode": "completed",
        },
        "n1": {
            "id": "n1", "parent_id": "root", "status": "in_progress",
            "score": None, "difficulty": 2, "depth": 1, "children": [],
            "strategy_combination": ["P1"], "mode": "strategy_guided",
        },
    }
    wm = {
        "best_score": 1.0,
        "stagnation_count": 0,
        "target_speedup": 1.2,
        "decision_tree": {"nodes": nodes},
        "session": {},
    }
    results_dir = tmp_path / "round_1"
    results_dir.mkdir()
    for i, er in enumerate(eval_results_per_parallel):
        pdir = results_dir / f"parallel_{i}"
        pdir.mkdir()
        (pdir / "evaluation_results.json").write_text(json.dumps(er), encoding="utf-8")

    parallel_map = {str(i): "n1" for i in range(len(eval_results_per_parallel))}
    MODULE.refine(wm, round_num=1, results_dir=str(results_dir), parallel_map=parallel_map)
    return wm, nodes


def _make_multi_shape_eval(target_speedups, gating, any_regression=False,
                            precision_passed=True, compile_ok=True):
    shape_results = {
        "target": [
            {
                "name": f"T{i+1}", "speedup": s,
                "baseline_time_us": 100.0, "evolved_time_us": 100.0 / s if s > 0 else -1,
                "precision_passed": precision_passed,
                "compilation_success": compile_ok,
                "pipeline": {}, "bottleneck": "unknown", "cv_pct": 0.0,
            }
            for i, s in enumerate(target_speedups)
        ],
        "generalization": [],
    }
    min_s = min(target_speedups) if target_speedups else 0.0
    return {
        "op_name": "test_op",
        "shapes_mode": "target",
        "shape_results": shape_results,
        "aggregate": {
            "target_min_speedup": min_s,
            "target_max_speedup": max(target_speedups) if target_speedups else 0.0,
            "all_target_above_baseline": all(s >= 1.0 for s in target_speedups),
            "any_target_regression": any_regression,
            "all_target_meet_target": all(s >= 1.2 for s in target_speedups),
        },
        "gating": gating,
        # legacy
        "baseline": {"tag": "baseline", "time_us": 100.0, "precision_passed": True,
                     "pipeline": {}, "bottleneck": "unknown", "cv_pct": 0.0},
        "evolved": {"tag": "evolved", "time_us": 100.0 / min_s if min_s > 0 else -1,
                    "precision_passed": precision_passed, "pipeline": {},
                    "bottleneck": "unknown", "cv_pct": 0.0},
        "comparison": {
            "compilation_success": compile_ok,
            "precision_passed": precision_passed,
            "speedup": min_s, "measurement_quality": "good",
            "time_delta_us": 0.0,
        },
    }


def test_refine_writes_gating_and_parent_eligible_for_fully_passed(tmp_path):
    er = _make_multi_shape_eval([1.5, 1.3], MODULE.GATING_FULLY_PASSED)
    wm, nodes = _refine_setup(tmp_path, [er])
    node = nodes["n1"]
    assert node["status"] == "passed"
    assert node["score"] == 1.3  # min(1.5, 1.3)
    assert node["gating"] == MODULE.GATING_FULLY_PASSED
    assert node["target_shape_regression"] is False
    assert node["parent_eligible"] is True


def test_refine_target_regression_sets_parent_eligible_false(tmp_path):
    er = _make_multi_shape_eval([1.5, 0.8], MODULE.GATING_TARGET_REGRESSION, any_regression=True)
    wm, nodes = _refine_setup(tmp_path, [er])
    node = nodes["n1"]
    # status 仍是 passed（评测通过，只是 gating 不达标）
    assert node["status"] == "passed"
    assert node["target_shape_regression"] is True
    assert node["parent_eligible"] is False
    assert "target_shape_regression" in (node.get("failure_reason") or "")
    assert MODULE.P_SHAPE_SPEC in (node.get("failure_reason") or "")


def test_refine_target_regression_injects_shape_spec_into_children(tmp_path):
    """target_regression 后生成的 open 子节点 strategy_combination 应自动含 P-ShapeSpec-01"""
    er = _make_multi_shape_eval([1.5, 0.8], MODULE.GATING_TARGET_REGRESSION, any_regression=True)
    wm, nodes = _refine_setup(tmp_path, [er])
    node = nodes["n1"]
    # refine 会给 n1 派生 children（按当前 mode=strategy_guided 默认派 1-2 个）
    child_ids = node.get("children", [])
    assert len(child_ids) >= 1
    for cid in child_ids:
        child = nodes[cid]
        assert MODULE.P_SHAPE_SPEC in child["strategy_combination"], (
            f"Child {cid} strategy_combination missing P-ShapeSpec-01: "
            f"{child['strategy_combination']}"
        )
        assert "shape-specialized" in child.get("description", "")


def test_refine_partial_passed_keeps_parent_eligible_true(tmp_path):
    er = _make_multi_shape_eval([1.15, 1.05], MODULE.GATING_PARTIAL_PASSED, any_regression=False)
    wm, nodes = _refine_setup(tmp_path, [er])
    node = nodes["n1"]
    assert node["gating"] == MODULE.GATING_PARTIAL_PASSED
    assert node["target_shape_regression"] is False
    assert node["parent_eligible"] is True
    # partial_passed 的子节点不强制注入 P-ShapeSpec-01
    child_ids = node.get("children", [])
    if child_ids:
        # 没有强制注入（除非未来 shape_divergence 主动选择路径触发，这里只测最小契约）
        first_child = nodes[child_ids[0]]
        # description 里不应该有 shape-specialized 提示
        assert "shape-specialized" not in first_child.get("description", "")


def test_refine_legacy_single_shape_still_works(tmp_path):
    """旧 evaluation_results.json（只有 comparison + baseline/evolved）的 REFINE 行为不破坏"""
    legacy_er = {
        "op_name": "legacy_op",
        "baseline": {"tag": "baseline", "time_us": 100.0, "precision_passed": True,
                     "pipeline": {}, "bottleneck": "unknown", "cv_pct": 0.0},
        "evolved": {"tag": "evolved", "time_us": 75.0, "precision_passed": True,
                    "pipeline": {}, "bottleneck": "unknown", "cv_pct": 0.0},
        "comparison": {
            "compilation_success": True, "precision_passed": True,
            "speedup": 100.0 / 75.0, "measurement_quality": "good",
        },
    }
    wm, nodes = _refine_setup(tmp_path, [legacy_er])
    node = nodes["n1"]
    assert node["status"] == "passed"
    assert abs(node["score"] - round(100.0 / 75.0, 4)) < 1e-3
    # multi-shape 字段：legacy 默认值
    assert node.get("parent_eligible") is True
    assert node.get("target_shape_regression") is False
    assert "gating" not in node  # legacy 不写 gating
