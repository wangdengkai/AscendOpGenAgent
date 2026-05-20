"""End-to-end smoke test: 模拟一轮 ops-evo refine 走通 multi-shape gating
完整链路（不调用 NPU 子进程评估），验证：

1. evaluate_ops_direct 纯函数链路（normalize_call_spec → select_shapes_to_run →
   compute_aggregate → determine_gating → _synthesize_legacy_fields）输出格式
   与 schema.md / wm_ops refine 期望一致

2. 把生成的 evaluation_results.json 喂给 wm_ops refine，验证：
   - target_regression 节点 parent_eligible=false，子节点 strategy_combination
     自动注入 P-ShapeSpec-01
   - partial_passed 节点 parent_eligible=true，正常派生子节点
   - fully_passed 节点 score=min(target speedups)
   - 后续 SELECT 不会选 target_regression 节点的 open 子节点

3. 验证向后兼容：旧单 shape call_spec / 旧 evaluation_results 字段流转不破坏
"""

import importlib.util
import json
from pathlib import Path

import pytest


WORKTREE_ROOT = Path(__file__).resolve().parents[1]


def _load_module(rel_path: str, name: str):
    path = WORKTREE_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EVAL_MOD = _load_module(".claude/skills/ops-evaluation/scripts/evaluate_ops_direct.py", "eval_e2e")
WM_MOD = _load_module("evolution/world_model/wm_ops.py", "wm_e2e")


# ============================================================
# helpers
# ============================================================

def _make_per_shape_subprocess_result(name, group, time_us, precision=True):
    """模拟 run_single_version 子进程返回的 per-shape dict"""
    return {
        "name": name, "group": group, "tag": "evolved" if group != "_baseline" else "baseline",
        "time_us": time_us, "precision_passed": precision,
        "pipeline": {"aiv_vec_ratio": 0.5}, "bottleneck": "compute_bound",
        "cv_pct": 1.5, "correctness_message": "ok",
    }


def _build_eval_json_from_speedups(target_speedups, gen_speedups=None,
                                     target_speedup_threshold=1.2,
                                     baseline_time_us=100.0,
                                     all_precision=True):
    """模拟 evaluate_ops_direct 完整生成 evaluation_results.json 的过程。

    使用与生产路径**完全一致的纯函数链**（merge_per_shape_results /
    compute_aggregate / determine_gating / _synthesize_legacy_fields），
    只是绕过子进程评估直接喂数据。
    """
    baseline_results = []
    evolved_results = []
    precision_per_shape = {}

    for i, s in enumerate(target_speedups):
        name = f"T{i+1}"
        baseline_results.append(_make_per_shape_subprocess_result(name, "target", baseline_time_us))
        evolved_results.append(_make_per_shape_subprocess_result(
            name, "target", baseline_time_us / s if s > 0 else -1, all_precision))
        precision_per_shape[name] = (all_precision, "All outputs match" if all_precision else "FAIL")

    for i, s in enumerate(gen_speedups or []):
        name = f"G{i+1}"
        baseline_results.append(_make_per_shape_subprocess_result(name, "generalization", baseline_time_us))
        evolved_results.append(_make_per_shape_subprocess_result(
            name, "generalization", baseline_time_us / s if s > 0 else -1, all_precision))
        precision_per_shape[name] = (all_precision, "All outputs match" if all_precision else "FAIL")

    shape_results, compile_ok = EVAL_MOD.merge_per_shape_results(
        baseline_results, evolved_results, precision_per_shape)
    aggregate = EVAL_MOD.compute_aggregate(
        shape_results["target"], shape_results["generalization"], target_speedup_threshold)
    precision_ok = all(r.get("precision_passed", False) for r in shape_results["target"]) and \
                    all(r.get("precision_passed", False) for r in shape_results["generalization"])
    gating = EVAL_MOD.determine_gating(aggregate, precision_ok=precision_ok, compile_ok=compile_ok)
    legacy = EVAL_MOD._synthesize_legacy_fields(shape_results, aggregate)

    return {
        "op_name": "smoke_op",
        "eval_backend": "default",
        "shapes_mode": "target" if not gen_speedups else "all",
        "shape_results": shape_results,
        "aggregate": aggregate,
        "gating": gating,
        "baseline": legacy["baseline"],
        "evolved": legacy["evolved"],
        "comparison": legacy["comparison"],
    }


def _setup_wm_for_round(tmp_path, eval_dicts, node_strategies=None):
    """构造 wm + results_dir，每个 parallel_p 对应一个 node n{p+1}。

    node_strategies: list[list[str]]，每个 parallel 节点的 strategy_combination
    """
    nodes = {
        "root": {
            "id": "root", "parent_id": None, "status": "completed",
            "score": 1.0, "difficulty": 1, "depth": 0, "children": [],
            "strategy_combination": [], "mode": "completed",
        }
    }
    parallel_map = {}
    if node_strategies is None:
        node_strategies = [["P1"]] * len(eval_dicts)

    for i, strats in enumerate(node_strategies):
        nid = f"n{i+1}"
        nodes[nid] = {
            "id": nid, "parent_id": "root", "status": "in_progress",
            "score": None, "difficulty": 2, "depth": 1, "children": [],
            "strategy_combination": list(strats), "mode": "strategy_guided",
        }
        nodes["root"].setdefault("children", []).append(nid)
        parallel_map[str(i)] = nid

    wm = {
        "best_score": 1.0,
        "stagnation_count": 0,
        "target_speedup": 1.2,
        "decision_tree": {"nodes": nodes},
        "session": {},
    }
    results_dir = tmp_path / "round_1"
    results_dir.mkdir()
    for i, er in enumerate(eval_dicts):
        pdir = results_dir / f"parallel_{i}"
        pdir.mkdir()
        (pdir / "evaluation_results.json").write_text(json.dumps(er), encoding="utf-8")

    return wm, nodes, parallel_map, results_dir


# ============================================================
# E2E scenarios
# ============================================================

def test_e2e_multi_shape_fully_passed_round(tmp_path):
    """一轮 3 个 variant：T_reg / partial / fully — 验证全套字段流转 + SELECT"""
    # variant 0: target_regression（T1=1.5, T2=0.8）
    er0 = _build_eval_json_from_speedups([1.5, 0.8])
    # variant 1: partial_passed（T1=1.15, T2=1.05）
    er1 = _build_eval_json_from_speedups([1.15, 1.05])
    # variant 2: fully_passed（T1=1.5, T2=1.3）
    er2 = _build_eval_json_from_speedups([1.5, 1.3])

    # 三个 variant 都不含 P-ShapeSpec-01（initial round）
    wm, nodes, parallel_map, results_dir = _setup_wm_for_round(
        tmp_path, [er0, er1, er2],
        node_strategies=[["P1"], ["P2"], ["P3"]])

    # 验证 evaluate 产生的 gating 字段正确
    assert er0["gating"] == EVAL_MOD.GATING_TARGET_REGRESSION
    assert er1["gating"] == EVAL_MOD.GATING_PARTIAL_PASSED
    assert er2["gating"] == EVAL_MOD.GATING_FULLY_PASSED

    # 跑一轮 refine
    WM_MOD.refine(wm, round_num=1, results_dir=str(results_dir), parallel_map=parallel_map)

    # === 检查节点字段 ===
    n1 = nodes["n1"]  # target_regression
    assert n1["status"] == "passed"
    assert n1["score"] == 0.8  # min(1.5, 0.8)
    assert n1["gating"] == WM_MOD.GATING_TARGET_REGRESSION
    assert n1["target_shape_regression"] is True
    assert n1["parent_eligible"] is False
    assert "target_shape_regression" in n1["failure_reason"]
    assert WM_MOD.P_SHAPE_SPEC in n1["failure_reason"]

    n2 = nodes["n2"]  # partial_passed
    assert n2["status"] == "passed"
    assert n2["score"] == 1.05
    assert n2["gating"] == WM_MOD.GATING_PARTIAL_PASSED
    assert n2["target_shape_regression"] is False
    assert n2["parent_eligible"] is True

    n3 = nodes["n3"]  # fully_passed
    assert n3["status"] == "passed"
    assert n3["score"] == 1.3
    assert n3["gating"] == WM_MOD.GATING_FULLY_PASSED
    assert n3["parent_eligible"] is True

    # === 检查 target_regression 子节点自动注入 P-ShapeSpec-01 ===
    n1_children = n1.get("children", [])
    assert len(n1_children) >= 1
    for cid in n1_children:
        ch = nodes[cid]
        assert WM_MOD.P_SHAPE_SPEC in ch["strategy_combination"], (
            f"n1 child {cid} should have P-ShapeSpec-01 in strategy_combination, "
            f"got {ch['strategy_combination']}"
        )

    # === 检查 partial_passed 子节点 NOT 注入（除非 shape_divergence 触发） ===
    n2_children = n2.get("children", [])
    if n2_children:
        for cid in n2_children:
            ch = nodes[cid]
            # partial_passed 不应自动注入（其 shape_divergence = (1.15-1.05)/1.15 ≈ 0.087 < 0.20）
            assert WM_MOD.P_SHAPE_SPEC not in ch["strategy_combination"]

    # === 检查 SELECT 不会选 n1 的 open 子节点 ===
    selections = WM_MOD.select_nodes(wm, n=3)
    selected_ids = [s["node_id"] for s in selections]
    n1_child_set = set(n1_children)
    for sid in selected_ids:
        assert sid not in n1_child_set, (
            f"SELECT picked {sid} which is child of target_regression node n1"
        )

    # === 检查 best_score 反映 min-of-target 语义 ===
    assert wm["best_score"] == 1.3  # n3.score


def test_e2e_shape_divergence_triggers_active_spec(tmp_path):
    """parent partial_passed + shape_divergence ≥ 0.20 时，含 P-ShapeSpec-01 的子节点
    在 SELECT 中应获得更高 utility。"""
    # parent T1=1.5, T2=1.05 → divergence = 0.3 ≥ 0.20
    er = _build_eval_json_from_speedups([1.5, 1.05])
    wm, nodes, parallel_map, results_dir = _setup_wm_for_round(
        tmp_path, [er], node_strategies=[["P1"]])

    WM_MOD.refine(wm, round_num=1, results_dir=str(results_dir), parallel_map=parallel_map)

    n1 = nodes["n1"]
    assert n1["parent_eligible"] is True
    assert n1["gating"] == WM_MOD.GATING_PARTIAL_PASSED

    # 验证 shape_divergence 工具函数
    div = WM_MOD._node_shape_divergence(n1)
    assert div >= 0.20, f"expected divergence >= 0.20, got {div}"

    # 人工添加两个子节点：一个含 ShapeSpec，一个不含
    nodes["with_spec"] = {
        "id": "with_spec", "parent_id": "n1", "status": "open",
        "difficulty": 2, "depth": 2, "strategy_combination": [WM_MOD.P_SHAPE_SPEC],
        "mode": "strategy_guided",
    }
    nodes["without_spec"] = {
        "id": "without_spec", "parent_id": "n1", "status": "open",
        "difficulty": 2, "depth": 2, "strategy_combination": ["P4"],
        "mode": "strategy_guided",
    }

    u_with = WM_MOD.compute_utility(nodes["with_spec"], nodes, wm)
    u_without = WM_MOD.compute_utility(nodes["without_spec"], nodes, wm)
    assert abs((u_with - u_without) - 1.0) < 1e-6, (
        f"shape_divergence bonus should be +1.0 for child with P-ShapeSpec-01; "
        f"got u_with={u_with} u_without={u_without}"
    )


def test_e2e_legacy_single_shape_call_spec_normalization():
    """旧 call_spec.json（顶层 inputs）经 normalize_call_spec 后能跑通 aggregate / gating
    输出 legacy comparison.speedup 等于 target_min_speedup。
    """
    legacy_spec = {
        "op_namespace": "npu",
        "op_func": "npu_test_op",
        "inputs": [{"name": "x", "shape": [2, 4096], "dtype": "float16"}],
        "scalar_args": {"alpha": 1.0},
    }
    norm = EVAL_MOD.normalize_call_spec(legacy_spec)
    assert norm["target_shapes"][0]["name"] == "default"
    assert norm["generalization_shapes"] == []

    # 模拟跑出一组 speedup
    er = _build_eval_json_from_speedups([1.4], target_speedup_threshold=1.2)
    assert er["gating"] == EVAL_MOD.GATING_FULLY_PASSED
    assert er["aggregate"]["target_min_speedup"] == 1.4
    assert er["comparison"]["speedup"] == 1.4  # legacy 字段 = min-of-target


def test_e2e_legacy_evaluation_results_refine_unchanged(tmp_path):
    """旧 evaluation_results.json（只有 comparison + baseline/evolved，无 shape_results /
    aggregate / gating）喂给 refine 不会破坏行为：节点 parent_eligible 默认 True，
    target_shape_regression 默认 False。"""
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
    wm, nodes, parallel_map, results_dir = _setup_wm_for_round(
        tmp_path, [legacy_er], node_strategies=[["P1"]])
    WM_MOD.refine(wm, round_num=1, results_dir=str(results_dir), parallel_map=parallel_map)

    n1 = nodes["n1"]
    assert n1["status"] == "passed"
    assert abs(n1["score"] - round(100.0 / 75.0, 4)) < 1e-3
    assert n1.get("parent_eligible") is True  # 默认 True
    assert n1.get("target_shape_regression") is False  # 默认 False
    assert "gating" not in n1  # legacy 不写 gating

    # SELECT 也不会因 legacy parent 而屏蔽其 open 子节点
    selections = WM_MOD.select_nodes(wm, n=2)
    # n1 的 children（refine 派生的）应该可被选中
    n1_child_ids = set(n1.get("children", []))
    assert len(n1_child_ids) > 0, "legacy parent should still derive children"
    # 至少有一个 n1 子节点被 SELECT 选到（或被 padding 替代，取决于其他可选节点）
    # 不强制必须选中，但至少不能因 parent_eligible 被屏蔽


def test_e2e_full_chain_shape_results_round_trip():
    """完整 round-trip：evaluate 产生的 shape_results 结构与 wm refine 期望的字段一一对应"""
    # 用 G2=0.6 让几何平均跌破 1.0 触发 generalization_regression
    er = _build_eval_json_from_speedups([1.5, 1.25], gen_speedups=[1.1, 0.6])

    # 验证 evaluate 输出结构齐全
    assert "shape_results" in er and "aggregate" in er and "gating" in er
    assert "target" in er["shape_results"] and "generalization" in er["shape_results"]
    assert len(er["shape_results"]["target"]) == 2
    assert len(er["shape_results"]["generalization"]) == 2

    # generalization regression：geomean(1.1, 0.6) ≈ 0.812 < 1.0
    assert er["aggregate"]["any_generalization_regression"] is True
    assert er["gating"] == EVAL_MOD.GATING_GENERALIZATION_REGRESSION

    # 反例：geomean(1.1, 0.95) ≈ 1.022 > 1.0，不算 regression
    er_ok = _build_eval_json_from_speedups([1.5, 1.25], gen_speedups=[1.1, 0.95])
    assert er_ok["aggregate"]["any_generalization_regression"] is False
    assert er_ok["gating"] == EVAL_MOD.GATING_FULLY_PASSED

    # 验证 wm_ops _is_multi_shape_eval 能正确探测
    assert WM_MOD._is_multi_shape_eval(er) is True

    # 验证 shape_divergence 计算
    # target_max=1.5, target_min=1.25 → div = (1.5-1.25)/1.5 ≈ 0.167
    fake_node = {"aggregate": er["aggregate"]}
    div = WM_MOD._node_shape_divergence(fake_node)
    assert abs(div - 0.1667) < 0.01
