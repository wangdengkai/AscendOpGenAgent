#!/usr/bin/env python3
"""
profiling_evidence.py — 空泡诊断→世界模型策略映射

将 profiling 分析结果 (T1/T2/T3) 转换为世界模型可用的策略建议，
用于 lingxi-evo Refine 步骤中注入子节点的 strategy_combination。

用法:
    from profiling_evidence import extract_profiling_evidence

    evidence = extract_profiling_evidence(evaluation_results)
    # → {"bottleneck_type": "mte2_stall", "suggested_strategies": ["P1", "P10"], ...}
"""

from typing import Optional


# 瓶颈类型 → 推荐/反推荐策略映射
BOTTLENECK_STRATEGY_MAP = {
    "mte2_stall": {
        "primary": ["P1", "P10"],
        "secondary": ["P2", "P25", "P66", "P7", "P19", "P18", "P56", "P59"],
        "anti": ["P3"],
        "description": "MTE2 数据加载等待是主瓶颈，应通过双缓冲重叠搬运与计算；辅以 DataCopyPad 对齐 (P25) 和 512B 对齐 (P66) 减少带宽折损",
    },
    "mte3_stall": {
        "primary": ["P1", "P10"],
        "secondary": ["P8", "P19", "P40", "P56", "P59"],
        "anti": [],
        "description": "MTE3 写回等待是主瓶颈，双缓冲可重叠写回与下一轮计算",
    },
    "tiling_imbalance": {
        "primary": ["P4", "P2"],
        "secondary": ["P11", "P47", "P51", "P58", "P72", "P73", "P75"],
        "anti": ["P3"],
        "description": "多核负载不均衡，应优化分块策略使各核工作量均匀",
    },
    "scalar_loading": {
        "primary": ["P5", "P67"],
        "secondary": ["P2", "P10", "P54", "P84"],
        "anti": [],
        "description": "标量参数加载阻塞 VEC，应预取参数或简化地址计算",
    },
    "scalar_compute": {
        "primary": ["P5", "P67"],
        "secondary": ["P10", "P84"],
        "anti": [],
        "description": "标量地址计算阻塞 VEC，应简化 tiling 逻辑或预计算偏移",
    },
    "compute_bound": {
        "primary": ["P13", "D1", "P46"],
        "secondary": ["A1", "P47", "P68", "P69", "P79", "P84"],
        "anti": [],
        "description": "计算密集型瓶颈，应考虑算法优化或混合精度",
    },
    "near_optimal": {
        "primary": [],
        "secondary": [],
        "anti": [],
        "description": "内核已接近最优，无明显可优化空泡",
    },
    # ── 新增子类型 ──
    "no_overlap": {
        "primary": ["P1"],
        "secondary": ["P10", "P19", "P63", "P82"],
        "anti": [],
        "description": "CopyIn/Compute/CopyOut 无时间重叠，需启用双缓冲",
    },
    "partial_overlap": {
        "primary": ["P1", "P2"],
        "secondary": ["P8", "P18", "P26", "P53", "P83"],
        "anti": [],
        "description": "双缓冲已启用但重叠不充分，需调整 tile 大小或 UB 分区",
    },
    "undersize_transfer": {
        "primary": ["P2", "P10"],
        "secondary": ["P7", "P25", "P45", "P66"],
        "anti": [],
        "description": "DMA 搬运粒度过小，增大 tile 或使用向量化搬运、合并搬运",
    },
    "icache_miss": {
        "primary": ["P8", "P54"],
        "secondary": [],
        "anti": ["P13"],
        "description": "指令缓存未命中，内核代码体积过大，应简化或拆分（P8）、合并小 kernel 减少 launch 开销（P54）",
    },
    "bus_contention": {
        "primary": ["P8", "P65"],
        "secondary": ["P1", "P28", "P64"],
        "anti": [],
        "description": "MTE2/MTE3 总线竞争，应错开搬运时序或使用 HardEvent 同步",
    },
    "l2_cache_thrash": {
        "primary": ["P52", "P74"],
        "secondary": ["P61", "P78"],
        "anti": [],
        "description": "L2 cache 频繁失效，多核竞争导致缓存抖动，应禁用或引导 L2 缓存",
    },
    "ub_memory_pressure": {
        "primary": ["P8", "P85"],
        "secondary": ["P81", "P88"],
        "anti": ["P20"],
        "description": "UB 内存压力过大，应优化 buffer 分区复用或减少同时驻留的 tensor 数量",
    },
}


def classify_bottleneck(profiling_analysis: dict) -> str:
    """
    从 profiling_analysis 字段判定主瓶颈类型。

    Args:
        profiling_analysis: evaluation_results.json 中的 profiling_analysis 字段

    Returns:
        瓶颈类型字符串 (BOTTLENECK_STRATEGY_MAP 的 key)
    """
    if not profiling_analysis:
        return "near_optimal"

    imbalance_ratio = profiling_analysis.get("imbalance_ratio", 1.0)
    d_class_pct = profiling_analysis.get("d_class_pct", 0)
    c_class_pct = profiling_analysis.get("c_class_pct", 0)
    primary_bottleneck = profiling_analysis.get("primary_bottleneck")

    # 优先级: tiling_imbalance > D类 > C类 > near_optimal
    if imbalance_ratio > 1.3:
        return "tiling_imbalance"

    if d_class_pct > 30:
        # 细化 D 类子类型 — 使用 T7/T8/T9 数据
        overlap_status = profiling_analysis.get("overlap_status")
        if overlap_status == "no_overlap":
            return "no_overlap"
        if overlap_status == "partial_overlap":
            return "partial_overlap"

        # DMA 效率
        dma_eff = profiling_analysis.get("dma_efficiency", {})
        mte2_short = dma_eff.get("mte2_short_pct", 0)
        if mte2_short > 30:
            return "undersize_transfer"

        # 周期性 icache miss
        pattern_type = profiling_analysis.get("pattern_type")
        dominant_subtype = profiling_analysis.get("dominant_subtype", "")
        if pattern_type == "periodic" and "icache" in str(dominant_subtype):
            return "icache_miss"

        if primary_bottleneck == "MTE2":
            return "mte2_stall"
        elif primary_bottleneck == "MTE3":
            return "mte3_stall"
        else:
            return "mte2_stall"  # 默认归为 MTE2

    if c_class_pct > 20:
        c_cause = profiling_analysis.get("c_class_primary_cause")
        if c_cause == "SCALARLDST":
            return "scalar_loading"
        elif c_cause == "SCALAR":
            return "scalar_compute"
        return "scalar_loading"  # 默认

    # 高计算占比 → compute_bound
    compute_pct = profiling_analysis.get("pure_compute_pct", 0)
    if compute_pct > 70:
        return "compute_bound"

    return "near_optimal"


def extract_profiling_evidence(evaluation_results: dict) -> Optional[dict]:
    """
    从 evaluation_results.json 提取 profiling 证据，生成世界模型可用的结构。

    Args:
        evaluation_results: 完整的 evaluation_results.json 内容

    Returns:
        profiling_evidence dict (可直接写入世界模型节点)，
        若无 profiling 数据则返回 None
    """
    profiling = evaluation_results.get("profiling_analysis")
    if not profiling:
        return None

    bottleneck_type = classify_bottleneck(profiling)
    strategy_info = BOTTLENECK_STRATEGY_MAP.get(bottleneck_type, {})

    suggested = strategy_info.get("primary", []) + strategy_info.get("secondary", [])
    anti = strategy_info.get("anti", [])

    return {
        "bottleneck_type": bottleneck_type,
        "d_class_pct": profiling.get("d_class_pct", 0),
        "c_class_pct": profiling.get("c_class_pct", 0),
        "imbalance_ratio": profiling.get("imbalance_ratio", 1.0),
        "primary_bottleneck": profiling.get("primary_bottleneck"),
        "suggested_strategies": suggested,
        "anti_strategies": anti,
        "description": strategy_info.get("description", ""),
        "top_recommendation": profiling.get("top_recommendation", ""),
        # ── 新增字段 ──
        "pattern_type": profiling.get("pattern_type"),
        "overlap_status": profiling.get("overlap_status"),
        "dominant_subtype": profiling.get("dominant_subtype"),
        "dma_efficiency": profiling.get("dma_efficiency"),
    }


def synthesize_analysis_from_pipeline(
    pipeline: Optional[dict],
    bottleneck_hint: Optional[str] = None,
) -> Optional[dict]:
    """Coarse-grained pipeline → profiling_analysis mapping (CSV-level fallback).

    evaluate_ops_direct.py only writes pipeline ratios (aiv_mte2_ratio,
    aiv_vec_ratio, aiv_scalar_ratio, aiv_mte3_ratio). This helper promotes
    those ratios to the subset of profiling_analysis fields that
    classify_bottleneck() relies on, so extract_profiling_evidence() can be
    used from the main refine loop without requiring deep-profiling.

    Fields that truly need instruction-level trace (pattern_type, overlap_status,
    dominant_subtype, dma_efficiency) are left None — deep-profiling still owns
    those.
    """
    if not pipeline:
        return None

    def _pct(key: str) -> float:
        v = pipeline.get(key)
        if v is None:
            return 0.0
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        # Normalize ratios (0..1) to percentages (0..100)
        if 0.0 <= f <= 1.0:
            f *= 100.0
        return f

    mte2 = _pct("aiv_mte2_ratio")
    mte3 = _pct("aiv_mte3_ratio")
    vec = _pct("aiv_vec_ratio")
    scalar = _pct("aiv_scalar_ratio")
    # icache miss rate is usually stored as a ratio (0..1); _pct already
    # normalizes it to percentage.
    icache_miss_pct = _pct("aiv_icache_miss_rate")

    # D-class: non-VEC ratio (MTE2 + MTE3 dominate)
    d_class_pct = max(mte2, mte3)
    if mte2 >= mte3:
        primary_bn = "MTE2"
    else:
        primary_bn = "MTE3"

    # C-class proxy: scalar ratio above ~20% is a scalar pressure signal.
    c_class_pct = scalar if scalar > 15 else 0.0
    c_cause = "SCALARLDST" if scalar > 15 else None

    # CSV-level icache miss signal. classify_bottleneck() gates icache_miss on
    # (d_class_pct > 30) AND pattern_type=="periodic" AND "icache" in
    # dominant_subtype. Synthesize the latter two from the CSV column so the
    # main refine loop can route to icache_miss without waiting for deep
    # profiling. Only flag when miss rate is meaningfully high (>10%).
    if icache_miss_pct > 10.0:
        pattern_type = "periodic"
        dominant_subtype = "icache"
        # Ensure d_class gate passes so classify_bottleneck picks icache_miss
        # over compute_bound fallback.
        if d_class_pct <= 30:
            d_class_pct = 31.0
    else:
        pattern_type = None
        dominant_subtype = None

    return {
        "imbalance_ratio": 1.0,   # not observable from CSV
        "d_class_pct": d_class_pct,
        "c_class_pct": c_class_pct,
        "primary_bottleneck": primary_bn if d_class_pct > 30 else None,
        "pure_compute_pct": vec,
        "c_class_primary_cause": c_cause,
        "overlap_status": None,
        "dominant_subtype": dominant_subtype,
        "pattern_type": pattern_type,
        "dma_efficiency": {},
        "icache_miss_pct": icache_miss_pct,
        "_synthesis_source": "pipeline_csv",
        "_bottleneck_hint": bottleneck_hint,
    }


def merge_strategies_with_evidence(
    base_strategies: list[str],
    evidence: Optional[dict],
) -> list[str]:
    """
    将 profiling 证据的策略建议合并到节点的 strategy_combination 中。

    - 添加 suggested_strategies (去重)
    - 移除 anti_strategies

    Args:
        base_strategies: 节点原有的策略列表
        evidence: extract_profiling_evidence() 的输出

    Returns:
        合并后的策略列表
    """
    if not evidence:
        return base_strategies

    result = list(base_strategies)

    # 添加建议策略
    for s in evidence.get("suggested_strategies", []):
        if s not in result:
            result.append(s)

    # 移除反推荐策略
    anti = set(evidence.get("anti_strategies", []))
    result = [s for s in result if s not in anti]

    return result
