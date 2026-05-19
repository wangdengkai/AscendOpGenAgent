# P51: 动态 AIC/AIV 核配比 (Dynamic AIC/AIV Core Ratio)

## Overview
根据 K 维大小动态选择 AIC:AIV 比例（1:1 vs 1:2 vs CUBE_ONLY），在 Cube 密集和 Vector 密集场景间自动切换，释放闲置核资源。

## When to Use
- AIC/AIV 混合核架构的 Cube+Vector 融合算子
- K 维跨度大（从几百到几千），固定比例无法兼顾
- 大 K 场景 Vector 核闲置（Cube 是瓶颈）
- 含 GELU 等重 Vector 后处理的算子需要更多 Vector 核
- 与 P16（代价模型分核）互补：P16 是多维度代价模型，P51 是 K 维阈值切换

## Trade-off
- 需要 Host 侧根据 K 值选择 kernel type，增加 tiling 逻辑
- 阈值参数（K=2048）需要根据具体算子调优
- 动态切换可能影响系统级多算子调度

**Source operators**: grouped_matmul, quant_matmul_reduce_sum

---

## Variant A: K 阈值二模式切换
Source: quant_matmul_reduce_sum

K<2048 用 MIX_AIC_1_2（1 Cube : 2 Vector），K≥2048 用 MIX_AIC_1_1（1:1）。

**Expert implementation:**
```cpp
// Host 侧 tiling 决策
bool IsAivAicRatioTwoRequired() {
    return k < DOUBLE_VECTOR_THRESHOLD_K_UPPER;  // K < 2048
}

if (IsAivAicRatioTwoRequired()) {
    kernelType = KERNEL_TYPE_MIX_AIC_1_2;  // 小 K: Cube 快，需要更多 Vector
} else {
    kernelType = KERNEL_TYPE_MIX_AIC_1_1;  // 大 K: Vector 成瓶颈，释放多余 Vector
}

// Kernel 侧通过 TilingKey 分发
if (TILING_KEY_IS(0)) { /* 1:2 模式 kernel */ }
if (TILING_KEY_IS(1)) { /* 1:1 模式 kernel */ }
```

**vs. baseline (固定 1:2):**
```cpp
// 始终使用 KERNEL_TYPE_MIX_AIC_1_2
// 大 K 场景: Vector 核闲置，系统资源浪费
```

Benefit: 大 K 场景系统吞吐提升 10-20%（释放的 Vector 核可服务其他算子）
Trade-off: 阈值需要调优

## Variant B: CUBE_ONLY/1:1/1:2 三模式自动选择
Source: grouped_matmul

根据工作负载特征自动选择三种模式之一。

```cpp
// 三模式决策逻辑
if (noVectorPostProcess) {
    kernelType = KERNEL_TYPE_CUBE_ONLY;     // 纯 matmul，无 Vector 后处理
} else if (k >= THRESHOLD_K && !hasGELU) {
    kernelType = KERNEL_TYPE_MIX_AIC_1_1;   // 大 K + 轻 Vector
} else {
    kernelType = KERNEL_TYPE_MIX_AIC_1_2;   // 小 K 或重 Vector（GELU）
}
// GELU 激活: 始终 1:2（Vector 负载重）
// per-token/per-group scale: 检查可用性后决定
```

Benefit: 三模式覆盖更多场景，CUBE_ONLY 最大化 Cube 利用率
Trade-off: 决策逻辑更复杂，需要更多 tiling 参数
