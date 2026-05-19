# P52: L2 Cache Hint 优化 (L2 Cache Hint Optimization)

## Overview
通过 SetL2CacheHint 在多核竞争场景下主动禁用 L2 缓存，避免缓存抖动（cache thrashing）。在 blockDimM==1 或输出数据量超过 L2 容量时，禁用 L2 可提升整体带宽利用率。

## When to Use
- 多核写同一输出区域（SplitK 场景）
- blockDimM==1（单行 M 分块，多核竞争同一权重列）
- 输出数据量 > L2 cache 容量，输出数据污染 L2
- 权重矩阵单行场景，多核加载同一权重导致 L2 抖动

## Trade-off
- 禁用 L2 后依赖 DDR 带宽，小数据量场景可能反而变慢
- 需要根据 blockDimM 和数据量动态决策
- 仅两行 API 调用，实现成本极低

**Source operators**: grouped_matmul, grouped_matmul_swiglu_quant_v2, quant_matmul_reduce_sum

---

## Variant A: blockDimM==1 禁用 L2
Source: quant_matmul_reduce_sum, grouped_matmul_swiglu_quant_v2

当 M 维仅一个分块时，多核竞争同一权重列，禁用权重的 L2 缓存避免抖动。

**Expert implementation:**
```cpp
if (blockDimM == 1) {
    // 多核加载同一权重列 → L2 cache line 频繁被驱逐
    // 禁用 L2，直接走 DDR 并行访问
    weightGmLocal.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
}
```

**vs. baseline (默认 L2):**
```cpp
// 所有 buffer 使用默认 L2 缓存
// blockDimM==1 时: 多核竞争 → cache thrashing → 有效带宽下降
```

Benefit: blockDimM==1 场景带宽提升 15-25%（避免 cache thrashing）
Trade-off: blockDimM>1 时不应禁用（L2 复用有收益）

## Variant B: 输出禁用 L2 + 权重单行 M 禁用 L2
Source: grouped_matmul

组合策略：输出 buffer 在 SplitK 场景禁用 L2，权重 buffer 在单行 M 场景禁用 L2。

```cpp
// 输出 L2 控制
if (isOutputDisableL2Cache) {
    // 输出数据量 > L2 容量，写出会驱逐输入/权重数据
    yGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
}

// 权重 L2 控制
if (blockDimM == 1) {
    weightGmLocal.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
}

// 输入保持 L2 启用（复用率高）
// xGm 默认 CACHE_MODE_ENABLE
```

Benefit: 输入 L2 命中率提升 10-20%（输出不再污染 L2），整体性能提升 5-10%
Trade-off: 需要 Host 侧计算 isOutputDisableL2Cache 标志
