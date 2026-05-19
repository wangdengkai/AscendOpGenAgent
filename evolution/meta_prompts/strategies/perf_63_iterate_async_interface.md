# P63 Iterate 异步接口避免 AIC/AIV 同步依赖 (Iterate Async Interface)
## Overview
在 MIX（AIC+AIV 混合编程）场景中，使用 Matmul Iterate<false> 或 IterateAll<false> 异步接口替代同步 Iterate<true>，仅第一次调用需发送 AIV→AIC 消息，后续调用无需消息发送，减少 Cube 与 Vector 核间通信开销。需配合 SetWorkspace 使用。

## When to Use
- MIX 场景（AIC+AIV 混合编程）下 Matmul Iterate/IterateAll 调用
- 多次 Iterate 调用导致核间消息频繁发送
- Profiling 显示 AIC/AIV 同步等待时间较长

## Trade-off
- 需要额外分配 Workspace 空间（singleCoreM * singleCoreN * sizeof(float)）
- 仅适用于 MIX 场景

**Source operators**: SIMD算子性能优化/流水编排

---
## Variant A: Iterate<false> 异步调用
Source: SIMD算子性能优化/流水编排/使能Iterate或IterateAll异步接口避免AIC_AIV同步依赖.md

同步方式每次 Iterate 调用触发一次 AIV→AIC 消息，异步方式仅第一次需要发送消息。

**Expert implementation:**
```cpp
// 同步方式（反例）：每次 Iterate 都发送消息
matmulObj.Iterate<true>(cGlobal);  // 每次触发 AIV→AIC 消息

// 异步方式（正例）：仅第一次发送消息
matmulObj.SetWorkspace(workspace);
matmulObj.Iterate<false>(cGlobal);  // 仅首次发送消息，后续无需消息交互
```

**vs. baseline (lingxi-code):**
```cpp
// 基线：使用同步 Iterate<true>，每次调用都有核间通信开销
```

Benefit: 减少核间消息发送次数，降低 AIC/AIV 同步等待延迟
Trade-off: 需要额外分配 Workspace 空间
