# R2: 寄存器复用

## Overview
通过布尔代数简化、指令重排和原地复用减少 RegTensor 使用量，避免超过 32 个限制导致的 spill/fill。

## When to Use
- VF 函数中 RegTensor 数量接近或超过 32
- Profiling 显示 spill/fill 指令
- 复杂计算链（如 GELU、Swish、LayerNorm 的 Compute 部分）

## Trade-off
- **收益**: 消除 spill/fill 指令，性能提升 10-30%
- **风险**: 过度复用可能引入数据依赖，阻碍双发射
- **复杂度**: 需仔细分析寄存器生命周期

## Variant A: 原地复用

```cpp
// Baseline: 5 个 RegTensor
RegTensor<float> a, b, c, d, result;
Add(c, a, b, mask);
Mul(d, c, a, mask);
Add(result, d, b, mask);

// Evolved: 3 个 RegTensor
RegTensor<float> a, b, tmp;
Add(tmp, a, b, mask);
Mul(tmp, tmp, a, mask);   // 原地复用 tmp
Add(tmp, tmp, b, mask);   // 继续复用
```

**Benefit**: RegTensor 从 5 减少到 3
**Trade-off**: tmp 的依赖链变长，可能影响双发射

## Variant B: 指令重排缩短生命周期

```cpp
// Baseline: r0, r1, r2 同时存活
LoadAlign(r0, addr0);
LoadAlign(r1, addr1);
LoadAlign(r2, addr2);
Add(r3, r0, r1, mask);
Mul(r4, r3, r2, mask);

// Evolved: 最多 2 个同时存活
LoadAlign(r0, addr0);
LoadAlign(r1, addr1);
Add(r0, r0, r1, mask);    // r1 可释放
LoadAlign(r1, addr2);      // 复用 r1
Mul(r0, r0, r1, mask);    // r0 原地复用
```

## Variant C: 拆分 VF 降低寄存器压力

当单个 VF 无法在 32 个 RegTensor 内完成时，拆分为多个 VF，中间结果通过 UB 传递。

**Trade-off**: 引入额外 Store/Load，但避免 spill/fill（通常 Store/Load 比 spill/fill 更可控）
