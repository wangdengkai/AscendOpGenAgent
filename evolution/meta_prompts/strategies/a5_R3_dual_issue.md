# R3: 指令双发射

## Overview
通过循环展开和指令重排，暴露无数据依赖的指令对，最大化 A5 Vector Core 的双发射能力。

## When to Use
- 指令依赖链长（每条指令都依赖前一条的结果）
- Profiling 显示双发射利用率低
- 计算密集型 VF 函数

## Trade-off
- **收益**: 理论 2x 吞吐提升（实际 1.3-1.8x）
- **风险**: RegTensor 使用量翻倍（2x 展开 → ~2x RegTensor）
- **复杂度**: 需处理尾部余数

## Variant A: 2x 循环展开

```cpp
// Baseline
for (uint16_t i = 0; i < repTimes; ++i) {
    mask = UpdateMask<T>(count);
    LoadAlign(r0, src + i * repSize);
    Adds(r1, r0, scalar, mask);
    StoreAlign(dst + i * repSize, r1, mask);
}

// Evolved: 2x 展开，交错无依赖指令
uint16_t mainLoop = repTimes / 2;
uint16_t rem = repTimes % 2;
for (uint16_t i = 0; i < mainLoop; ++i) {
    uint16_t i0 = i * 2, i1 = i * 2 + 1;
    m0 = UpdateMask<T>(count);
    m1 = UpdateMask<T>(count);
    LoadAlign(r0, src + i0 * repSize);
    LoadAlign(r2, src + i1 * repSize);     // 双发射
    Adds(r1, r0, scalar, m0);
    Adds(r3, r2, scalar, m1);              // 双发射
    StoreAlign(dst + i0 * repSize, r1, m0);
    StoreAlign(dst + i1 * repSize, r3, m1); // 双发射
}
for (uint16_t i = 0; i < rem; ++i) { /* 余数处理 */ }
```

**Benefit**: Load/Compute/Store 均可双发射
**Trade-off**: RegTensor 从 2 增加到 4，MaskReg 从 1 增加到 2

## Variant B: 指令交错（不展开）

在单次迭代内，处理两组独立数据。

```cpp
// 适用于多输入场景（如 Add(a+b) 和 Mul(c*d) 独立）
LoadAlign(a, addrA);
LoadAlign(c, addrC);       // 与 a 无依赖
Add(a, a, b, mask);
Mul(c, c, d, mask);         // 与 Add 无依赖
StoreAlign(addrA, a, mask);
StoreAlign(addrC, c, mask);
```
