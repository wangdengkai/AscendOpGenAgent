# P66 GM 地址 512B 对齐优化带宽 (GM 512B Alignment Bandwidth)
## Overview
AI 处理器内部设计约束下，GM 地址 512B 对齐可最有效发挥带宽效率。实测数据显示 32B 对齐场景带宽最差时仅为 512B 对齐的 70%。Kernel 入参（包括 Workspace/Tiling）地址已保证 512B 对齐，开发者需关注偏移量是否保持 512B 对齐。

## When to Use
- Atlas A2 训练/推理系列产品
- GM 数据搬运场景
- Profiling 显示 MTE2/MTE3 带宽利用率低

## Trade-off
- 可能需要调整数据布局以保持 512B 对齐
- 与 P7（32B 对齐）互补：P7 关注最低对齐要求，本策略关注最优带宽对齐

**Source operators**: SIMD算子性能优化/内存访问

---
## Variant A: GM→UB 方向 512B 对齐
Source: SIMD算子性能优化/内存访问/GM地址尽量512B对齐.md

确保 GM 源地址 512B 对齐，最大化 MTE2 搬运带宽。

**Expert implementation:**
```cpp
// 确保偏移量保持 512B 对齐
uint32_t offset = AlignUp(rawOffset, 512);
DataCopy(ubTensor, gmTensor[offset], dataSize);

// 实测带宽对比（GM→UB）：
// 512B 对齐: 100% 带宽效率
// 256B 对齐: ~90% 带宽效率
// 32B 对齐:  ~70% 带宽效率（最差情况）
```

Benefit: 带宽效率提升最高 30%（32B 对齐→512B 对齐）
Trade-off: 可能需要调整数据布局或增加 padding
