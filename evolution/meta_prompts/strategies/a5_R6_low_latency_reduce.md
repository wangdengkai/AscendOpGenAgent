# R6: 低延迟归约

## Overview
选择最优的归约指令组合，减少 Reduce 类操作的延迟。

## When to Use
- ReduceSum/ReduceMax/ReduceMin 操作
- Profiling 显示归约操作是瓶颈
- 数据量较大（> VL 元素）

## Trade-off
- **收益**: 归约延迟降低 2-3x
- **风险**: 需要额外 RegTensor 做累加器
- **复杂度**: 中等

## Variant A: BlockReduceSum + WholeReduceSum

```cpp
// Baseline: 单次 WholeReduceSum（高延迟）

// Evolved: 先按 VL 粒度累加，再全局归约
RegTensor<float> accReg;
MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
Duplicate(accReg, 0.0f);

for (uint16_t i = 0; i < repTimes; ++i) {
    MaskReg mask = UpdateMask<float>(count);
    LoadAlign(srcReg, srcAddr + i * oneRepSize);
    Add(accReg, accReg, srcReg, mask);  // VL 粒度累加
}

// 最终归约: VL → 标量
StoreAlign(workAddr, accReg, fullMask);
// 在 VF 外部用高阶 API 完成最终归约
```

## Variant B: 二分累加（repTimes 为 2 的幂）

适用于小数据量，利用树形归约减少依赖链深度。
