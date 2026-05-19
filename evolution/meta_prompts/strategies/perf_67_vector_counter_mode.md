# P67 Vector Counter 模式 (Vector Counter Mode)
## Overview
Normal 模式下 Vector API 需手动计算 repeatTimes、区分主块/尾块、多次设置 mask，涉及大量 Scalar 计算。Counter 模式下只需 SetMaskCount() 后设置 mask 为总元素个数，一次 API 调用完成所有计算，无需主尾块判断，大幅减少指令数量和 Scalar 计算量。

## When to Use
- 所有 Vector 矢量计算 API 场景
- 大数据量场景（如 15000 个元素）
- Profiling 显示 Scalar 时间占比高

## Trade-off
- 需要硬件支持 Counter 模式
- 对极小数据量可能无明显收益

**Source operators**: SIMD算子性能优化/矢量计算

---
## Variant A: SetMaskCount 替代手动 mask 计算
Source: SIMD算子性能优化/矢量计算/Vector算子灵活运用Counter模式.md

使用 Counter 模式一次调用完成所有元素计算。

**Expert implementation:**
```cpp
// Normal 模式（反例）：手动计算 repeatTimes 和 mask
uint32_t repeatTimes = dataSize / ONE_REPEAT_SIZE;
uint32_t tailSize = dataSize % ONE_REPEAT_SIZE;
Add(dst, src1, src2, FULL_MASK, repeatTimes, {...});
if (tailSize > 0) {
    SetVectorMask(tailMask);
    Add(dst[offset], src1[offset], src2[offset], tailMask, 1, {...});
}

// Counter 模式（正例）：一次调用完成
SetMaskCount();
SetVectorMask<MaskMode::COUNTER>(ELE_SIZE);  // 总元素个数
Add(dst, src1, src2, MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
ResetMask();
```

Benefit: 大幅简化代码，降低 Scalar 和 Vector 计算耗时，减少指令条数
Trade-off: 需要硬件支持 Counter 模式
