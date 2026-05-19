# P56 DataCopy UB2GM 合并输出 (Batch DataCopy Output)
## Overview
将多次小批量 DataCopy UB2GM 操作合并为一次大批量写回，减少 MTE3 指令发射次数，提升写回效率。

## When to Use
- 循环内每次迭代都有独立的 DataCopy UB2GM 操作
- 多次计算的结果可以累积后一次性写回
- 单次写回数据量不超过 UB 容量

## Trade-off
- 需要额外的 UB 空间累积结果
- 增加代码复杂度，需要管理累积计数
- 首次迭代可能有额外判断逻辑

**Source operators**: IFA nUpdate 输出

---

## Variant A: 循环内累积，循环外统一写回
Source: 【案例总结】OBP IFA优化点汇总/V1的nUpdate UB2GM拷贝优化.md

原始方案在循环内每次 vec 计算后立即 DataCopy，优化方案将多次计算结果累积到 UB，循环结束后统一写回。

**Expert implementation:**
```cpp
// 原始方案：循环内每次都 DataCopy
template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::ProcessVec1Inner(const ExtraInfoMla &info) {
    for (uint32_t i = 0; i < loopCount; i++) {
        DealBmm1ResBaseBlock(info, ...);  // 包含 DataCopy
    }
}

// 优化方案：循环外统一 DataCopy
template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::ProcessVec1Inner(const ExtraInfoMla &info) {
    LocalTensor<int32_t> nInt32Out = outputQue2.template AllocTensor<int32_t>();

    // 循环内累积，不写回
    for (uint32_t i = 0; i < loopCount; i++) {
        DealBmm1ResBaseBlock(info, nInt32Out, ...);  // 不包含 DataCopy
    }

    // 循环外统一写回
    outputQue2.EnQue(nInt32Out);
    outputQue2.DeQue<int32_t>();
    uint32_t dealRowCount = (loopCount - 1) * gSplitSize + tailSplitSize;
    DataCopy(nUpdateGm[...], nInt32Out, dealRowCount);  // 一次性写回
    outputQue2.FreeTensor(nInt32Out);
}
```

**vs. baseline (lingxi-code):**
```cpp
// 基线：循环内多次小 DataCopy
for (int i = 0; i < loopCount; i++) {
    Compute();
    DataCopy(gm[offset], ub, size);  // 每次 16 个元素
}
```

Benefit: 减少 MTE3 指令发射次数，从 loopCount 次降为 1 次，显著提升写回效率
Trade-off: 需要管理累积计数和尾块处理；UB 占用增加