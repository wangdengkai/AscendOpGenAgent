# P68 低延迟归约指令组合 (Low-Latency Reduction Instruction Combination)
## Overview
归约操作中，WholeReduceSum 等归约指令延迟约为 Add 指令的 2-5 倍。通过二分累加（Add 指令逐步折半）将数据缩减到 256B 以内，再用 WholeReduceSum 一次完成最终归约，性能优于纯 WholeReduceSum 方案。进一步地，BlockReduceSum + WholeReduceSum 组合在特定 shape 下优于纯 WholeReduceSum（8.44us vs 13us，循环 100 次）。

## When to Use
- 连续数据的归约操作场景（如 ReduceSum）
- 大数据量归约（如 shape 30000 的 float 数据）
- Profiling 显示 Vector 归约耗时占比高

## Trade-off
- 二分累加方案代码复杂度较高
- 小数据量或特殊 shape 下需具体分析，不一定有收益
- 需配合 Counter 模式（SetMaskCount）使用效果更佳

**Source operators**: SIMD算子性能优化/矢量计算

---
## Variant A: 二分累加 + WholeReduceSum 组合归约
Source: SIMD算子性能优化/矢量计算/选择低延迟指令，优化归约操作性能.md

使用 Add 指令逐步二分折半，当数据量 ≤ 256B 时用 WholeReduceSum 完成最终归约。

**Expert implementation:**
```cpp
// 二分累加方案：shape(bsLength, hLength) → shape(bsLength, 1)
SetMaskCount();
for (uint32_t i = 0; i < bsLength; i++) {
    LocalTensor<float> srcTmp = src[i * hLength];
    LocalTensor<float> dstTmp = dst[i * hLength];
    uint32_t totalNum = hLength / 16 * 16;
    uint32_t remaining = hLength - totalNum;
    while (totalNum > ONE_REPEAT_FLOAT_SIZE) {
        uint32_t halfNum = DivCeil(totalNum, 16) * DEFAULT_REP_STRIDE;
        SetVectorMask<uint8_t, MaskMode::COUNTER>(0, totalNum - halfNum);
        Add<float, false>(dstTmp, srcTmp, srcTmp[halfNum], MASK_PLACEHOLDER, 1, binaryParams);
        totalNum = halfNum;
        srcTmp = dstTmp;
    }
    // 最终 WholeReduceSum
    SetVectorMask<uint8_t, MaskMode::COUNTER>(0, totalNum);
    WholeReduceSum<float, false>(dstTmp, srcTmp, MASK_PLACEHOLDER, 1, ...);
}
ResetMask(); SetMaskNorm();
```

Benefit: shape=30000 float 场景下 172 cycle vs WholeReduceSum 的 242 cycle（提升 29%）
Trade-off: 代码复杂度增加，需要额外的临时 buffer

---
## Variant B: BlockReduceSum + WholeReduceSum 最优组合
Source: SIMD算子性能优化/矢量计算/选择低延迟指令，优化归约操作性能.md

对于特定 shape（如 256 float），先用 BlockReduceSum 按 block 归约，再用 WholeReduceSum 完成最终归约。

**Expert implementation:**
```cpp
// BlockReduceSum + WholeReduceSum 组合（shape=256, float）
constexpr uint32_t c0Count = BLK_LEN / sizeof(DTYPE);
const uint32_t blockNum0 = (totalLength + c0Count - 1) / c0Count;
SetMaskCount();
SetVectorMask<DTYPE>(0, totalLength);
BlockReduceSum<DTYPE, false>(tempTensor1, xLocal, 1, MASK_PLACEHOLDER, ...);
PipeBarrier<PIPE_V>();
SetVectorMask<DTYPE>(0, blockNum0);
WholeReduceSum<DTYPE, false>(zLocal, tempTensor1, 1, MASK_PLACEHOLDER, ...);
PipeBarrier<PIPE_V>();
SetMaskNorm();
```

Benefit: 8.44us vs 纯 WholeReduceSum 的 13us 和纯 BlockReduceSum 的 13.94us（循环 100 次）
Trade-off: 需要额外的临时 buffer 空间
