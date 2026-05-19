# P84 Vector Compute Efficiency Patterns (UB 内 Vector 计算效率优化模式)
## Overview
在纯 Vector 或 Vector 主导的算子中，性能瓶颈常来自三类 UB 内部效率损失：Scalar 侧为每个 Vector API 手动计算 repeat/mask，归约场景反复使用高延迟 reduce 指令，以及连续 Vector 计算链把中间结果写回 GM 后再读回 UB。该策略将这些问题统一为“Vector 计算阶段内部效率优化”：优先用 Counter 模式减少控制指令，用低延迟指令组合替代单一高延迟归约路径，并把连续 Vector 链的中间结果保留在 UB/VECCALC 中直通消费。它不改变算子数学逻辑和外部接口，主要通过 Vector API 调用方式、指令组合和 UB 内数据生命周期调整减少 Scalar 开销、Vector 指令延迟和 GM 往返搬运。三者均作用于单核 UB 内的 Vector 执行路径，不涉及分核策略、跨核同步或跨循环状态管理。

## When to Use
- 算子主体是 Vector 计算，且 profiling 显示 Scalar 控制、Vector reduce 或 MTE2/MTE3 中间搬运占比较高
- Vector API 处理的元素数较大或尾块较多，手动 repeatTimes/tail mask 计算带来明显 Scalar 开销
- ReduceSum 等归约操作中纯 `WholeReduceSum` 延迟较高，且可接受额外临时 UB buffer 做分阶段归约
- 多步 Vector 计算前后依赖明确，例如 `Exp -> Abs -> Mul`，中间结果无需在每步之间对外写回 GM
- 适用于 UB 容量足以容纳输入、输出和少量临时中间结果的场景
- 三个 Variant 对应不同 profiling 瓶颈，通常按实测信号单独选用，不需要同时应用

## Trade-off
- Counter 模式依赖硬件和 API 支持，使用后必须正确恢复 mask 模式，避免影响后续 Vector API
- 低延迟归约组合会增加临时 buffer、`PipeBarrier` 和 shape-specific 分支，复杂度高于直接调用单个 reduce API
- UB 融合连续 Vector 链会延长中间 tensor 生命周期，链条过长或 dtype 较宽时可能挤占双缓冲空间
- 对极小数据量、单次 Vector API 或归约长度很短的场景，额外 mask 设置和临时 buffer 管理可能抵消收益

**Source operators**: SIMD算子性能优化/矢量计算

---

## Variant A: Counter 模式消除 Scalar mask 控制
Source: SIMD算子性能优化/矢量计算/Vector算子灵活运用Counter模式.md

当 Vector API 的主块/尾块处理需要大量 Scalar 计算时，可用 `SetMaskCount()` 和 `MaskMode::COUNTER` 直接指定总元素数，让一次 Vector API 调用覆盖完整数据区间，避免手动计算 `repeatTimes`、`tailSize` 和尾块 mask。

**Expert implementation:**
```cpp
// Normal 模式（反例）：手动计算 repeatTimes 和 tail mask
uint32_t repeatTimes = dataSize / ONE_REPEAT_SIZE;
uint32_t tailSize = dataSize % ONE_REPEAT_SIZE;
Add(dst, src1, src2, FULL_MASK, repeatTimes, {...});
if (tailSize > 0) {
    SetVectorMask(tailMask);
    Add(dst[offset], src1[offset], src2[offset], tailMask, 1, {...});
}

// Counter 模式（正例）：按总元素个数一次性覆盖
SetMaskCount();
SetVectorMask<MaskMode::COUNTER>(ELE_SIZE);
Add(dst, src1, src2, MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
ResetMask();
```

**vs. baseline (lingxi-code):**
```cpp
for (uint32_t loop = 0; loop < loopCount; ++loop) {
    uint64_t mask = CalcMask(loop, totalLength);
    uint32_t repeat = CalcRepeat(loop, totalLength);
    Add(dst[offset], src0[offset], src1[offset], mask, repeat, params);
}
```

Benefit: 减少 repeat/tail/mask 的 Scalar 计算和分支，降低 Vector loop 的控制开销。
Trade-off: 需要在 Counter 模式使用后恢复 mask 状态，且极小数据量收益有限。

---

## Variant B: 低延迟归约指令组合
Source: SIMD算子性能优化/矢量计算/选择低延迟指令，优化归约操作性能.md

归约长度较大时，单纯依赖 `WholeReduceSum` 可能受到高延迟 reduce 指令限制。可先用 Add 二分累加或 `BlockReduceSum` 将数据压缩到较小规模，再用一次 `WholeReduceSum` 完成最终归约；具体组合应根据 shape、dtype 和临时 UB 预算选择。

**Expert implementation:**
```cpp
// 方案一：二分累加 + WholeReduceSum
SetMaskCount();
for (uint32_t i = 0; i < bsLength; i++) {
    LocalTensor<float> srcTmp = src[i * hLength];
    LocalTensor<float> dstTmp = dst[i * hLength];
    uint32_t totalNum = hLength / 16 * 16;
    while (totalNum > ONE_REPEAT_FLOAT_SIZE) {
        uint32_t halfNum = DivCeil(totalNum, 16) * DEFAULT_REP_STRIDE;
        SetVectorMask<uint8_t, MaskMode::COUNTER>(0, totalNum - halfNum);
        Add<float, false>(dstTmp, srcTmp, srcTmp[halfNum], MASK_PLACEHOLDER, 1, binaryParams);
        totalNum = halfNum;
        srcTmp = dstTmp;
    }
    SetVectorMask<uint8_t, MaskMode::COUNTER>(0, totalNum);
    WholeReduceSum<float, false>(dstTmp, srcTmp, MASK_PLACEHOLDER, 1, ...);
}
ResetMask();
SetMaskNorm();

// 方案二：BlockReduceSum + WholeReduceSum
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

**vs. baseline (lingxi-code):**
```cpp
SetMaskCount();
SetVectorMask<float>(0, totalLength);
WholeReduceSum<float, false>(outLocal, xLocal, MASK_PLACEHOLDER, repeat, params);
SetMaskNorm();
```

Benefit: 在大长度或特定 shape 归约中，用低延迟 Add/BlockReduceSum 缩短主归约路径，减少纯 `WholeReduceSum` 延迟。
Trade-off: 需要额外临时 buffer 和 shape 调参；对小 shape 或非连续归约不一定优于直接 reduce。若瓶颈是按固定间隔提取元素的 stride 搬运而非归约延迟，应参考 P41。

---

## Variant C: UB 融合连续 Vector 计算链
Source: SIMD算子性能优化/矢量计算/通过Unified_Buffer融合实现连续vector计算.md

当多个 Vector API 串联且前一步输出仅作为后一步输入时，应把中间结果留在 UB/VECCALC 中继续消费，而不是每一步都 `CopyOut` 到 GM 再 `CopyIn`。该模式适用于纯 Vector 链或局部 Vector 后处理链，且中间结果生命周期仅限于当前 `Process` 调用内的单次 Vector 链，不跨迭代、不跨 loop；跨迭代状态常驻场景应参考 P67 Variant E。它不适用于中间结果必须跨 kernel、跨 core 或被外部算子观察的场景。

**Expert implementation:**
```cpp
// 反例：Exp 后搬出 GM，再搬入做 Abs（4 次 GM 搬运）
void Process() {
    CopyIn();
    Compute_Exp();
    CopyOut();
    CopyIn1();
    Compute_Abs();
    CopyOut1();
}

// 正例：UB 融合，仅首次读 GM 和最终写 GM
void Compute() {
    LocalTensor<float> src0Local = inQueueSrc0.DeQue<float>();
    LocalTensor<float> dstLocal = outQueueDst.AllocTensor<float>();
    Exp(dstLocal, src0Local, 1024);
    Abs(dstLocal, dstLocal, 1024);
    outQueueDst.EnQue<float>(dstLocal);
    inQueueSrc0.FreeTensor(src0Local);
}
```

**vs. baseline (lingxi-code):**
```cpp
for (uint32_t stage = 0; stage < vectorStages; ++stage) {
    CopyIn(stage);
    RunVectorStage(stage);
    CopyOut(stage);
}
```

Benefit: n 段连续 Vector 计算的 GM 搬运次数可从 2n 次降到 2 次，显著减少中间 MTE2/MTE3 开销。
Trade-off: 中间结果占用 UB 时间更长，需确认链条长度和 tile 大小不会破坏原有双缓冲容量。
