# P37: 梯度累加器跨 batch 常驻

## Overview
gradWeight 需要在所有 batch 和 seq 上累加。使用 TBuf 分配常驻 UB，在整个 B×loopS 循环中持续累加，最后一次性 ReduceSum 并搬出。

## When to Use
- 训练算子中需要跨 batch 累加梯度的场景
- 梯度维度较小（如 calNum），可在 UB 中常驻累加器而不影响数据 tile 的双缓冲

## Trade-off
- gradWeightBuf 是 UB 中最大的常驻 buffer 之一，baseS 的 tiling 直接受此约束
- 最终需要 ReduceSum 搬出，batch 数很大时累加精度可能需要 FP32 中间结果

**Source operators**: ai_infra_aggregate_hidden_grad

---

## Variant A: TBuf 常驻 gradWeight 跨 batch 累加
Source: ai_infra_aggregate_hidden_grad

使用 TBuf 分配 gradWeight buffer，初始化为 0 后在 B×loopS 双层循环中持续累加，循环结束后 ReduceSum 并搬出。

```cpp
TBuf<QuePosition::VECCALC> gradWeightBuf;
void Process() {
    LocalTensor<float> gradWeight = gradWeightBuf.Get<float>();
    Duplicate(gradWeight, 0.0f, baseS * calNum * 3);
    PipeBarrier<PIPE_V>();
    for (int64_t idxB = 0; idxB < batchSize; idxB++) {
        for (int64_t idxS = 0; idxS < loopS; idxS++) {
            GradWeightConv(sTileLen);
        }
    }
    SumAndCopyOutGradWeight();
}
```

Benefit: 避免每个 batch 单独搬出梯度再在 GM 累加，减少 MTE3 搬运量
Trade-off: gradWeightBuf 常驻 UB，是最大的常驻 buffer，直接约束 baseS tiling
