# P81 Resident UB/TBuf Buffer Reuse (UB/TBuf 常驻 Buffer 复用)
## Overview
将小尺寸参数、跨迭代状态、梯度累加器或迭代式中间结果一次性搬入或初始化到 UB/TBuf 常驻 buffer，并在整个 `Process` 生命周期或多层循环中反复复用。该策略把原本每个 loop/batch/iteration 都要发生的 GM↔UB 搬运、Broadcast、Cast 或临时分配，改写为循环外一次初始化、循环内片上读写，适用于数据生命周期长且尺寸可控的参数与状态数据。

## When to Use
- 算子包含 weight、gamma、scale、offset 等小尺寸参数，且这些参数在所有 tile/loop 中保持不变
- 递推、归一化、梯度或迭代式算法需要在多个 loop/batch/iteration 之间保留状态或累加结果
- 常驻 buffer 占用 UB 后，仍能容纳主数据 tile、必要的双缓冲和临时计算空间
- 目标瓶颈来自重复 MTE2/MTE3 搬运、重复 Broadcast/Cast，或循环内反复分配临时 buffer
- 不适用于 Flash Attention online softmax 的 max/sum/exp 状态常驻；该类场景应优先参考 P33
- 对 n² 级增长的迭代式中间结果，需确保问题规模受控且迭代次数足够多（通常 ≥3）

## Trade-off
- 常驻 buffer 会长期占用 UB/TBuf，直接压缩主数据 tile 和双缓冲可用空间
- TBuf/VECCALC buffer 不受队列同步保护，需要手动使用 PipeBarrier 或明确阶段边界保证一致性
- 初始化阶段可能增加一次 Cast、Broadcast、Duplicate 或 Transpose 开销，仅在复用次数足够多时收益明显
- 累加器和迭代状态通常需要 FP28 中间结果，可能进一步增加 UB 占用

**Source operators**: ai_infra_aggregate_hidden, ai_infra_causal_conv1d_add, ai_infra_aggregate_hidden_grad, ai_infra_kv_rms_norm_rope_cache, ai_infra_sinkhorn_grad

---

## Variant A: Process 开头搬入 weight/scale 并全程复用
Source: ai_infra_aggregate_hidden, ai_infra_causal_conv1d_add, ai_infra_kv_rms_norm_rope_cache

在 `Process` 入口将 weight、gamma、scale 或 offset 搬入 UB，并按计算精度需要 Cast 到 FP28。后续 B×S、ubLoop 或 tile 循环中直接读取常驻 LocalTensor，不再重复发起 GM→UB 搬运。对于更小的参数张量，还可以利用 `ReinterpretCast` 在同一块 float buffer 的后半段暂存 half 数据，再原地 Cast 为 FP28，以减少额外 buffer 需求。

**Expert implementation:**
```cpp
__aicore__ inline void Process() {
    LocalTensor<float> weightFp32 = this->inQueueW.template AllocTensor<float>();
    DataCopyPad(weightLocal, weightGm, copyParams, padParams);
    Cast(weightFp32, weightLocal, RoundMode::CAST_NONE, alignBaseH);

    for (int64_t bIdx = 0; bIdx < baseB; ++bIdx) {
        for (int64_t sIdx = 0; sIdx < baseS; ++sIdx) {
            Compute(xLocalFp32, weightFp32, y0Fp32, y1Fp32, y2Fp32);
        }
    }
    this->inQueueW.FreeTensor(weightFp32);
}
```

**vs. baseline (lingxi-code):**
```cpp
for (int64_t bIdx = 0; bIdx < baseB; ++bIdx) {
    for (int64_t sIdx = 0; sIdx < baseS; ++sIdx) {
        DataCopyPad(weightLocal, weightGm, copyParams, padParams);
        Cast(weightFp32, weightLocal, RoundMode::CAST_NONE, alignBaseH);
        Compute(xLocalFp32, weightFp32, y0Fp32, y1Fp32, y2Fp32);
    }
}
```

Benefit: 小参数只搬运和 Cast 一次，MTE2 带宽留给主数据 tile，循环越深收益越明显。
Trade-off: weight/scale buffer 在整个 Process 期间常驻 UB，减少可用于数据 tile 的空间。

---

## Variant B: Broadcast 预计算后常驻
Source: ai_infra_aggregate_hidden_grad

当 weight 需要在 S 维或 batch 维广播时，在初始化阶段完成 Cast 和 Broadcast，并把扩展后的结果放入 TBuf 常驻。循环内直接读取广播后形状，避免每次迭代重复 Broadcast。

**Expert implementation:**
```cpp
TBuf<QuePosition::VECCALC> weightBroadCastBuf;

void InitWeightBuf() {
    LocalTensor<float> weight0 = weightBuf.Get<float>();
    Cast(weight0, weightLocal[0], RoundMode::CAST_NONE, calNum);
    uint32_t srcShape[2] = {1, static_cast<uint32_t>(calNum)};
    uint32_t dstShape[2] = {static_cast<uint32_t>(baseS), static_cast<uint32_t>(calNum)};
    Broadcast<float, 2, 0>(weightBroadCastBuf.Get<float>(), weight0, dstShape, srcShape);
}
```

**vs. baseline (lingxi-code):**
```cpp
for (int64_t idxS = 0; idxS < loopS; ++idxS) {
    Cast(weight0, weightLocal[0], RoundMode::CAST_NONE, calNum);
    Broadcast<float, 2, 0>(weightBroadcastLocal, weight0, dstShape, srcShape);
    ComputeWithBroadcastWeight(weightBroadcastLocal, idxS);
}
```

Benefit: 消除循环内重复 Broadcast，广播后 weight 直接参与向量计算。
Trade-off: 需要同时容纳原始、Cast 和 Broadcast 后 buffer，baseS tiling 受 UB 容量约束。

---

## Variant C: TBuf 常驻递推状态
Source: ai_infra_causal_conv1d_add, ai_infra_sinkhorn_grad

使用 `TBuf<VECCALC>` 声明跨迭代状态 buffer，在递推计算中持续读写 y0/y1/y2 或类似状态。状态 buffer 在 Init 后常驻，循环内只做片上更新。

**Expert implementation:**
```cpp
TBuf<TPosition::VECCALC> y0Buf_;
TBuf<TPosition::VECCALC> y1Buf_;
TBuf<TPosition::VECCALC> y2Buf_;

void ComputeAndUpdate() {
    LocalTensor<float> y0 = y0Buf_.Get<float>();
    LocalTensor<float> y1 = y1Buf_.Get<float>();
    LocalTensor<float> y2 = y2Buf_.Get<float>();
    Mul(y2, x32BufLocal, weight2, curDim_);
    Add(y2, y1, y2, curDim_);
    Mul(y1, x32BufLocal, weight1, curDim_);
    Add(y1, y0, y1, curDim_);
    Mul(y0, x32BufLocal, weight0, curDim_);
}
```

**vs. baseline (lingxi-code):**
```cpp
void ComputeAndUpdate() {
    LocalTensor<float> y0 = tmpQueue.AllocTensor<float>();
    LocalTensor<float> y1 = tmpQueue.AllocTensor<float>();
    LocalTensor<float> y2 = tmpQueue.AllocTensor<float>();
    DataCopy(y0, y0StateGm[offset], curDim_);
    DataCopy(y1, y1StateGm[offset], curDim_);
    UpdateState(y0, y1, y2);
    DataCopy(y0StateGm[offset], y0, curDim_);
    DataCopy(y1StateGm[offset], y1, curDim_);
}
```

Benefit: 状态在多次迭代之间保持在片上，避免反复分配、初始化和 GM 状态回写。
Trade-off: TBuf 不受队列同步保护，需手动插入 PipeBarrier 保证读写顺序。

---

## Variant D: 常驻梯度累加器跨 batch 累加
Source: ai_infra_aggregate_hidden_grad

将 gradWeight 或类似梯度累加器放入 TBuf，在 B×loopS 双层循环前初始化为 0，循环中持续累加，循环结束后一次性 ReduceSum 并搬出。

**Expert implementation:**
```cpp
TBuf<QuePosition::VECCALC> gradWeightBuf;

void Process() {
    LocalTensor<float> gradWeight = gradWeightBuf.Get<float>();
    Duplicate(gradWeight, 0.0f, baseS * calNum * 3);
    PipeBarrier<PIPE_V>();
    for (int64_t idxB = 0; idxB < batchSize; ++idxB) {
        for (int64_t idxS = 0; idxS < loopS; ++idxS) {
            GradWeightConv(sTileLen);
        }
    }
    SumAndCopyOutGradWeight();
}
```

**vs. baseline (lingxi-code):**
```cpp
for (int64_t idxB = 0; idxB < batchSize; ++idxB) {
    Duplicate(gradWeightTmp, 0.0f, baseS * calNum * 3);
    for (int64_t idxS = 0; idxS < loopS; ++idxS) {
        GradWeightConv(sTileLen);
    }
    DataCopy(gradWeightWorkspace[idxB], gradWeightTmp, copyLen);
}
ReduceSumAcrossBatch(gradWeightWorkspace, gradWeightOut);
```

Benefit: 避免每个 batch 单独搬出梯度再在 GM/workspace 累加，减少 MTE3 搬运量。
Trade-off: gradWeightBuf 通常是较大的常驻 buffer，直接约束 baseS 和 calNum tiling。

---

## Variant E: 迭代式中间结果常驻并原地更新
Source: ai_infra_sinkhorn_grad

对需要多轮原地更新的中间结果，首次转换或转置后放入 VECCALC buffer 常驻 UB。迭代循环中只做片上 col/row 或前后向更新，最后一次再转回目标布局并搬出。

**Expert implementation:**
```cpp
TBuf<TPosition::VECCALC> gradTransposeBuf_;

void Process() {
    TransposeXIn();
    for (int j = numIters_ - 1; j > 0; --j) {
        colNormGrad();
        rowNormGrad();
    }
    TransposeXOut();
    CopyOut(offset);
}
```

**vs. baseline (lingxi-code):**
```cpp
for (int j = numIters - 1; j > 0; --j) {
    DataCopy(gradLocal, gradGm[offset], len);
    Transpose(gradTransLocal, gradLocal, params);
    colNormGrad(gradTransLocal);
    rowNormGrad(gradTransLocal);
    Transpose(gradLocal, gradTransLocal, params);
    DataCopy(gradGm[offset], gradLocal, len);
}
```

Benefit: 多次迭代期间零 DMA，所有更新在片上完成，迭代次数越多收益越高。
Trade-off: 常驻 buffer 大小可能按 n² 增长，仅适合尺寸受控且迭代次数足够多的场景。
