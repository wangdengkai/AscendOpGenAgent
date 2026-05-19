# P34: Weight 常驻与预计算复用

## Overview
将 weight/scale 等小尺寸、跨 loop 不变的数据在 Process 开头搬入 UB 并 cast 到 FP32，之后在整个 B×S 循环中反复使用。对于需要广播的 weight，还可在搬入后做 Broadcast 扩展到目标形状常驻 UB（见 Variant B）。

## When to Use
- 算子包含小尺寸常量参数（weight/scale/bias），大小不超过 UB 总容量的 5%（如 192KB UB 上 ≤10KB）
- 参数在整个 Process 的 B×S 双层循环中保持不变，可一次搬入全程复用
- UB 空间在扣除常驻 weight buffer 后仍足够容纳数据 tile 的双缓冲
- weight 需要在 S 维度广播的梯度计算算子，广播后结果可常驻 UB 不超过总容量的 10%

## Trade-off
- weight buffer 在整个 Process 期间常驻 UB，减少可用于数据 tile 的空间
- 需要在 Process 开头额外做一次 Cast（如 half→float），增加初始化延迟
- Broadcast 预计算需要额外 UB 空间（原始 + Cast + Broadcast 三份），仅适用于单维度广播

**Source operators**: ai_infra_aggregate_hidden, ai_infra_causal_conv1d_add

---

## Variant A: Process 开头搬入 weight 并 cast 后全程复用
Source: ai_infra_aggregate_hidden, ai_infra_causal_conv1d_add

在 Process 函数入口搬入 weight 并 Cast 到 FP32，整个 B×S 双层循环中直接使用，避免重复搬运。

```cpp
__aicore__ inline void Process() {
    LocalTensor<float> weightFp32 = this->inQueueW.template AllocTensor<float>();
    DataCopyPad(weightLocal, weightGm, copyParams, padParams);
    Cast(localW0FP32, localW0, RoundMode::CAST_NONE, this->alignBaseH);
    Cast(localW1FP32, localW1, RoundMode::CAST_NONE, this->alignBaseH);
    for (int64_t bIdx = 0; bIdx < this->baseB; ++bIdx) {
        for (int64_t sIdx = 0; sIdx < this->baseS; ++sIdx) {
            Compute<DTYPE>(xLocalFp32, weightFp32, y0Fp32, y1Fp32, y2Fp32, ...);
        }
    }
    this->inQueueW.FreeTensor(weightFp32);
}
```

Benefit: 消除 weight 重复搬运开销，MTE2 带宽留给数据 tile
Trade-off: weight buffer 常驻 UB，减少可用于数据 tile 的空间

---

## Variant B: Weight Broadcast 预计算常驻
Source: ai_infra_aggregate_hidden_grad

在初始化阶段将 weight Cast 到 FP32 后 Broadcast 到目标形状，存入 TBuf 常驻 UB。

```cpp
TBuf<QuePosition::VECCALC> weightBroadCastBuf;
void InitWeightBuf() {
    Cast(weight0, weightLocal[0], RoundMode::CAST_NONE, calNum);
    uint32_t srcShape[2] = {1, static_cast<uint32_t>(calNum)};
    uint32_t dstShape[2] = {static_cast<uint32_t>(baseS), static_cast<uint32_t>(calNum)};
    Broadcast<float, 2, 0>(weightBroadCastBufLocal[0], weight0, dstShape, srcShape);
    Broadcast<float, 2, 0>(weightBroadCastBufLocal[baseS*calNum], weight1, dstShape, srcShape);
}
```

Benefit: 消除每次迭代的 Broadcast 开销，广播后 weight 直接参与向量计算
Trade-off: 占用 3×baseS×calNum×4 字节 UB 空间，baseS 受此约束
