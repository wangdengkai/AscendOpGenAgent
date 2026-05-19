# P43: 反向梯度 4路PingPong 常驻与 Scatter-Add 延迟写回
## Overview
反向传播中 dK/dV 梯度累积在 workspace 的 4 路 PingPong buffer 中（selectdKPPPidx % 4 轮转），延迟到整个 S1 处理完毕后再统一 scatter-add，将 scatter-add 从 O(S2) 次降为 O(S1) 次。

## When to Use
- 稀疏 Attention 反向传播中需要按 topK 索引 scatter-add 梯度
- S2 维度远大于 S1 维度，延迟 scatter-add 可将写出次数从 O(S2) 降为 O(S1)

## Trade-off
- 4 路 PingPong workspace 占用 = 4 × buffer_size × 核数，HBM 开销显著
- 跨核同步逻辑复杂，changeS1 条件判断和 ScatterAddByS1 触发时机需要精确控制

**Source operators**: sparse_flash_attention_grad_enhance

---

## Variant A: 4 路 PingPong + 延迟 Scatter-Add
Source: sparse_flash_attention_grad_enhance

梯度累积在 4 路 workspace buffer 中轮转写入，仅在 S1 边界触发 scatter-add，大幅减少 scatter 次数。

**Expert implementation:**
```cpp
selectedKGmOffset = selectdKPPPidx * selectedKWspOffset;
mmPingPongIdx = 1 - mmPingPongIdx;
selectdKPPPidx = (selectdKPPPidx + 1) % 4;
if (scatterRunInfo.changeS1) {
    CrossCoreWaitFlag<2, PIPE_MTE2>(SCATTER_SYNC_FLAG);
    ScatterAddByS1(vecOp, actual_seq_qlen, actual_seq_kvlen);
}
```

Benefit: scatter-add 次数从 O(S2) 降为 O(S1)，显著减少 GM 写回开销
Trade-off: 4 路 workspace 占用 4× 单路大小；跨核同步逻辑复杂
