# P38: Softmax 状态 Buffer 跨 S2 循环常驻

## Overview
Flash Attention 的 online softmax 需要在 S2 方向多次循环中累积 softmaxMax/Sum/Exp 三个状态。这些 buffer 在 InitBuffers 中一次性分配，通过 loop % preLoadNum 索引实现双缓冲复用。

## When to Use
- Flash Attention / Sparse Flash Attention 的 Vector 后处理阶段
- online softmax 需要跨 S2 循环维护 max/sum/exp 三个状态，不能每次重新分配

## Trade-off
- 常驻占用 3×2K×preLoadNum = 12KB UB 空间
- preLoadNum（双缓冲深度）增大可提升流水重叠率，但线性增加 UB 占用

**Source operators**: ai_infra_sparse_flash_attention_gqa, ai_infra_fused_infer_attention_sink

---

## Variant A: softmax 状态三 buffer 常驻 + 双缓冲索引
Source: ai_infra_sparse_flash_attention_gqa, ai_infra_fused_infer_attention_sink

在 InitBuffers 中分配 softmaxMax/Exp/Sum 三个常驻 buffer，通过 loop % preLoadNum 索引实现双缓冲复用。

```cpp
pipe->InitBuffer(softmaxMaxBuff, SOFTMAX_TMP_BUFFER_SIZE * constInfo.preLoadNum);
pipe->InitBuffer(softmaxExpBuff, SOFTMAX_TMP_BUFFER_SIZE * constInfo.preLoadNum);
pipe->InitBuffer(softmaxSumBuff, SOFTMAX_TMP_BUFFER_SIZE * constInfo.preLoadNum);

uint32_t outIdx = info.loop % (constInfo.preLoadNum);
uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T);
SoftmaxFlashV2<...>(mmResUb, softmaxSumUb[softmaxOutOffset],
    softmaxMaxUb[softmaxOutOffset], mmResUb,
    softmaxExpUb[softmaxOutOffset], inSumTensor, inMaxTensor, ...);
```

Benefit: 状态 buffer 常驻避免每次 S2 循环重新分配，双缓冲索引支持流水重叠
Trade-off: 常驻占用 3×2K×preLoadNum = 12KB UB 空间

<!-- TODO: 当前仅 1 个 variant，深度不足。后续可补充：
     Variant B: 非 Flash Attention 场景的 softmax 状态常驻（如 Decoder Attention）
     Variant C: preLoadNum > 2 的多缓冲变体 -->
