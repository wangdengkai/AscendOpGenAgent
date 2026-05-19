# P26: Stride 搬运模式（转置与列提取）

## Overview
通过 srcStride=0（连读）+ dstStride=跳步（跳写）实现零拷贝转置，将 UB 上连续数据写回 GM 上不连续布局。反向地，通过 blockCount/blockLen/srcStride 参数还可实现非连续列的提取搬运，一次 DataCopyPad 搬运多行，每行跳过无关列（见 Variant B）。

## When to Use
- FlashAttention 输出需要做 [G,S1,D] → [N,G,B,S,D] 的布局转换
- 转置可表达为 srcStride=0（连读）+ dstStride=跳步（跳写）的模式，即源数据连续、目标数据等间距分布
- 输入张量包含多个子字段（如 RMS_NORM_LENGTH + ROPE_LENGTH），需要分别提取，子字段在最内维上连续且长度固定

## Trade-off
- DataCopyParams stride 为 uint16_t 最大 65535；超限需切换到 DataCopyExtParams
- 仅适用于单维度转置，多维度复杂重排需要组合多次 stride 搬运或使用 Gather
- 列提取时 srcStride 和 blockLen 必须满足 32B 对齐，每次提取一个子字段需多次搬运调用

**Source operators**: common/vector_common.h

---

## Variant A: srcStride=0 连读 + dstStride 跳写融合转置
Source: common/vector_common.h

利用 DataCopyExtParams 的 srcStride=0 实现连续读取，dstStride 设置为跳步值实现写回时的布局转换，一次搬运完成转置。

```cpp
DataCopyExtParams dataCopyParams;
dataCopyParams.blockCount = gCountOneS1;
dataCopyParams.blockLen = headDim * sizeof(OUT_T);
dataCopyParams.srcStride = 0;                                     // 连读
dataCopyParams.dstStride = (tSize - 1) * headDim * sizeof(OUT_T); // 跳写
DataCopyPad(attentionOutGm[attenOutOffset], attenOutUb[attenOutUbOffset], dataCopyParams);
```

Benefit: 零额外拷贝完成布局转换，节省 UB 空间和搬运带宽
Trade-off: DataCopyParams stride 为 uint16_t 最大 65535；超限需切换到 DataCopyExtParams

---

## Variant B: blockLen + srcStride 非连续列提取
Source: ai_infra_kv_rms_norm_rope_cache

blockLen 指定每行需要提取的字节数，srcStride 指定每行跳过的字节数，一次 DataCopyPad 完成多行非连续列的提取搬运。

```cpp
DataCopyExtParams copyParams{
    static_cast<uint16_t>(ubFactor),
    static_cast<uint32_t>(RMS_NORM_LENGTH * sizeof(KV_DTYPE)),
    static_cast<uint32_t>(ROPE_LENGTH * sizeof(KV_DTYPE)),
    0, 0};
DataCopyPad(xLocal, kvGm[kvGlobalMemoryOffset], copyParams, padParams);
```

Benefit: 一次搬运提取目标列，避免搬运全部数据后再做 UB 上的列裁剪
Trade-off: srcStride 和 blockLen 必须满足 32B 对齐
