# P41: gamma/scale/offset 常驻 UB
## Overview
gamma（RmsNorm 权重）、quantization scale/offset 等小张量在整个 kernel 生命周期内只搬入一次，常驻 UB 供所有 ubLoop 迭代复用。利用 ReinterpretCast 在同一块 buffer 中先存 half 再原地 Cast 为 fp32。

## When to Use
- 包含 RmsNorm/LayerNorm 权重或量化 scale 的推理算子
- 参数大小较小（通常 ≤2KB），常驻 UB 不影响数据 tile 的正常分配

## Trade-off
- 常驻 buffer 占用约 2KB UB 空间
- ReinterpretCast 原地 Cast 技巧要求 float buffer 后半段空间足够存放 half 数据

**Source operators**: ai_infra_kv_rms_norm_rope_cache

---

## Variant A: ReinterpretCast 原地 Cast + 常驻复用
Source: ai_infra_kv_rms_norm_rope_cache

分配 fp32 大小的 buffer，用 ReinterpretCast 偏移到后半段存 half 数据，再原地 Cast 为 fp32，整个 ubLoop 中不再重新搬运。

**Expert implementation:**
```cpp
LocalTensor<float> gammaLocalFp32 = inQueueGamma.AllocTensor<float>();
LocalTensor<KV_DTYPE> gammaLocal = gammaLocalFp32.ReinterpretCast<KV_DTYPE>()[RMS_NORM_LENGTH];
DataCopyPad(gammaLocal, gammaGm, copyParams, padParams);
Cast(gammaLocalFp32, gammaLocal, RoundMode::CAST_NONE, RMS_NORM_LENGTH);
// gammaLocalFp32 在所有 ubLoop 中复用
```

Benefit: 小张量只搬运一次，消除重复 DMA 开销
Trade-off: 常驻 buffer 占用固定 UB 空间，headSize 较大时需注意总量
