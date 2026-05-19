# P25: DataCopyPad 带 Padding 的搬运

## Overview
当源数据列数不对齐 32B 时，使用 DataCopyPad + DataCopyPadExtParams 在搬运过程中自动补零对齐。

## When to Use
- 输入张量最内维不满足 32B 对齐（如 fp16 下 shape 非 16 的倍数、fp32 下非 8 的倍数），后续 Vector 计算要求对齐
- GM→UB 搬运阶段需要自动补零，避免在 UB 上手动做 padding 操作
- 不适用于 padding 量超过 255 字节的场景（DataCopyPadExtParams 的 rightPadding 为 uint8_t）

## Trade-off
- padding 参数为 uint8_t 最大补零 255 字节；blockLen 单位为字节需注意转换
- 补零后的数据参与后续计算，需确保补零不影响算法正确性（如 ReduceSum 场景补零安全，但 ReduceMax 可能受影响）

**Source operators**: common/vector_common.h, ai_infra_kv_rms_norm_rope_cache

---

## Variant A: DataCopyPad 自动补零对齐搬运
Source: ai_infra_kv_rms_norm_rope_cache

通过 DataCopyPadExtParams 指定右侧补零字节数，搬运过程中自动将非对齐列补齐到 32B 边界。

```cpp
uint32_t attenMaskSizeAlign = Align(info.s2dealNum, 32U);
DataCopyExtParams dataCopyParams;
dataCopyParams.blockCount = s1EndIdx - s1StartIdx;
dataCopyParams.blockLen = info.s2dealNum;
dataCopyParams.srcStride = info.attenMaskStride - info.s2dealNum;
dataCopyParams.dstStride = 0;
DataCopyPadExtParams<bool> padParams{true, 0,
    static_cast<uint8_t>(attenMaskSizeAlign - info.s2dealNum), 0};
DataCopyPad(attenMaskUb, srcGmAddr[maskOffset], dataCopyParams, padParams);
```

Benefit: 一次搬运完成对齐，后续 Vector 计算无需额外处理非对齐尾部
Trade-off: padding 参数为 uint8_t 最大补零 255 字节；blockLen 单位为字节需注意转换
