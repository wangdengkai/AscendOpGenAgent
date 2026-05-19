# P24: ND↔NZ 格式转换搬运

## Overview
在 GM→L1 搬运阶段同时完成 ND→NZ 格式转换，避免额外 TransData 指令。对齐到 16 元素（fp16）或 32 元素（fp8）的分形边界。反向地，写出时可通过 ScatterUpdateNZ 逐 token 完成 ND→NZ 格式转换，避免额外 TransData 算子（见 Variant B）。

## When to Use
- Matmul 输入需要 NZ 格式，GM 上数据为 ND 连续格式
- 最内维满足分形对齐要求（fp16 为 16 元素、fp8 为 32 元素），否则需先做 padding
- KV Cache 写入需要 NZ 格式的推理算子，写入 token 数较少时逐 token 循环开销可接受

## Trade-off
- 对齐 padding 浪费部分 L1 空间；不同数据类型对齐基数不同需条件编译
- 仅支持 GM→L1 方向的格式转换，L1→L0 仍需标准 LoadData 路径
- 写出方向逐 token 循环 MTE3 利用率低，需要 SToMTE3Sync 引入流水气泡

**Source operators**: common/CopyInL1.h, ai_infra_sparse_flash_attention_gqa

---

## Variant A: GM→L1 搬运时融合 ND2NZ 格式转换
Source: common/CopyInL1.h

通过 Nd2NzParams 参数控制搬运过程中的格式转换，dstNzC0Stride 按 16 元素对齐，一次 DataCopy 同时完成搬运和格式变换。

```cpp
template<typename INPUT_T>
__aicore__ inline void CopyToL1Nd2Nz(const LocalTensor<INPUT_T> &l1Tensor,
    const GlobalTensor<INPUT_T> &gmTensor,
    uint32_t nValue, uint32_t dValue, uint32_t srcDValue) {
    Nd2NzParams gm2L1Nd2NzParams;
    gm2L1Nd2NzParams.nValue = nValue;
    gm2L1Nd2NzParams.dValue = dValue;
    gm2L1Nd2NzParams.srcDValue = srcDValue;
    gm2L1Nd2NzParams.dstNzC0Stride = (nValue + 15) >> 4 << 4;
    gm2L1Nd2NzParams.dstNzNStride = 1;
    DataCopy(l1Tensor, gmTensor, gm2L1Nd2NzParams);
}
```

Benefit: 省去独立 TransData 指令，减少一次 L1 读写往返
Trade-off: 对齐 padding 浪费部分 L1 空间；不同数据类型对齐基数不同需条件编译

---

## Variant B: ScatterUpdateNZ 逐 token ND→NZ 写出
Source: ai_infra_kv_rms_norm_rope_cache

通过 DataCopyExtParams 设置 NZ 格式的 stride 参数，逐 token 循环写出，每次写出前用 SToMTE3Sync 保证数据就绪。

```cpp
template <typename T, bool isPA, int64_t D1, int64_t D0>
__aicore__ inline void ScatterUpdateNZ(
    const GlobalTensor<T>& dst, ..., int64_t rows, int64_t headSize) {
    DataCopyExtParams copyParamsNz{
        static_cast<uint16_t>(D1),
        static_cast<uint32_t>(D0 * sizeof(T)), 0,
        static_cast<uint32_t>(blockSize * D0 * sizeof(T) - D0 * sizeof(T)), 0};
    for (int64_t i = 0; i < rows; i++) {
        int64_t gmOffsetNz = pageId * D1 * blockSize * D0 + tokenOffsetInCurrentPage * D0;
        SToMTE3Sync();
        DataCopyPad(dst[gmOffsetNz], outLocal[ubOffset], copyParamsNz);
    }
}
```

Benefit: 避免额外 TransData 算子，在写出时完成格式转换
Trade-off: 逐 token 循环写出 MTE3 利用率低，SToMTE3Sync 引入流水气泡
