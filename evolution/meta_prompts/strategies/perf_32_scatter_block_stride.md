# P32: Scatter Block Update 的 stride 寻址

## Overview
支持输入张量 dim-0 非连续（通过 inputStride0_ 和 inputStride1_），实现 input[indices[k,0], indices[k,1], :] = update[k, :] 的散射写入。

## When to Use
- 需要按索引散射更新非连续张量的 index 类算子
- 输出张量的 stride 不等于 shape 最内维（即非连续存储），无法用标准 DataCopy 直接写出

## Trade-off
- 逐行写出每次只搬 1 行 MTE3 效率低
- indices 需要先搬入 UB 解析，增加额外的 MTE2 搬运和标量计算开销

**Source operators**: ai_infra_scatter_block_update

---

## Variant A: stride 寻址逐行 Scatter 写出
Source: ai_infra_scatter_block_update

通过 indices 计算 GM 偏移，使用 inputStride0_ 和 inputStride1_ 支持非连续张量寻址，逐行搬出 update 数据。

```cpp
for (int64_t i = 0; i < loadCount; i++) {
    IndexT idx0 = indLocal.GetValue(i * INDICES_LAST_DIM);
    IndexT idx1 = indLocal.GetValue(i * INDICES_LAST_DIM + 1);
    int64_t gmOffset = static_cast<int64_t>(idx0) * inputStride0_
                     + static_cast<int64_t>(idx1) * inputStride1_;
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = updateDimSize_ * sizeof(T);
    DataCopyPad(inputGm_[gmOffset], updLocal[i * updateRowElements_], copyParams);
}
```

Benefit: 支持非连续张量的散射更新，无需预先 reshape
Trade-off: 逐行写出 MTE3 效率低，适合 update 行数较少的场景
