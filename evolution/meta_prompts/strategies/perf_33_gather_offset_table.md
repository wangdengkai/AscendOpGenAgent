# P33: Gather 偏移表驱动的结构化数据提取

## Overview
通过预计算字节偏移表，使用 Gather 指令一次性从混合矩阵中提取多个子张量，偏移表只计算一次常驻 UB。

## When to Use
- matmul 结果为混合矩阵需要按固定模式提取多个子张量（如 h_pre/h_post/h_res）
- 提取模式在编译期已知且跨迭代不变，偏移表可一次预计算常驻 UB

## Trade-off
- 偏移表占用 UB 空间（元素数 × 4 字节），子张量数量多时占用显著
- offset 必须是字节偏移且 Gather 要求源数据在 UB 中连续

**Source operators**: ai_infra_manifold_constrained_hyper_connection_pre

---

## Variant A: 预计算偏移表 + Gather 提取子张量
Source: ai_infra_manifold_constrained_hyper_connection_pre

在 Init 阶段预计算字节偏移表并常驻 UB，后续通过 Gather 指令按偏移表一次性提取目标子张量。

```cpp
// 预计算偏移表（Init 阶段，只执行一次）
for (uint32_t i = 0; i < V1_BASE_T; i++) {
    for (uint32_t j = 0; j < N_; j++)
        preOffsetBuf_.SetValue(offset1++, curOffset * sizeof(P));
    curOffset += N_;
    for (uint32_t j = 0; j < N_; j++)
        postOffsetBuf_.SetValue(offset2++, curOffset * sizeof(P));
    curOffset += nSquare;
}
Gather(hPreBuff_, matmulRes_, preOffsetBuf_, 0, lenT * N_);
Gather(hPostBuff_, matmulRes_, postOffsetBuf_, 0, lenT * N_);
```

Benefit: 偏移表只计算一次，后续 Gather 提取无需逐元素寻址
Trade-off: 偏移表占用 UB 空间，offset 必须是字节偏移且对齐
