# P55 BlockReduceMax 替代 DataCopy slice (Block-level Reduction Optimization)
## Overview
当需要从 UB 中按固定间隔提取元素（如每 8 个元素取 1 个）并写回 GM 时，使用 BlockReduceMax 指令替代 DataCopy slice 操作，在 UB 内完成归约后再写回，提升搬运效率。

## When to Use
- DataCopy 源操作数和目的操作数 shape 不一致，相当于做 slice 操作
- 需要按 datablock（32 字节）粒度提取元素
- 提取后的数据需要进一步处理（如 Cast、Shift）

## Trade-off
- 需要额外的 UB 空间存放归约结果
- 仅支持 half/float 数据类型
- 需要精确计算 repeat、mask、stride 参数

**Source operators**: IFA nUpdate 输出

---

## Variant A: BlockReduceMax 替代 stride DataCopy
Source: 【案例总结】OBP IFA优化点汇总/V1的nUpdate UB2GM拷贝优化.md

原始方案使用 DataCopy stride 参数做 slice，效率低。优化方案使用 BlockReduceMax 在 UB 内完成归约，再 DataCopy 连续写回 GM。

**Expert implementation:**
```cpp
// 原始方案：DataCopy stride slice，效率低
DataCopyExtParams dataCopyParams;
dataCopyParams.blockCount = dealRowCount;  // 16
dataCopyParams.blockLen = 1 * sizeof(int32_t);  // 4
dataCopyParams.srcStride = (8 - 1) / (BYTE_BLOCK / sizeof(int32_t));  // stride=0 实际 32 字节
dataCopyParams.dstStride = 0;
DataCopyPad(nUpdateGm[...], nInt32Out, dataCopyParams);

// 优化方案：BlockReduceMax + 连续 DataCopy
LocalTensor<int32_t> nInt32Out = outputQue2.template AllocTensor<int32_t>();
int32_t repeatTime = (dealRowCount * BLOCK_ELEMENT_NUM) / FP28_MAX_MASK_ELEMENT_NUM;  // =2
int32_t srcRepStride = FP28_MAX_MASK_ELEMENT_NUM / BLOCK_ELEMENT_NUM;  // =8

// BlockReduceMax：每个 datablock 内取最大值，实现 slice 效果
BlockReduceMax(nUpdateTmp, nUpdateTmp, repeatTime, FP28_MAX_MASK_ELEMENT_NUM, 1, 1, srcRepStride);
pipe_barrier(PIPE_V);

// 后续处理
Cast(nInt32Out, nUpdateTmp, RoundMode::CAST_ROUND, dealRowCount);
pipe_barrier(PIPE_V);
ShiftLeft(nInt32Out, nInt32Out, (int32_t)23, dealRowCount);
pipe_barrier(PIPE_V);

// 连续 DataCopy，效率更高
DataCopy(nUpdateGm[...], nInt32Out, dealRowCount);
```

**vs. baseline (lingxi-code):**
```cpp
// 基线：DataCopy slice 效率低
DataCopyPad(nUpdateGm[...], nInt32Out, dataCopyParams);  // stride 模式
```

Benefit: BlockReduceMax 在 UB 内完成归约，后续 DataCopy 为连续搬运，效率显著提升
Trade-off: 需要额外的 pipe_barrier 和 Cast 操作；仅适用于按 datablock 粒度提取的场景