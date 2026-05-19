# P59 输出 Transpose 融合 (Output Transpose Fusion)
## Overview
在 IFA 等算子输出阶段，通过 DataCopy stride 参数实现输出格式转换（Transpose），避免额外的转置算子调用。支持 BNSD→NBSD、BSND→NBSD、BSH→NBSD、TND→NTD 等多种布局转换。

## When to Use
- 下游算子要求不同的张量布局（如 Batch Matmul 需要 N 在最外层）
- 输出 UB→GM 阶段可利用 stride 参数实现跳写
- 避免额外的 Transpose 算子开销

## Trade-off
- DataCopy stride 模式效率低于连续搬运
- 需要精确计算各轴偏移量
- 头块和尾块需要单独处理

**Source operators**: IFA 输出布局转换

---

## Variant A: BNSD→NBSD 转置输出
Source: 【案例总结】OBP IFA优化点汇总/IFA支持输出transpose.md

利用 DataCopy 的 dstStride 参数实现 N 轴提取到最外层，连续读、跳写模式。

**Expert implementation:**
```cpp
// BNSD → NBSD 转置
// attenOutUb: gCount * S1 * D（连续）
// 输出: N * B * S * D（N 轴在外层）

DataCopyParams dataCopyParams;
dataCopyParams.blockCount = gCount;  // 处理多少个 G
dataCopyParams.blockLen = s1Size * headDim * sizeof(OUT_T) / 32U;  // 一个 S1*D
dataCopyParams.srcStride = 0;  // 连读
dataCopyParams.dstStride = (batchSize * qSeqSize - s1Size) * headDim * sizeof(OUT_T) / 32U;  // 跳写

uint64_t attenOutOffset = n2Idx * gSize * batchSize * qSeqSize * headDim +  // N2轴
                          gStartIdx * batchSize * qSeqSize * headDim +       // G轴
                          bIdx * qSeqSize * headDim;                         // B轴
DataCopy(attentionOutGm[attenOutOffset], attenOutUb, dataCopyParams);
```

**vs. baseline (lingxi-code):**
```cpp
// 基线：输出后调用 Transpose 算子
DataCopy(outputGm, attenOutUb, size);  // BNSD 格式
Transpose(outputGm, tempGm, ...);      // 额外转置操作
```

Benefit: 消除 Transpose 算子调用，减少 kernel 启动开销和数据搬运
Trade-off: stride 模式效率略低；需要处理头块和尾块边界

---

## Variant B: 头尾块单独处理
Source: 【案例总结】OBP IFA优化点汇总/IFA支持输出transpose.md

当 S1G 合轴切分导致 G 边界不对齐时，头块和尾块需要单独 DataCopy，中间块使用 stride 跳写。

**Expert implementation:**
```cpp
// 判断头尾块
bool hasHeadBlock = (s1StartIdx != 0);
bool hasTailBlock = ((s1EndIdx + 1) != s1Size);

if (hasHeadBlock) {
    // 头块：单独一条 DataCopy
    DataCopyParams dataCopyParamsHead;
    dataCopyParamsHead.blockCount = 1;
    dataCopyParamsHead.blockLen = (s1Size - s1StartIdx) * headDim * sizeof(OUT_T) / 32U;
    dataCopyParamsHead.dstStride = 0;  // 单块无需跳写
    DataCopy(attentionOutGm[offset], attenOutUb, dataCopyParamsHead);
    attenOutUbOffset += (s1Size - s1StartIdx) * headDim;
}

// 中间块：stride 跳写
DataCopyParams dataCopyParams;
dataCopyParams.blockCount = gCount - hasHeadBlock - hasTailBlock;
dataCopyParams.dstStride = (batchSize * qSeqSize - s1Size) * headDim / 32U;
DataCopy(attentionOutGm[offset], attenOutUb[attenOutUbOffset], dataCopyParams);

if (hasTailBlock) {
    // 尾块：单独一条 DataCopy
    DataCopyParams dataCopyParamsTail;
    dataCopyParamsTail.blockCount = 1;
    dataCopyParamsTail.blockLen = (s1EndIdx + 1) * headDim * sizeof(OUT_T) / 32U;
    DataCopy(attentionOutGm[offset], attenOutUb[attenOutUbOffset], dataCopyParamsTail);
}
```

Benefit: 正确处理不对齐边界，保证输出正确性
Trade-off: 增加 DataCopy 指令数量；逻辑复杂度增加

---

## Variant C: TND→NTD 变长序列转置
Source: 【案例总结】OBP IFA优化点汇总/IFA支持输出transpose.md

TND 场景下每个 batch 的 T 大小不同，需要动态计算 tSize 和 tBase。

**Expert implementation:**
```cpp
// TND → NTD 转置
uint64_t tSize = actualSeqLengthsGmQ.GetValue(batchSize - 1);  // 总 T
uint64_t tBase = (bIdx == 0) ? 0 : actualSeqLengthsGmQ.GetValue(bIdx - 1);  // 当前 batch 的 T 起始

DataCopyParams dataCopyParams;
dataCopyParams.blockCount = gCountOneS1;
dataCopyParams.blockLen = headDim * sizeof(OUT_T) / 32U;
dataCopyParams.dstStride = (tSize - 1) * headDim * sizeof(OUT_T) / 32U;  // 按 T 跳写

uint64_t attenOutOffset = n2Idx * gSize * tSize * headDim +  // N2轴
                          gIdx * tSize * headDim +            // G轴
                          tBase * headDim +                   // B轴（动态）
                          s1Idx * headDim;                    // S1轴
DataCopy(attentionOutGm[attenOutOffset], attenOutUb[attenOutUbOffset], dataCopyParams);
```

Benefit: 支持变长序列的转置输出，适配 TND 场景
Trade-off: 需要从 GM 读取 actual sequence 信息