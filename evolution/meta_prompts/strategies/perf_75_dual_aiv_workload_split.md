# P75 双 AIV M/S1 轴工作量分裂 (Dual AIV M/S1-Axis Workload Split)
## Overview
利用 Ascend 910B 的 1:2 AIC:AIV 核比例，将 Vector 阶段的 M 轴或 S1 轴工作量在两个 AIV 核之间分裂。每个 AIC 核的 Cube 计算结果被 2 个 AIV 核并行处理，实现 Cube:Vector = 1:2 的计算比例匹配，充分利用双 AIV 核的 Vector 算力。

## When to Use
- KERNEL_TYPE_MIX_AIC_1_2 模式下 1 AIC : 2 AIV 架构
- Vector 阶段成为性能瓶颈（Vector Bound）
- M 轴或 S1 轴足够大（> 16）可以有效切分

## Trade-off
- M/S1 <= 16 时无法分裂，退化为单 AIV
- 分裂边界需要 16 对齐，可能有少量负载不均
- 两个 AIV 核需要独立的 workspace 区域，增加内存开销

**Source operators**: ai_infra_sparse_flash_attention_pioneer, flash_attention_score_enhance, sparse_lightning_indexer_grad_kl_loss_enhance

---
## Variant A: M 轴近似均分（推理 MLA）
Source: ai_infra_sparse_flash_attention_pioneer op_kernel

Pioneer MLA 算子将 M 轴按 16 对齐后近似均分为两半，AIV0 处理前半部分，AIV1 处理后半部分。

**Expert implementation:**
```cpp
// M 轴分裂公式
info.mSizeV = (info.mSize <= 16) ? info.mSize :
    (((info.mSize + 15) / 16 + 1) / 2 * 16);

// AIV 核区分
uint32_t aivIdx = GetBlockIdx() % 2;
if (aivIdx == 0) {
    vecStartM = 0;
    vecDealM = mSizeV;
} else {
    vecStartM = mSizeV;
    vecDealM = mSize - mSizeV;
}
```

**vs. baseline (lingxi-code):**
```cpp
// 基线：单 AIV 处理全部 M 轴
vecStartM = 0;
vecDealM = mSize;  // 单核处理，Vector 成为瓶颈
```

Benefit: Vector 阶段吞吐量翻倍；两个 AIV 核并行处理不同 M 行，无数据依赖
Trade-off: M <= 16 时无法分裂；分裂边界需要 16 对齐

---
## Variant B: S1 轴不等分切分（训练 FAE）
Source: flash_attention_score_enhance op_kernel

FAE-SAB 变体中 Cube 核处理 cubeS1BaseSize 行，每个 Vector 核处理其中一半。

**Expert implementation:**
```cpp
// S1 轴切分
uint32_t cubeS1BaseSize = tilingData->coreParams.s1BaseSize;
uint32_t vecS1BaseSize = cubeS1BaseSize / 2;

// AIV 子块区分
uint32_t cubeSubIdx = vecBlockIdx % 2;
uint32_t s1RealSize = (cubeS1RealSize + 1) / 2;
uint32_t vecCoreOffset = cubeSubIdx * s1RealSize;
// 第二个 AIV 核处理剩余部分
```

Benefit: 充分利用 2 个 AIV 核的 Vector 算力；避免单个 AIV 核成为瓶颈
Trade-off: S1 轴必须足够大才能有效切分；奇数行时两个 AIV 核负载不完全均衡

---
## Variant C: P/SY 双 AIV 异构任务分裂
Source: sparse_lightning_indexer_grad_kl_loss_enhance op_kernel

两个 AIV 子块分别执行不同的 Vector 计算任务（VectorP 和 VectorSy），而非处理同一任务的不同数据。

**Expert implementation:**
```cpp
// 异构任务分裂
uint32_t subBlockIdx = aivIdx % 2;
if (runInfo.calcP) {
    VectorP(runInfo);   // AIV0: softmax 计算
} else {
    VectorSy(runInfo);  // AIV1: KL-divergence loss
}
```

Benefit: 两个独立 Vector 计算任务并行执行，吞吐翻倍
Trade-off: 两个任务的计算量需要大致均衡；任务间需要通过 workspace 同步
