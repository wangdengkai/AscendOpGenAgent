# P76 V_TEMPLATE KV 预合并 Workspace (V_TEMPLATE KV Pre-Merge Workspace)
## Overview
在 Sparse Attention 的 V_TEMPLATE 模式下，由 Vector 核在 Cube 计算之前将分散在 PageAttention block table 中的稀疏 KV 数据预合并到连续的 GM workspace 中。Cube 核直接从连续 workspace 读取，避免处理不连续 PA 地址的复杂性。通过 sparseBlockSize 阈值（<=4）自动选择 V_TEMPLATE 或 C_TEMPLATE 路径。

## When to Use
- Sparse Attention + PageAttention 推理场景
- sparseBlockSize 较小（<=4），Cube 核随机访存效率低
- MLA 场景下 KV 数据分散在多个 PA block 中

## Trade-off
- 需要额外的 4*512*576*sizeof(half) ≈ 2.25MB workspace 空间
- Vector0 阶段增加了额外的搬运开销和 V0-C1 同步
- sparseBlockSize 较大时 C_TEMPLATE 更优（无需预合并）

**Source operators**: ai_infra_sparse_flash_attention_pioneer, sparse_flash_attention_enhance, sparse_lightning_indexer_grad_kl_loss_enhance

---
## Variant A: MergeKv 4-GM-Buffer 轮转预合并（推理）
Source: ai_infra_sparse_flash_attention_pioneer op_kernel

Pioneer MLA 使用 4 个 GM buffer 轮转（MERGE_CACHE_GM_BUF_NUM=4），每个 buffer 大小为 512*576*sizeof(half)。

**Expert implementation:**
```cpp
// V_TEMPLATE: Vector0 阶段 MergeKv
void MergeKv(const ExtraInfo& info) {
    for (int i = 0; i < sparseBlockCount; i++) {
        int realS2Idx = GetRealS2Idx(topKIndices, i);
        // 从 PA block table 中 gather KV 数据
        DataCopyPad(ubBuf, kvGm[realS2Idx * headDim], copyParams);
        // ND→NZ 格式转换
        TransDataTo5HD(nzBuf, ubBuf, transParams);
        // 写入连续 workspace
        DataCopy(kvMergeGm[mergeOffset], nzBuf, outParams);
        mergeOffset += blockSize * headDim;
    }
}
// Cube 阶段直接从 kvMergeGm 读取连续数据
LoadData(l1Tensor, kvMergeGm[offset], loadParams);
```

**vs. baseline (lingxi-code):**
```cpp
// C_TEMPLATE: Cube 核直接从 GM 随机读取
for (int i = 0; i < sparseBlockCount; i++) {
    int realS2Idx = CalcTopKBlockInfo(topKIndices, i);
    LoadData(l1Tensor, kvGm[realS2Idx * headDim], loadParams);
    // 每个 sparse block 一次随机 DMA
}
```

Benefit: Cube 核只需顺序读取连续数据；ND→NZ 转换在 Vector 阶段完成，与 Cube 计算重叠
Trade-off: 需要额外 workspace 空间；V0 阶段增加搬运开销

---
## Variant B: 双 SubBlock 并行 Gather（训练）
Source: sparse_flash_attention_enhance op_kernel

训练算子的 MergeKv 使用双 SubBlock 并行：两个 AIV 子块分别处理 S2 的前半和后半。

**Expert implementation:**
```cpp
// 双 SubBlock 并行 gather
uint32_t subBlockIdx = GetSubBlockIdx();
int64_t s2GmStartOffset = (subBlockIdx == 0) ? 0 :
    CeilDiv(s2Pair, 2L) * 2 * sparseBlockSize;
int64_t s2GmLimit = (subBlockIdx == 0) ?
    CeilDiv(s2Pair, 2L) * 2 * sparseBlockSize : s2Size;

// 两块合并搬运：一条 DataCopyPad 搬运两个相邻 sparse block
DataCopyPadParams params;
params.blockCount = (keyOffset1 >= 0) + (keyOffset2 >= 0);
params.srcStride = (keyOffset2 - keyOffset1) / headDim - 1;
DataCopyPad(ubBuf, kvGm[keyOffset1], params);
```

Benefit: 双 SubBlock 并行将 gather 吞吐翻倍；两块合并搬运减少 DMA 指令数
Trade-off: stride 溢出时回退到单块搬运；需要运行时检查 stride 有效性
