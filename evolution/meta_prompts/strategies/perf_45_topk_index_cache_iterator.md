# P45: 稀疏 Attention 离散块处理
## Overview
将 topK 选出的离散 block indices 预取到片上 cache 数组中，封装为 L1ChunkIterator 迭代器，按 L1 容量自动切分连续段，使上层调度逻辑与底层离散寻址解耦。完整的离散块处理流程包括：索引缓存与迭代（Variant A）、Gather-Merge 到连续 workspace（Variant B）、块内偏移追踪（Variant C）、PagedAttention 离散块搬运适配（Variant D）。

## When to Use
- 稀疏 Attention 算子中 KV 序列由 topK 离散 block 组成
- topK block 数量较多（≥8），需要按 L1 容量自动切分为多个 chunk 分批处理
- 稀疏 Attention MLA 场景，KV 数据在 GM 中按 block_table 离散分布，需要 gather-merge 到连续 workspace
- sparse block 有效长度不等于 sparseBlockSize（threshold 截断），需要精确追踪块内偏移
- PagedAttention 场景下 KV cache 按 block_table 映射到不连续的物理块

## Trade-off
- TOPK_CACHE_SIZE=128 限制可缓存数量；迭代器逻辑增加标量计算开销
- 每个 chunk 边界需要重新计算 GM 偏移，chunk 过小时标量开销占比增大
- Gather-Merge 需要额外 workspace GM 空间，V0 阶段增加核间同步开销
- 块内偏移追踪依赖运行时 threshold 值，标量循环开销在 sparse block 很小时显著
- PA 离散块搬运每个物理块边界需要独立 DataCopy 调用，无法跨块合并

**Source operators**: ai_infra_sparse_flash_attention_gqa

---

## Variant A: L1ChunkIterator 迭代器封装
Source: ai_infra_sparse_flash_attention_gqa

将离散 block indices 缓存到片上数组，封装为迭代器按 L1 容量自动切分连续段，上层只需调用 Next() 获取下一个搬运段。

**Expert implementation:**
```cpp
int32_t blockIndexCache_[TOPK_CACHE_SIZE] = {0};
__aicore__ inline bool Next() {
    while (blockIndexCursor_ < range_->GetSparseBlockCount()) {
        auto [blockStart, blockEnd] = range_->GetBlockRange(blockIndexCursor_, ...);
        if (currentL1Size_ + (chunkEnd - blockCursor_) >= range_->GetL1Size()) {
            l1ChunkJustCompleted_ = true;
        }
        currentSegment_ = {blockCursor_, chunkEnd, l1ChunkJustCompleted_};
        blockCursor_ = chunkEnd;
        return true;
    }
}
```

Benefit: 调度逻辑与离散寻址解耦，上层代码简洁且可复用
Trade-off: TOPK_CACHE_SIZE 限制可缓存数量；迭代器标量计算有额外开销

---

## Variant B: 离散 KV Gather-Merge 到连续 Workspace（V0 阶段）
Source: sparse_flash_attention_enhance

Vector 核按 topK 索引查表获取离散 KV 地址，每次合并 2 个 sparse block 为一条 DataCopyPad，写入连续 workspace GM。

**Expert implementation:**
```cpp
for (int64_t s2GmOffsetArray = s2GmStartOffset; s2GmOffsetArray < s2GmLimit;
     s2GmOffsetArray += 2 * constInfo.sparseBlockSize) {
    GetRealS2Idx(s2GmOffsetArray, s2IdxArray0, topkGmBaseOffset, runInfo);
    GetRealS2Idx(s2GmOffsetArray + constInfo.sparseBlockSize, s2IdxArray1, ...);
    CopyInKv(mte2Size, mte3Size, mergeMte3Idx, s2IdxArray0, s2IdxArray1, runInfo);
    if (mte2Size - mte3Size + 2 * constInfo.sparseBlockSize > 32) {
        CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, runInfo);
    }
}
v0ValidSizeUb_.SetValue(runInfo.loop % MERGE_CACHE_GM_BUF_NUM, mte2Size);
```

Benefit: 离散 KV 合并为连续 workspace，后续 Cube 搬运效率最大化
Trade-off: 额外 workspace GM 空间；V0 阶段增加核间同步开销

---

## Variant C: TopK 块内偏移追踪的分段搬运
Source: sparse_flash_attention_enhance

精确维护 curTopKIdx 和 curOffsetInSparseBlock，按 threshold 截断有效长度，跨 block 边界时自动切换到下一个 topK block。

**Expert implementation:**
```cpp
__aicore__ inline void CalcTopKBlockInfo(const RunInfo &info,
    uint32_t &curTopKIdx, uint64_t &curOffsetInSparseBlock,
    uint32_t curSeqIdx, uint32_t &copyRowCnt, int64_t &idInTopK) {
    uint64_t blockEnd = (blockBegin + constInfo.sparseBlockSize > info.threshold)
        ? info.threshold : blockBegin + constInfo.sparseBlockSize;
    uint64_t blockLen = blockEnd - blockBegin;
    if (curOffsetInSparseBlock + copyRowCnt < blockLen) {
        curOffsetInSparseBlock += copyRowCnt;
    } else {
        curTopKIdx++; curOffsetInSparseBlock = 0;
        copyRowCnt = blockLen;
    }
}
```

Benefit: 精确填满 L1 切分，无浪费搬运；自动处理 threshold 截断
Trade-off: 标量循环逐块查找，sparse block 很小时标量开销显著

---

## Variant D: PA（PagedAttention）离散块搬运适配
Source: sparse_flash_attention_enhance, ai_infra_sparse_flash_attention_gqa, common/CopyInL1.h

封装 block_table 查表和分段搬运逻辑，按物理块边界自动切分，循环内逐段 DataCopy 直到搬运完成。

**Expert implementation:**
```cpp
template <typename T, SFA_LAYOUT SRC_LAYOUT>
__aicore__ inline void DataCopyPA(LocalTensor<T> &dstTensor,
    GlobalTensor<T> &srcTensor, GlobalTensor<int32_t> &blockTableGm,
    const PAShape &shape, const Position &startPos) {
    while (copyFinishRowCnt < shape.copyRowNum) {
        uint64_t blockIdOffset = curS2Idx / shape.blockSize;
        uint64_t idInBlockTable = blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset);
        uint32_t copyRowCnt = shape.blockSize - remainRowCnt;
        uint64_t offset = idInBlockTable * shape.blockSize * shape.headNum * shape.headDim;
        DataCopyGmNDToL1<T>(tmpDstTensor, tmpSrcTensor, copyRowCnt, ...);
        copyFinishRowCnt += copyRowCnt;
    }
}
```

Benefit: 上层无需关心 PA 离散寻址细节，统一接口适配连续和离散场景
Trade-off: 每个物理块边界一次独立 DataCopy，无法跨块合并；查表是标量操作
