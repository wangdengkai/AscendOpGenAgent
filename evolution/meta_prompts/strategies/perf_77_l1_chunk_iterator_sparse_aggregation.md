# P77 L1 Chunk Iterator 稀疏块自适应聚合 (L1 Chunk Iterator Sparse Block Adaptive Aggregation)
## Overview
在 Sparse Attention 推理场景中，多个不连续的 sparse block 可能无法单独填满一个 L1 buffer。L1ChunkIterator 将多个 sparse block 动态聚合为一个 L1 chunk，使每个 chunk 恰好填满一个 L1 KP buffer（64KB），最大化 L1 利用率。支持 FastAdvance 快速跳过 causal mask 裁剪后的无效区域。

## When to Use
- Sparse Attention 推理，sparseBlockSize 较小（4/8/16）导致单个 sparse block 无法填满 L1
- S2 方向有大量 sparse block 需要迭代处理
- 需要结合 causal mask 跳过无效区域

## Trade-off
- 迭代器状态管理增加标量计算开销
- 不连续 sparse block 的 DMA 搬运可能产生碎片
- sparseBlockSize 较大时（>= L1 buffer 容量）退化为单块迭代

**Source operators**: ai_infra_sparse_flash_attention_gqa

---
## Variant A: Advance 动态聚合（推理 GQA）
Source: ai_infra_sparse_flash_attention_gqa op_kernel

GQA 算子的 `L1ChunkIterator::Advance()` 在每次迭代时累加 sparse block 大小，直到超过 L1 buffer 容量才切换到下一个 chunk。

**Expert implementation:**
```cpp
// L1ChunkIterator: 动态聚合多个 sparse block 到一个 L1 chunk
class L1ChunkIterator {
    int64_t Advance() {
        chunkStartIdx_ = chunkEndIdx_;
        int64_t accumulated = 0;
        while (chunkEndIdx_ < sparseBlockCount_) {
            int64_t blockSize = GetBlockSize(chunkEndIdx_);
            if (accumulated + blockSize > l1BufferCapacity_) break;
            accumulated += blockSize;
            chunkEndIdx_++;
        }
        return accumulated;  // 当前 chunk 的总大小
    }
};
```

**vs. baseline (lingxi-code):**
```cpp
// 基线：每个 sparse block 独立搬运到 L1
for (int i = 0; i < sparseBlockCount; i++) {
    LoadData(l1Tensor, kvGm[sparseIdx[i] * headDim], params);
    // 每个 block 一次 L1 搬运，L1 利用率低
}
```

Benefit: L1 利用率最大化；自动适应不同 sparse 配置
Trade-off: 迭代器状态管理增加标量计算开销

---
## Variant B: FastAdvance 快速跳过无效区域
Source: ai_infra_sparse_flash_attention_gqa op_kernel

`FastAdvance()` 支持跳过不需要计算的 segment（如 causal mask 裁剪后的无效区域），直接将迭代器推进到下一个有效位置。

**Expert implementation:**
```cpp
// FastAdvance: 跳过无效 segment
int64_t FastAdvance(int64_t skipToIdx) {
    // 直接跳过 [chunkEndIdx_, skipToIdx) 范围内的所有 block
    chunkStartIdx_ = skipToIdx;
    chunkEndIdx_ = skipToIdx;
    // 从 skipToIdx 开始正常聚合
    return Advance();
}
```

Benefit: 结合 causal mask 裁剪，跳过无效 S2 区域，减少无效搬运
Trade-off: 跳过粒度受 sparse block 边界限制
