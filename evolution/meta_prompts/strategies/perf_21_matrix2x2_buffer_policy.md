# P21: 二维矩阵缓冲策略（Matrix2x2BufferPolicy）
## Overview
针对 Matmul 切 M 和切 K 两个维度同时需要流水的场景，用 2x2=4 个 buffer 组成二维矩阵，按列优先分配、使用和释放。适用于 MLA headDim=576 的 nope/rope 分离等需要双维度流水的 Cube 算子。

## When to Use
- Matmul 需要同时在 M 和 K 两个维度做流水的场景（如 MLA headDim=576 的 nope/rope 分离）
- 单维度 PingPong 无法覆盖双维度切分的流水需求

## Trade-off
- 4 个 buffer 占用大量片上存储；Alloc/Reuse/Free 三套独立索引管理复杂
- 列优先遍历顺序与常规行优先直觉不同，容易出错

**Source operators**: ai_infra_sparse_flash_attention_gqa, ai_infra_fused_infer_attention_sink

---

## Variant A: 2x2 矩阵缓冲策略类
Source: ai_infra_sparse_flash_attention_gqa, ai_infra_fused_infer_attention_sink

4 个 buffer 按 (M0K0, M0K1, M1K0, M1K1) 排列，通过 mExtent_ 控制列优先遍历。AllocNext/ReuseNext/FreeNext 三套独立索引分别管理分配、使用和释放的进度。

**Expert implementation:**
```cpp
template<BufferType bufferType, SyncType syncType>
class Matrix2x2BufferPolicy {
    Buffer<bufferType, syncType> bufferM0k0_, bufferM0k1_, bufferM1k0_, bufferM1k1_;
    int32_t mExtent_ = 0;
    int32_t aIdx_ = -1, uIdx_ = -1, fIdx_ = -1;
    int32_t amIdx_ = 0, akIdx_ = 0, umIdx_ = 0, ukIdx_ = 0, fmIdx_ = 0, fkIdx_ = 0;
    static constexpr int32_t kSize_ = 2;

    __aicore__ inline Buffer<bufferType, syncType> &GetBuffer(
        int32_t xIdx, int32_t &mIdx, int32_t &kIdx) {
        mIdx = (mIdx + mExtent_ - 1) % mExtent_;
        kIdx = (xIdx / mExtent_) % kSize_;
        return buffers_[mIdx * kSize_ + kIdx];
    }
    __aicore__ inline Buffer<bufferType, syncType> &AllocNext() {
        aIdx_++; return GetBuffer(aIdx_, amIdx_, akIdx_);
    }
    __aicore__ inline Buffer<bufferType, syncType> &ReuseNext() {
        uIdx_++; return GetBuffer(uIdx_, umIdx_, ukIdx_);
    }
    __aicore__ inline Buffer<bufferType, syncType> &FreeNext() {
        fIdx_++; return GetBuffer(fIdx_, fmIdx_, fkIdx_);
    }
};
```

Benefit: 支持 M/K 双维度同时流水，最大化 Cube 单元利用率
Trade-off: 4 个 buffer 占用大量片上存储；三套索引管理复杂度高
