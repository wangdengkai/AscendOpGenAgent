# P20: 三缓冲轮转（BuffersPolicy3buff）
## Overview
在 AIC/AIV 混合核架构下，三缓冲允许搬运、Cube 计算、Vector 计算三个阶段同时进行，形成三级流水。每个阶段持有独立的 buffer 和轮转指针，互不干扰。

## When to Use
- AIC/AIV 混合核架构下的 Cube+Vector 融合算子
- 需要搬运、Cube、Vector 三级流水同时重叠的场景

## Trade-off
- 片上存储占用增加到 3 倍；三套独立轮转指针调试难度大
- 同步事件数量增多，需要为每个阶段独立管理 event

**Source operators**: ai_infra_sparse_flash_attention_gqa, ai_infra_fused_infer_attention_sink

---

## Variant A: 三缓冲独立轮转策略类
Source: ai_infra_sparse_flash_attention_gqa, ai_infra_fused_infer_attention_sink

分配 a/b/c 三个 buffer，搬运、Cube、Vector 各持有独立的 flag 轮转指针（flag1_、flag1_vec1_、flag1_bmm2_），通过 Get/GetVec/GetCube 分别访问，三级流水互不阻塞。

**Expert implementation:**
```cpp
template<BufferType bufferType, SyncType syncType>
class BuffersPolicy3buff {
    Buffer<bufferType, syncType> a_, b_, c_;
    uint32_t flag1_ = 0, flag1_vec1_ = 0, flag1_bmm2_ = 0;
public:
    __aicore__ inline Buffer<bufferType, syncType> &Get() {
        if (flag1_ == 0) { flag1_ = 1; return a_; }
        else if (flag1_ == 1) { flag1_ = 2; return b_; }
        else { flag1_ = 0; return c_; }
    }
    __aicore__ inline Buffer<bufferType, syncType> &GetVec() {
        if (flag1_vec1_ == 0) { flag1_vec1_ = 1; return a_; }
        else if (flag1_vec1_ == 1) { flag1_vec1_ = 2; return b_; }
        else { flag1_vec1_ = 0; return c_; }
    }
    __aicore__ inline Buffer<bufferType, syncType> &GetCube() {
        if (flag1_bmm2_ == 0) { flag1_bmm2_ = 1; return a_; }
        else if (flag1_bmm2_ == 1) { flag1_bmm2_ = 2; return b_; }
        else { flag1_bmm2_ = 0; return c_; }
    }
    // 三级流水使用示例：搬运(MTE2) / Cube(M) / Vector(V) 同时进行
    // 每级持有独立 buffer，通过 HardEvent 同步数据就绪
    //
    // WaitFlag<HardEvent::MTE2_MTE1>(eventId);  // 等待搬运完成
    // LoadData(Get());                            // Cube 使用搬运完成的 buffer
    // SetFlag<HardEvent::MTE1_M>(eventId);       // 通知 Cube 数据就绪
    //
    // WaitFlag<HardEvent::MTE1_M>(eventId);      // 等待 Cube 完成
    // Compute(GetCube());                         // Vector 使用 Cube 完成的 buffer
    // SetFlag<HardEvent::M_FIX>(eventId);        // 通知 Vector 数据就绪
};
```

Benefit: 搬运/Cube/Vector 三级流水完全重叠，理论吞吐接近硬件峰值
Trade-off: 片上存储占用 3 倍；三套独立 flag 管理复杂，调试困难
