# P82 BufferPolicy Class for Multi-Stage Pipeline (手写 BufferPolicy 类替代 TQue 的多阶段流水策略)
## Overview
当标准 `TQue` 或单一双缓冲无法表达复杂的 buffer 访问路径时，可以手写 `BufferPolicy` 类，通过显式的 buffer 集合、轮转指针和独立的 alloc/use/free 入口管理片上 buffer 生命周期。该策略的边界是“用策略类替代队列流控”：根据流水阶段数与分块维度，选择 2-buffer PingPong、3-buffer 三级轮转或 2×2 matrix buffer，将当前轮、上一轮复用、Cube/Vector 分阶段消费和多维分块释放等访问关系显式编码。

## When to Use
- 标准 `TQue` 或普通双缓冲无法表达 Q/KV 复用、三级流水或多维分块的 buffer 访问模式
- 算子需要让 DMA、Cube、Vector 或多维度 Matmul 分块通过不同访问入口同时持有不同 buffer，且单一轮转顺序会造成流水气泡
- 算子能够接受额外的片上存储占用，以换取更高的流水并行度和硬件利用率
- 开发者可以明确设计 alloc/use/free、Get/GetPre/GetReused 或 Cube/Vector 访问协议，并承担手写 buffer 策略的调试成本
- 不适用于 `GM→L1→L0→Mmad` 存储层级级联的手动 Mmad 流水；该类场景应优先参考 P81

## Trade-off
- 需要显式管理 buffer 生命周期、轮转指针和同步事件，失去 `TQue` 自动流控保护
- buffer 数量通常增加到 2 倍、3 倍或 4 份，显著压缩片上存储预算
- 多套访问路径（如 alloc/use/free、Cube/Vector/DMA）和多级事件依赖容易引入静默数据错误，调试复杂度高
- 具体策略与流水拓扑绑定较强，需要根据实际阶段数和分块维度定制选择

**Source operators**: ai_infra_sparse_flash_attention_gqa, ai_infra_fused_infer_attention_sink

---

## Variant A: 手写 PingPong 双缓冲策略类
Source: ai_infra_sparse_flash_attention_gqa, ai_infra_fused_infer_attention_sink

当核心需求是“标准轮转 + 上一轮复用 + 独立复用轮转”并存时，可手写双缓冲策略类，用两个 buffer 和两套 flag 分别承载正常消费与复用消费路径。这样既能支持当前迭代正常取数，也能支持 Q 矩阵跨迭代复用或 KV buffer 交替复用。

**Expert implementation:**
```cpp
template<BufferType bufferType, SyncType syncType>
class BuffersPolicyDB {
public:
    __aicore__ inline void Init(BufferManager<bufferType> &bufferManager, uint32_t size){
        ping_ = bufferManager.template AllocBuffer<syncType>(size);
        pong_ = bufferManager.template AllocBuffer<syncType>(size);
        ping_.Init(); pong_.Init();
    }
    __aicore__ inline Buffer<bufferType, syncType> &Get() {
        if (flag1_) { flag1_ = 0; return ping_; } else { flag1_ = 1; return pong_; }
    }
    __aicore__ inline Buffer<bufferType, syncType> &GetPre() {
        if (flag1_) { return pong_; } else { return ping_; }
    }
    __aicore__ inline Buffer<bufferType, syncType> &GetReused() {
        if (flag2_ == 0) { flag2_ = 1; return pong_; } else { flag2_ = 0; return ping_; }
    }
};
```

**vs. baseline (lingxi-code):**
```cpp
TQue<QuePosition::VECIN, 2> inQueue;
LocalTensor<float> cur = inQueue.DeQue<float>();
Compute(cur);
// 无法表达上一轮 buffer 复用或独立复用轮转，只能按单一路径顺序消费
```

Benefit: 支持当前轮、上一轮和复用轮转三种访问模式，让搬运与 Cube 计算更充分重叠。
Trade-off: 需要自行维护 flag 与同步事件，失去 `TQue` 自动流控保护。

---

## Variant B: 三缓冲三级轮转策略类
Source: ai_infra_sparse_flash_attention_gqa, ai_infra_fused_infer_attention_sink

当算子需要让搬运、Cube、Vector 三个阶段同时持有不同 buffer 并独立推进时，可采用 3-buffer 轮转。每个阶段维护自己的轮转指针，分别访问 a/b/c 三个 buffer，实现三级流水互不阻塞。

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
};
```

**vs. baseline (lingxi-code):**
```cpp
TQue<QuePosition::VECIN, 2> inQueue;
for (int loop = 0; loop < loops; ++loop) {
    CopyIn(loop);
    CubeCompute(loop);
    VecCompute(loop);
}
// 同一套双缓冲被多个阶段复用，三级阶段难以同时持有独立 buffer
```

Benefit: 搬运、Cube、Vector 可并行占用不同 buffer，三级流水互不阻塞。
Trade-off: 片上存储占用提升到 3 份，三套独立轮转指针与同步关系更难验证。

---

## Variant C: 2×2 矩阵缓冲策略类
Source: ai_infra_sparse_flash_attention_gqa, ai_infra_fused_infer_attention_sink

当 Matmul 需要同时沿 M 和 K 两个维度推进流水时，单维度 pingpong 不够用，可以把 4 个 buffer 组织成 2×2 矩阵。通过列优先遍历与独立的 `AllocNext/ReuseNext/FreeNext` 索引，分别跟踪分配、使用和释放进度，使双维度分块能够持续重叠。

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
        if (mIdx == 0 && kIdx == 0) { return bufferM0k0_; }
        if (mIdx == 0 && kIdx == 1) { return bufferM0k1_; }
        if (mIdx == 1 && kIdx == 0) { return bufferM1k0_; }
        return bufferM1k1_;
    }

    __aicore__ inline Buffer<bufferType, syncType> &AllocNext() {
        aIdx_++;
        return GetBuffer(aIdx_, amIdx_, akIdx_);
    }
    __aicore__ inline Buffer<bufferType, syncType> &ReuseNext() {
        uIdx_++;
        return GetBuffer(uIdx_, umIdx_, ukIdx_);
    }
    __aicore__ inline Buffer<bufferType, syncType> &FreeNext() {
        fIdx_++;
        return GetBuffer(fIdx_, fmIdx_, fkIdx_);
    }
};
```

**vs. baseline (lingxi-code):**
```cpp
for (int mTile = 0; mTile < mLoops; ++mTile) {
    for (int kTile = 0; kTile < kLoops; ++kTile) {
        LoadA(mTile, kTile);
        LoadB(kTile);
        Mmad(cTensor, aTensor, bTensor, ...);
    }
}
// 仅按单一路径推进，无法同时维护 M/K 两个维度的独立 buffer 生命周期
```

Benefit: 支持 M/K 双维度同时流水，适合复杂 Matmul 分块场景。
Trade-off: 4 个 buffer 与三套索引管理带来显著的片上存储和心智负担。
