# P19: 自定义多级 PingPong 双缓冲
## Overview
绕过 TQue 标准队列机制，用手写 ping/pong 轮转实现 L0A/L0B/L1 等多级存储的双缓冲。支持 Q 复用（GetPre）和 KV 复用（GetReused），使搬运和计算完全重叠。对于 Cube 密集型算子，还可在 L1A/L1B、L0A/L0B、L0C 三级存储分别做 PingPong 双缓冲，使 GM→L1→L0→Mmad 三级流水线完全重叠（见 Variant B）。

## When to Use
- FlashAttention 等多级流水、多生产者-消费者关系的 Cube 算子
- 标准 TQue 无法表达复杂的 buffer 复用模式（如 Q 矩阵跨迭代复用、KV 交替消费）
- Cube 密集型算子需要最大化 Matmul 吞吐，GM→L1→L0→Mmad 链路成为性能瓶颈

## Trade-off
- 需要手动管理同步事件；buffer 数量翻倍，片上存储占用增加一倍
- 失去 TQue 的自动流控保护，同步错误会导致静默数据错误
- Cube 三级 PingPong 需要管理 6 组以上 event ID 数组，编程复杂度极高

**Source operators**: ai_infra_sparse_flash_attention_gqa, ai_infra_fused_infer_attention_sink

---

## Variant A: 手写 PingPong 双缓冲策略类
Source: ai_infra_sparse_flash_attention_gqa, ai_infra_fused_infer_attention_sink

通过 flag 位手动轮转 ping/pong buffer，提供 Get（标准轮转）、GetPre（获取上一轮 buffer 实现 Q 复用）、GetReused（独立 flag 的 KV 复用轮转）三种访问模式。

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

Benefit: 支持多种复用模式（Q 复用、KV 复用），使搬运与 Cube 计算完全重叠
Trade-off: 需要手动管理同步事件，失去 TQue 自动流控保护；buffer 数量翻倍

---

## Variant B: Cube 侧 L1/L0 三级 PingPong 双缓冲
Source: ai_infra_chunk_gated_delta_rule_recurrence, lightning_indexer_enhance

L1A/L1B 各分配 2 份 buffer，L0A/L0B 各分配 2 份 buffer，通过 `curIter & 1` 计算 offset 实现 PingPong 切换。每级之间用独立的 event ID 数组（mte1ToMte2A、mte2ToMte1A、mToMte1A、mte1ToMA）控制同步。

**Expert implementation:**
```cpp
static constexpr uint32_t L1A_PP_SIZE = 128 * 1024;
static constexpr uint32_t L0A_PP_SIZE = 32 * 1024;

// L1 PingPong
size_t l1Offset = (curL1Iter & 1) * (L1A_PP_SIZE >> 2);
WaitFlag<HardEvent::MTE1_MTE2>(mte1ToMte2A[curL1Iter & 1]);
LoadDataToL1<enableZZ>(aL1Tensor_[l1Offset], gm, m, k);
SetFlag<HardEvent::MTE2_MTE1>(mte2ToMte1A[curL1Iter & 1]);

// L0 PingPong
size_t aL0Offset = (aL0CurIter & 1) * (L0A_PP_SIZE >> 2);
WaitFlag<HardEvent::M_MTE1>(mToMte1A[aL0CurIter & 1]);
LoadDataToL0A<enTranspose>(aL0Tensor_[aL0Offset], aL1Tensor, ...);
SetFlag<HardEvent::MTE1_M>(mte1ToMA[aL0CurIter & 1]);
```

Benefit: GM→L1→L0→Mmad 三级流水完全重叠，理论吞吐接近 Cube 峰值算力
Trade-off: 每级存储空间减半；6+ 组 event ID 管理复杂度极高
