# P83 L1 Resident Reuse with Multi-Buffer Partitioning (L1 常驻复用与多 Buffer 分区)
## Overview
在 FlashAttention、IFA、MLA 等 Cube 密集型算子中，可将 Q 或左矩阵在首次迭代时搬入 L1 后常驻，并结合 L1 多 buffer 分区，把“跨 N/S2 迭代复用”与“多阶段预取/消费流水”统一起来。该策略的核心不是固定采用某个 buffer 数量，而是先判断左矩阵是否在后续迭代保持不变，再依据 L1 容量把空间划分为常驻区与轮转区：常驻区负责保存跨迭代复用的 Q/左矩阵，轮转区负责承载 KV 或后续分块的预取与消费。若 shape 与 L1 预算允许，可进一步落到 MLA 类场景常见的 4 个 QP buffer + 3 个 KV buffer 的 7-buffer 分区。

## When to Use
- Cube 密集型算子中 Q/左矩阵在多个 N/S2/KV 迭代中保持不变，重复搬运成为主要成本
- L1 容量足以同时容纳常驻左矩阵和至少 2~3 个轮转 buffer，值得用空间换搬运带宽
- headDim 较大（如 ≥288）或 MLA/IFA 类场景中，标准双缓冲不能充分利用 L1 容量
- 需要在“左矩阵复用”之外继续维持 KV 预取或 L1 分区轮转，以减少 GM→L1 搬运气泡
- 适用于仍以 Matmul 库或既有 Cube 数据通路为主、主要优化目标是减少 GM→L1 重复搬运或提升 L1 侧复用率的场景
- 若 kernel 已绕过 Matmul 库，并手动调度 L0A/L0B/L0C 分配、L1→L0 搬运和 Mmad 时序，应优先参考 P81，而不是本策略

## Trade-off
- 左矩阵常驻会长期占用 L1，大常驻期间可分配给其他中间结果或 P buffer 的空间显著减少；例如 headDim=576、M=128 时仅 Q 常驻就约需 144KB
- 多 buffer 分区通常把 L1 利用率推到极高水平；以 MLA 7-buffer 分区为例，7×72KB 约占 504KB/512KB，迁移到其他 headDim 或算子形状时需要重新计算切分
- 常驻区与轮转区需要独立索引和事件同步，代码复杂度明显高于普通双缓冲
- 首轮加载和切换阶段可能有额外断流风险，只有在后续复用次数足够多时收益明显

**Source operators**: ai_infra_fused_infer_attention_sink, IFA, PFA

---

## Variant A: 左矩阵跨 N/S2 迭代的 L1 大常驻
Source: IFA/PFA operator kernel（代码模式）, 【基础知识】L1内存复用：L1常驻减少内存重复搬运耗时.md（机制说明）

当 Q 或左矩阵在多个 N/S2 迭代中保持不变时，首轮将其搬入 L1，后续循环直接复用同一块 L1 数据。核心判断点不在 buffer 数量，而在于左矩阵是否跨迭代不变，以及 L1 是否还有足够余量支撑大常驻。

**Expert implementation:**
```cpp
__aicore__ inline LocalTensor<TRANS_T> LoadData(int curRow, int curCol,
    int tileHeight, int tileWidth, int batchNum = -1) {
    LocalTensor<TRANS_T> l1;
    UserDefDataType flag = MATMUL_PARAM_VAR.dataPtr_;
    LocalTensor<TRANS_T> dst;
    if (flag.reuseLeft) {
        // 非首轮：直接复用 L1 中已有的 Q/左矩阵数据
        l1 = MATMUL_MODULE(CubeInBuffer)->GetBuffer(flag.leftBufIdx);
        dst = l1[static_cast<int64_t>(callTimes_) * baseWidth_ * 64];
        ++callTimes_;
        return dst;
    } else {
        // 首轮：从 GM 加载 Q/左矩阵数据到 L1
        l1 = MATMUL_MODULE(CubeInBuffer)->AllocTensor(flag.leftBufIdx);
        dst = l1[static_cast<int64_t>(callTimes_) * baseWidth_ * 64];
        // ... 执行 DataCopy nd2nz ...
    }
    ++callTimes_;
    return dst;
}
```

**vs. baseline (lingxi-code):**
```cpp
for (uint32_t n = 0; n < nloops; n++) {
    CopyQToL1(info, mL1Size);
    // ... 计算 ...
}
```

Benefit: 将 Q/左矩阵搬运次数从 O(N_loops) 降到 O(1)，跨 S2/N 方向循环越深收益越明显。
Trade-off: 大常驻期间 L1 空间被左矩阵长期占用，后续阶段可用缓冲明显减少。

---

## Variant B: QP 4-buffer 常驻 + KV 3-buffer 轮转的 7-buffer 分区
Source: ai_infra_fused_infer_attention_sink

对 MLA headDim=576 一类大尺寸左矩阵场景，可把 L1 划分为 QP 4 块常驻区与 KV 3 块轮转区。QP 区按 2×2 矩阵布局保存 Q/Softmax-P，并仅在首次 N 迭代时搬入；KV 区使用三路轮转，使 GM→L1 加载、L1→L0 消费和下一轮预取并行推进。

**Expert implementation:**
```cpp
static constexpr uint32_t L1_BLOCK_SIZE =
    M_L1_SPLIT_SIZE * MM1_K_L1_SPLIT_SIZE * sizeof(Q_T);
pipe->InitBuffer(bufQPL1, L1_BLOCK_SIZE * 4);  // QP 常驻区
pipe->InitBuffer(bufKVL1, L1_BLOCK_SIZE * 3);  // KV 轮转区

// Phase 1: QP 常驻区。仅 nL1==0 时执行 GM→L1，后续 N 方向迭代复用同一批 L1 数据。
__aicore__ inline uint32_t GetQPL1RealIdx(uint32_t mIdx, uint32_t k1Idx) {
    uint32_t idxMap[] = {0, 2};
    return idxMap[mIdx % 2] + k1Idx;
}

for (uint32_t nL1 = 0; nL1 < nL1Loops; nL1++) {
    for (uint32_t kL1 = 0; kL1 < kL1Loops; kL1++) {
        uint32_t ka = GetQPL1RealIdx(mIdx, kL1);
        LocalTensor<Q_T> aL1Tensor = l1QPTensor[ka * L1_BLOCK_OFFSET];
        if (nL1 == 0) {
            WaitFlag<HardEvent::MTE1_MTE2>(mte21QPIds[ka]);
            copyQueryGmToL1(dstTensor, queryGmTensor, gmCoord);
            copyQueryGmToL1(dstRopeTensor, queryRopeGmTensor, gmCoordQRope);
            SetFlag<HardEvent::MTE2_MTE1>(mte21QPIds[ka]);
            WaitFlag<HardEvent::MTE2_MTE1>(mte21QPIds[ka]);
        }
    }
}

// Phase 2: KV 轮转区。KV block 按三路轮转推进，与上方 QP 常驻区共享 L1 分区方案，
// 但生命周期不同：QP 跨 N 复用，KV 随 kL1 迭代加载、消费、释放。
for (uint32_t kL1 = 0; kL1 < kL1Loops; kL1++) {
    kvL1BufIter++;
    uint32_t kb = kvL1BufIter % 3;
    WaitFlag<HardEvent::MTE1_MTE2>(mte21KVIds[kb]);
    bL1Tensor = l1KVTensor[kb * L1_BLOCK_OFFSET];
    copyKvGmToL1(dstTensor, keyGmTensor, gmCoord);
    SetFlag<HardEvent::MTE2_MTE1>(mte21KVIds[kb]);
    WaitFlag<HardEvent::MTE2_MTE1>(mte21KVIds[kb]);
    // ... L1→L0/Cube 消费 bL1Tensor ...
    SetFlag<HardEvent::MTE1_MTE2>(mte21KVIds[kb]);
}
```

**vs. baseline (lingxi-code):**
```cpp
static constexpr uint32_t L1_QP_SIZE = 128 * 1024;
static constexpr uint32_t L1_KV_SIZE = 128 * 1024;
pipe->InitBuffer(qpBufL1, L1_QP_SIZE * 2);

for (uint32_t nStart = 0; nStart < nSize; nStart += N_SPLIT_SIZE) {
    WaitFlag<HardEvent::MTE1_MTE2>(QP_EVENT0 + qpBufId % 2);
    CopyQGmToL1(...);
    SetFlag<HardEvent::MTE2_MTE1>(QP_EVENT0 + qpBufId % 2);
}
```

Benefit: 同时消除 Q 重复搬运，并让 KV 侧形成加载/消费/预取三阶段重叠，最大化 L1 利用率。
Trade-off: 7-buffer 会把 L1 占用推近上限，通常只适合特定 MLA/headDim 配置。

---

## Variant C: L1 容量感知的小常驻与切分特化
Source: 【基础知识】L1内存复用：L1常驻减少内存重复搬运耗时.md, ai_infra_fused_infer_attention_sink

当 L1 余量不足以支持跨 N/S2 的大常驻或完整 7-buffer 分区时，可退化为“小常驻 + 特化切分”：只在单个基本块内部、K 轴切分循环开始前搬入一次左矩阵，后续局部 K/N 子循环直接复用；若 headDim 固定，还可通过 host 侧预先计算 M/K/N 切分，尽量让常驻区与轮转区贴合实际 shape。该 Variant 是 Variant A 的容量不足 fallback，复用边界更小，不承担跨整个 S2/N 外循环的长生命周期常驻。

**Expert implementation:**
```cpp
template <CubeFormat OutFormat>
__aicore__ inline void ComputeMm1(const SplitSameABExtraInfo &info) {
    auto qTensor = CopyQToL1(info, mL1Size);
    for (uint32_t n = 0; n < nloops; n++) {
        if (!l1BufLoaded[kvL1BufIter % 2]) {
            CopyKToL1(info, n * nSplitSize, subNSizeAct);
        }
        // ... 使用 qTensor，不再重新拷贝 Q ...
    }
}

static constexpr uint32_t M_L1_SPLIT_SIZE = 128;
static constexpr uint32_t N_L1_SPLIT_SIZE = 128;
static constexpr uint32_t MM1_K_L1_SPLIT_SIZE = 288;
static constexpr uint32_t MM1_K_L0_SPLIT_SIZE = 96;
```

**vs. baseline (lingxi-code):**
```cpp
for (uint32_t k = 0; k < kloops; k++) {
    CopyQToL1(info, mL1Size);
    CopyKToL1(info, ...);
    MMAD();
}
```

Benefit: 在无法承担完整大常驻/7-buffer 的情况下，仍能消除局部循环中的左矩阵重复搬运。
Trade-off: 复用边界缩小为单基本块或局部切分循环，收益弱于跨 N/S2 的大常驻。
