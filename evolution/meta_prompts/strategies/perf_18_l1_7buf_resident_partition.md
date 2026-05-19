# P18: L1 7-Buffer Resident Partitioning (L1 七缓冲常驻分区)
## Overview
针对 Cube 核上 GM→L1→L0→Cube 的多级数据搬运流水线，将 L1 总容量（512KB）静态划分为 7 个等大小块（每块 72KB）：QP 区 4 块（2×2 矩阵布局）用于存放 Query/Softmax-P 数据，KV 区 3 块（三路旋转）用于存放 Key/Value 数据。QP 区采用"常驻"策略——仅在 N 方向首次迭代时从 GM 加载，后续 N 迭代直接复用 L1 中已有数据，消除冗余搬运；KV 区采用三路旋转缓冲，比双缓冲多一个空闲槽位，使 GM→L1 加载、L1→L0 消费、下一轮预加载三阶段完全重叠。该方案源自 FIA（Fused Infer Attention）MLA 高性能模板，专为 headDim=512+rope=64 的 MLA 场景设计。该方案同时支持 MTE2→L1→L0→Mmad 三级全流水调度（见 Variant D）。

## When to Use
- MLA/DeepSeek 类算子 headDim 较大（≥288），标准双缓冲无法充分利用 L1 容量
- L1 总容量 ≥512KB，可容纳 7 个独立 buffer（4 个 QP + 3 个 KV）
- Cube 密集型算子中 Q 矩阵在多轮 KV 迭代中保持不变，复用率高（如 N_loops≥4 时搬运量减少 75%）

## Trade-off
- L1 占用率极高（504KB/512KB），几乎无剩余空间，不适合需要额外 L1 临时缓冲的场景
- 7 个 buffer 需要 7 个独立的 MTE1_MTE2 硬件事件 ID 进行同步管理，事件资源紧张
- 切分尺寸（M=128, K=288, N=128）为 MLA headDim=576 特化，泛化到其他 headDim 需要重新计算分区

**Source operators**: ai_infra_fused_infer_attention_sink (MLA mode)

---

## Variant A: QP 4-Buffer 常驻复用（Matrix2×2 布局）
Source: ai_infra_fused_infer_attention_sink

将 L1 中 QP 区划分为 4 块，按 2×2 矩阵组织：行维度对应 M 方向切分（最多 2 块），列维度对应 K 方向切分（2 块，对应 headDim=576 的两半 288）。通过 `GetQPL1RealIdx(mIdx, kL1Idx)` 映射确保同一 M 块的两个 K 半块在物理地址上连续，使得 Q_nope（256d）和 Q_rope（64d）可以紧密排列。关键优化：仅在 N 方向首次迭代（`nL1 == 0`）时执行 GM→L1 搬运，后续 N 迭代通过 `ReuseNext()` 直接复用已驻留的 L1 数据。

**Expert implementation:**
```cpp
// L1 buffer 定义：QP 4块，每块 72KB
static constexpr uint32_t L1_BLOCK_SIZE = M_L1_SPLIT_SIZE * MM1_K_L1_SPLIT_SIZE * sizeof(Q_T);
// = 128 * 288 * 2 = 73728 bytes (72KB)

// 初始化：分配 4 块连续 L1 空间
pipe->InitBuffer(bufQPL1, L1_BLOCK_SIZE * 4); // 288KB
l1QPTensor = bufQPL1.Get<Q_T>();

// 2×2 索引映射：确保同一 M 块的两个 K 半块地址连续
__aicore__ inline uint32_t GetQPL1RealIdx(uint32_t mIdx, uint32_t k1Idx) {
    uint32_t idxMap[] = {0, 2}; // block 0,1 连续; block 2,3 连续
    return idxMap[mIdx % 2] + k1Idx;
}

// N 方向循环中的常驻复用
for (uint32_t nL1 = 0; nL1 < nL1Loops; nL1++) {
    for (uint32_t kL1 = 0; kL1 < kL1Loops; kL1++) {
        uint32_t ka = GetQPL1RealIdx(mIdx, kL1);
        LocalTensor<Q_T> aL1Tensor = l1QPTensor[ka * L1_BLOCK_OFFSET];

        if (nL1 == 0) {
            // 首次 N 迭代：从 GM 加载 Q 数据到 L1
            WaitFlag<HardEvent::MTE1_MTE2>(mte21QPIds[ka]);
            copyQueryGmToL1(dstTensor, queryGmTensor, gmCoord);
            // Q_rope 紧接 Q_nope 之后存放
            copyQueryGmToL1(dstRopeTensor, queryRopeGmTensor, gmCoordQRope);
            SetFlag<HardEvent::MTE2_MTE1>(mte21QPIds[ka]);
            WaitFlag<HardEvent::MTE2_MTE1>(mte21QPIds[ka]);
        }
        // nL1 > 0 时：直接复用 L1 中已有的 Q 数据，零搬运开销
    }
}
```

**vs. baseline (lingxi-code):**
```cpp
// 基线方案：QP 双缓冲，每次 N 迭代都重新加载 Q
static constexpr uint32_t L1_QP_SIZE = 128 * 1024;
pipe->InitBuffer(qpBufL1, L1_QP_SIZE * 2); // 256KB，仅 2 块

// 每次 N 迭代都要重新搬运 Q 数据
for (uint32_t nStart = 0; nStart < nSize; nStart += N_SPLIT_SIZE) {
    if (nStart == 0) {
        WaitFlag<HardEvent::MTE1_MTE2>(QP_EVENT0 + qpBufId % 2);
        CopyQGmToL1(...); // 每次都搬运
        SetFlag<HardEvent::MTE2_MTE1>(QP_EVENT0 + qpBufId % 2);
    }
    // nStart > 0 时仍需切换 buffer，无法常驻
}
```

Benefit: 消除 N 方向内循环中 Q/P 数据的冗余 GM→L1 搬运，搬运次数从 O(N_loops) 降为 O(1)；对于 s2BaseSize=512、N_L1_SPLIT=128 的典型配置，Q 搬运量减少 75%
Trade-off: 需要 4 块 L1 buffer（288KB），比双缓冲多占 32KB；2×2 索引映射增加代码复杂度

---

## Variant B: KV 3-Buffer 三路旋转缓冲
Source: ai_infra_fused_infer_attention_sink

KV 区使用 3 块 L1 buffer 进行三路旋转（triple buffering），相比双缓冲多一个空闲槽位。在任意时刻：一块正在被 MTE1 消费（L1→L0 搬运），一块正在被 MTE2 填充（GM→L1 搬运），第三块空闲等待下一轮 MTE2 写入。三路旋转通过 `kvL1BufIter % 3` 索引和 3 个独立的 MTE1_MTE2 事件 ID 管理。

**Expert implementation:**
```cpp
// L1 buffer 定义：KV 3块，每块 72KB
pipe->InitBuffer(bufKVL1, L1_BLOCK_SIZE * 3); // 216KB
l1KVTensor = bufKVL1.Get<KV_T>();

// 3 个独立事件 ID
static constexpr uint32_t mte21KVIds[3] = {L1_EVENT4, L1_EVENT5, L1_EVENT6};

// 三路旋转使用
for (uint32_t kL1 = 0; kL1 < kL1Loops; kL1++) {
    kvL1BufIter++;
    uint32_t kb = kvL1BufIter % 3; // 三路轮转

    WaitFlag<HardEvent::MTE1_MTE2>(mte21KVIds[kb]); // 等待该块可写
    bL1Tensor = l1KVTensor[kb * L1_BLOCK_OFFSET];

    // GM→L1：加载 K 或 V 数据
    copyKvGmToL1(dstTensor, keyGmTensor, gmCoord);

    SetFlag<HardEvent::MTE2_MTE1>(mte21KVIds[kb]); // 标记加载完成
    WaitFlag<HardEvent::MTE2_MTE1>(mte21KVIds[kb]);

    // L1→L0→Cube 计算...

    SetFlag<HardEvent::MTE1_MTE2>(mte21KVIds[kb]); // 标记消费完成，释放该块
}
```

**vs. baseline (lingxi-code):**
```cpp
// 基线方案：KV 双缓冲
static constexpr uint32_t KV_EVENT0 = EVENT_ID4;
static constexpr uint32_t KV_EVENT1 = EVENT_ID5;
uint32_t kvBufId = 0;

// 双缓冲轮转
WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + kvBufId % 2);
uint64_t l1Offset = (kvBufId % 2) * (L1_KV_SIZE / sizeof(KV_T));
CopyKeyToL1(info, ...);
SetFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + kvBufId % 2);
// 双缓冲：MTE2 加载和 MTE1 消费只能两阶段重叠
```

Benefit: 三路旋转使 GM→L1 加载、L1→L0 消费、下一轮预加载三阶段完全重叠，消除双缓冲中"等待消费完成才能开始下一次加载"的气泡
Trade-off: 比双缓冲多占 72KB L1 空间；多一个事件 ID 占用

---

## Variant C: L1 容量感知的 7-Buffer 联合分区
Source: ai_infra_fused_infer_attention_sink

Host 侧 tiling 根据 L1 总容量（512KB）和 MLA headDim（512+64=576）联合计算 7 个 buffer 的切分尺寸。每块大小 = M_L1_SPLIT × MM1_K_L1_SPLIT × sizeof(dtype) = 128 × 288 × 2 = 72KB。7 块总计 504KB，占 L1 的 98.4%。K 方向切分为 288 = 576/2，恰好将 MLA 的 nope（512d）和 rope（64d）拼接后的总维度对半分。M 方向切分为 128，与 Cube 单元的 fractal 基本块对齐。

**Expert implementation:**
```cpp
// Host 侧：切分参数定义
static constexpr uint32_t M_L1_SPLIT_SIZE = 128;       // M 方向 L1 切分
static constexpr uint32_t N_L1_SPLIT_SIZE = 128;       // N 方向 L1 切分
static constexpr uint32_t MM1_K_L1_SPLIT_SIZE = 288;   // K 方向 L1 切分 = (512+64)/2
static constexpr uint32_t MM1_K_L0_SPLIT_SIZE = 96;    // K 方向 L0 切分 = 288/3

// 每块大小计算
static constexpr uint32_t L1_BLOCK_SIZE =
    M_L1_SPLIT_SIZE * MM1_K_L1_SPLIT_SIZE * sizeof(Q_T);
// = 128 * 288 * 2 = 73728 bytes ≈ 72KB

// L1 总分配
// QP: 72KB × 4 = 288KB
// KV: 72KB × 3 = 216KB
// Total: 504KB / 512KB = 98.4% 利用率

// Kernel 侧：L1→L0 进一步切分
uint32_t kL1Size = MM1_K_L1_SPLIT_SIZE; // 288
uint32_t kL0Size = 96;
uint32_t kL0Loops = (kL1Size + kL0Size - 1) / kL0Size; // 288/96 = 3 次 L0 迭代
```

**vs. baseline (lingxi-code):**
```cpp
// 基线方案：QP/KV 各 128KB×2，未充分利用 L1
static constexpr uint32_t L1_QP_SIZE = 128 * 1024; // 128KB
static constexpr uint32_t L1_KV_SIZE = 128 * 1024; // 128KB
// QP: 128KB × 2 = 256KB
// KV: 128KB × 2 = 256KB
// Total: 512KB，但每块内部利用率不高

// M/K/N 切分未针对 MLA headDim 特化
constexpr uint32_t M_SPLIT_SIZE = 256;
constexpr uint32_t K_SPLIT_SIZE = 512;
constexpr uint32_t N_SPLIT_SIZE = 128;
```

Benefit: L1 利用率达 98.4%，切分尺寸与 MLA headDim=576 精确匹配；K 方向 288 的切分使 nope/rope 数据在 L1 中紧密排列，减少碎片
Trade-off: 切分参数为 MLA headDim=576 特化，其他 headDim 需要重新计算；L1 几乎满载，无法为其他用途预留空间

---

## Variant D: L1 多 Buffer 常驻实现搬运-计算全流水
Source: sparse_flash_attention_enhance

4 个 Q buffer + 3 个 KV buffer 常驻 L1，通过 kvL1BufIter % 3 轮转索引实现 MTE2 搬入与 L0 消费的全流水。

**Expert implementation:**
```cpp
pipe->InitBuffer(bufQPL1, L1_BLOCK_SIZE * 4);  // 288K
pipe->InitBuffer(bufKVL1, L1_BLOCK_SIZE * 3);  // 216K
kvL1BufIter++;
uint32_t kb = kvL1BufIter % 3;
WaitFlag<HardEvent::MTE1_MTE2>(mte21KVIds[kb]);
bL1Tensor = l1KVTensor[kb * L1_BLOCK_OFFSET];
SetFlag<HardEvent::MTE2_MTE1>(mte21KVIds[kb]);
```

Benefit: MTE2→L1、L1→L0、L0 Mmad 三级全流水，Cube 利用率最大化
Trade-off: 504K L1 占用接近总容量，留给其他 buffer 的空间极少
