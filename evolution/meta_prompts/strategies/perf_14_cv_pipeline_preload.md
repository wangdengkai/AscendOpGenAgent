# P14: CV Pipeline Preload (CV 流水预发射)
## Overview
针对 Cube+Vector 融合算子（如 Flash Attention 系列），利用昇腾 AI Core 中 Cube 核与 Vector 核可并行执行的硬件特性，将多轮迭代的4个计算阶段（MM1: Q×K, Vec1: softmax, MM2: S×V, Vec2: output update）通过3槽环形任务缓存交错编排，使不同迭代的 Cube 和 Vector 阶段重叠执行。配合6个跨核同步 flag 精确管理数据依赖，以及 workspace 双缓冲消除读写冲突，实现 Cube 和 Vector 的持续并行，消除流水空泡。适用于训练（SFA/Prefill）和推理（FIA/Decode）场景。

## When to Use
- CV fused operators with iterative Cube-Vector data dependency chain
- 算子计算流程包含多轮 Cube→Vector→Cube→Vector 迭代模式（如 Flash Attention: MM1→Softmax→MM2→OutputUpdate）
- Profiling 显示 Cube 或 Vector 利用率低，存在明显的跨核等待空泡（一方计算时另一方空闲）
- 适用于训练 Prefill（SFA）和推理 Decode（FIA）场景的 Attention 类算子

## Trade-off
- 代码复杂度显著增加：需维护3槽环形任务缓存、6个跨核同步 flag、workspace 双缓冲索引，三者必须协调一致
- Workspace 内存翻倍（PRELOAD_NUM=2），每个 AI Core 需要2份中间结果空间（mm1Res, vec1Res, mm2Res, vec2Res）
- 流水排空阶段需特别注意同步信号收发平衡和累加模式（SetAtomicAdd/SetAtomicNone）切换，否则导致精度错误
- 调试难度高：跨核同步错误表现为死锁或静默数据错误，难以复现和定位

**Source operators**: ai_infra_sparse_flash_attention, ai_infra_fused_infer_attention_sink

---

## Variant A: 3阶段流水编排 (3-Stage Pipeline Orchestration)
Source: ai_infra_sparse_flash_attention, ai_infra_fused_infer_attention_sink

使用3槽环形任务缓存（`PRELOAD_TASK_CACHE_SIZE = 3`），在 `ExecuteTask` 中将不同任务的 Cube 和 Vector 阶段交错执行：

| 阶段 | Cube 核 (AIC) | Vector 核 (AIV) |
|------|---------------|-----------------|
| Stage 0 (当前轮 N) | MM1: Q×K → S | — |
| Stage 1 (上一轮 N-1) | MM2: S×V → O | Vec1: softmax(S) → A |
| Stage 2 (上两轮 N-2) | — | Vec2: output update |

流水时序示意：
```
时间步:    T0          T1          T2          T3          ...
Cube:   MM1[task0]  MM1[task1]  MM1[task2]  MM1[task3]  ...
                    MM2[task0]  MM2[task1]  MM2[task2]  ...
Vector:             Vec1[task0] Vec1[task1] Vec1[task2] ...
                                Vec2[task0] Vec2[task1] ...
```

**流水排空 (Pipeline Drain)**: 当任务分发结束后，循环继续执行直到所有已创建任务的3个阶段全部完成。通过 `isValid` 标志逐阶段关闭，确保最后的 Vec1+MM2 和 Vec2 正确执行。排空阶段需特别注意跨核同步信号的收发平衡和累加模式（`SetAtomicAdd`/`SetAtomicNone`）的正确切换。

**Expert implementation:**
```cpp
// ==================== 常量定义（SFA/FIA 通用）====================
static constexpr uint32_t PRELOAD_NUM = 2;                    // workspace ping-pong
static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = 3;         // 3阶段环形任务缓存
// SFA 中为 SFA_PRELOAD_TASK_CACHE_SIZE, FIA 中为 FIA_PRELOAD_TASK_CACHE_SIZE

// ==================== 任务执行：3阶段交错 ====================
__aicore__ inline void ExecuteTask(uint64_t loop,
    RunInfo extraInfo[PRELOAD_TASK_CACHE_SIZE])
{
    RunInfo &extraInfo0 = extraInfo[loop % PRELOAD_TASK_CACHE_SIZE];       // 本轮任务
    RunInfo &extraInfo2 = extraInfo[(loop + 2) % PRELOAD_TASK_CACHE_SIZE]; // 上一轮任务
    RunInfo &extraInfo1 = extraInfo[(loop + 1) % PRELOAD_TASK_CACHE_SIZE]; // 上两轮任务

    // Stage 0: Cube 预发射当前轮 MM1 (Q×K)
    if (extraInfo0.isValid) {
        if ASCEND_IS_AIC {
            ComputeMm1(extraInfo0);
        }
    }
    // Stage 1: Vector 处理上一轮 softmax + Cube 执行上一轮 MM2 (S×V)
    if (extraInfo2.isValid) {
        if ASCEND_IS_AIV {
            ComputeVec1(extraInfo2);
        }
        if ASCEND_IS_AIC {
            ComputeMm2(extraInfo2);
        }
    }
    // Stage 2: Vector 完成上两轮的输出累加
    if (extraInfo1.isValid) {
        if ASCEND_IS_AIV {
            ComputeVec2(extraInfo1);
        }
        extraInfo1.isValid = false;   // 标记任务完成
    }
}

// ==================== 主循环：任务分发 + 流水执行 + 排空 ====================
__aicore__ inline void FlashAttention()
{
    RunInfo extraInfo[PRELOAD_TASK_CACHE_SIZE];
    uint64_t createdTaskCount = 0;
    uint64_t executedTaskCount = 0;
    bool shouldDispatchTask = true;
    bool shouldExecuteTask = false;

    while (shouldDispatchTask || shouldExecuteTask) {
        // 分发任务：创建新任务填入环形缓存
        shouldDispatchTask = ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur);
        if (shouldDispatchTask) {
            TASK_DEAL_MODE taskDealMode = GetTaskDealMode(bN2Cur, gS1Cur, s2Cur);
            if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
                CreateTask(createdTaskCount, bN2Cur, gS1Cur, s2Cur, extraInfo);
                createdTaskCount++;
                UpdateAxisInfo(bN2Cur, gS1Cur, s2Cur);
            }
        }
        // 执行任务：3阶段交错执行
        shouldExecuteTask = ShouldExecuteTask(extraInfo);
        if (shouldExecuteTask) {
            ExecuteTask(executedTaskCount, extraInfo);
            executedTaskCount++;
        }
        // 当 shouldDispatchTask=false 时进入排空阶段，
        // 继续执行直到所有 isValid 任务完成
    }
}
```

**vs. baseline (lingxi-code):**
```cpp
// 无流水编排的顺序执行：Cube 和 Vector 交替空等
for (uint64_t loop = 0; loop < totalLoops; loop++) {
    RunInfo info;
    CalcParams(loop, bN2Cur, gS1Cur, s2Cur, info);

    if ASCEND_IS_AIC { ComputeMm1(info); }
    SyncAll();  // 全局同步 — 所有核等待

    if ASCEND_IS_AIV { ComputeVec1(info); }
    SyncAll();

    if ASCEND_IS_AIC { ComputeMm2(info); }
    SyncAll();

    if ASCEND_IS_AIV { ComputeVec2(info); }
    SyncAll();  // 下一轮才能开始
}
```

Benefit: Cube 和 Vector 核几乎无空等，流水利用率接近理论峰值；3槽缓存确保流水线始终满载
Trade-off: 需要维护3份任务上下文的循环缓存，流水排空阶段需确保同步信号收发平衡

---

## Variant B: 跨核同步Flag协议 (Cross-Core Sync Flag Protocol)
Source: ai_infra_sparse_flash_attention, ai_infra_fused_infer_attention_sink

定义6个专用同步 Flag，通过 `CrossCoreSetFlag` / `CrossCoreWaitFlag` 原语精确管理 Cube→Vector→Cube→Vector 的数据依赖链。每个 Flag 对应一个特定的生产者-消费者关系，避免使用全局 `SyncAll()` 带来的不必要等待。

同步依赖链：
```
                    ┌──[syncC2V1]──┐
                    ↓              │
Cube MM1 ──[syncC1V1]──→ Vector Vec1 ──[syncV1C2]──→ Cube MM2 ──[syncC2V2]──→ Vector Vec2
                                                        ↑                         │
                                                        └──[syncV1NupdateC2]──────┘
```

| Flag ID | 名称 | 生产者 | 消费者 | 含义 | Pipe 类型 |
|---------|------|--------|--------|------|-----------|
| 7 | syncC1V1 | Cube (MM1完成) | Vector (Vec1开始) | MM1结果S就绪 | PIPE_FIX |
| 8 | syncV1C2 | Vector (Vec1完成) | Cube (MM2开始) | softmax结果A就绪 | PIPE_MTE3 |
| 9 | syncC2V2 | Cube (MM2完成) | Vector (Vec2开始) | MM2结果O就绪 | PIPE_FIX |
| 4 | syncC2V1 | Cube (MM2完成) | Vector (下一轮Vec1) | workspace槽位释放 | PIPE_FIX |
| 5 | syncV1NupdateC2 | Vector (N-update完成) | Cube (下一轮MM2) | 输出缓冲区可写 | PIPE_MTE3 |
| 6 | syncV0C1 | Vector (初始化完成) | Cube (首轮MM1) | 初始状态就绪 | PIPE_FIX |

Cube 侧使用 `PIPE_FIX`（Fixpipe 输出后发信号），Vector 侧使用 `PIPE_MTE3`（MTE3 写出后发信号）。选择正确的 Pipe 类型确保信号在数据实际写出后才发出。

**Expert implementation:**
```cpp
// ==================== Flag 定义（SFA/FIA 通用）====================
static constexpr uint32_t SYNC_V0_C1_FLAG = 6;
static constexpr uint32_t SYNC_C1_V1_FLAG = 7;
static constexpr uint32_t SYNC_V1_C2_FLAG = 8;
static constexpr uint32_t SYNC_C2_V2_FLAG = 9;
static constexpr uint32_t SYNC_C2_V1_FLAG = 4;
static constexpr uint32_t SYNC_V1_NUPDATE_C2_FLAG = 5;

// ==================== Cube 侧 (ComputeMm1) ====================
// Cube 完成 MM1 后，通知 Vector 可以开始 Vec1
// 在 nBufferLoop 内每完成一个 M 分块的 MM1 即发信号
CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V1);

// ==================== Vector 侧 (ComputeVec1) ====================
// Vector 等待 Cube MM1 完成
CrossCoreWaitFlag(constInfo.syncC1V1);
ProcessVec1SingleBuf(info);
// Vec1 完成后，通知 Cube 可以开始 MM2
CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1C2);

// ==================== Cube 侧 (ComputeMm2) ====================
// Cube 等待 Vector Vec1 完成
CrossCoreWaitFlag(constInfo.syncV1C2);
// ... MM2 matmul computation ...
// MM2 完成后，发两个信号：
CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC2V2);  // → Vec2
CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC2V1);  // → 下一轮 Vec1

// ==================== Vector 侧 (ComputeVec2) ====================
// Vector 等待 Cube MM2 完成
CrossCoreWaitFlag(constInfo.syncC2V2);
ProcessVec2SingleBuf(info);
// Vec2 中 N-update 完成后，通知 Cube 可以开始下一轮 MM2
CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1NupdateC2);
```

**vs. baseline (lingxi-code):**
```cpp
// 使用全局 SyncAll 的粗粒度同步：所有核等待，包括不相关的核
ComputeMm1(info);
SyncAll();          // 全局屏障
ComputeVec1(info);
SyncAll();
ComputeMm2(info);
SyncAll();
ComputeVec2(info);
SyncAll();
```

Benefit: 精确的点对点同步，仅阻塞有数据依赖的核，其他核可继续执行下一任务的计算
Trade-off: Flag 编号管理复杂，错误的 Flag 配对会导致死锁或数据竞争；需要根据流水深度仔细规划 Flag 数量和收发平衡

---

## Variant C: Workspace Ping-Pong 双缓冲 (Double-Buffered GM Workspace)
Source: ai_infra_sparse_flash_attention, ai_infra_fused_infer_attention_sink

在 GM workspace 中为每个中间结果缓冲区（mm1ResGm, vec1ResGm, mm2ResGm, vec2ResGm）分配 `PRELOAD_NUM = 2` 的双缓冲空间。每个 AI Core 拥有独立的双份 workspace slot，Cube 写入 slot[loop%2] 的同时，Vector 可以读取 slot[(loop-1)%2]，实现读写并行，消除 GM 访问冲突。这是3阶段流水线能够运转的内存基础。

**Expert implementation:**
```cpp
static constexpr uint32_t PRELOAD_NUM = 2;  // 双缓冲
static constexpr uint32_t dbWorkspaceRatio = PRELOAD_NUM;

__aicore__ inline void InitWorkspace(__gm__ uint8_t *workspace)
{
    uint64_t offset = 0;

    // mm1Res: 每个 core 分配 2 × mmResUbSize（双缓冲）
    mm1ResGm.SetGlobalBuffer(
        (__gm__ MM1_OUT_T *)(workspace + offset +
            aiCoreIdx * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(MM1_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(MM1_OUT_T);

    // vec1Res: softmax 结果双缓冲
    vec1ResGm.SetGlobalBuffer(
        (__gm__ KV_T *)(workspace + offset +
            aiCoreIdx * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(KV_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(KV_T);

    // mm2Res: S×V 结果双缓冲
    mm2ResGm.SetGlobalBuffer(
        (__gm__ MM2_OUT_T *)(workspace + offset +
            aiCoreIdx * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(MM2_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(MM2_OUT_T);

    // vec2Res: output update 结果双缓冲
    vec2ResGm.SetGlobalBuffer(
        (__gm__ UPDATE_T *)(workspace + offset +
            aiCoreIdx * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(UPDATE_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(UPDATE_T);
}

// ==================== Ping-Pong 索引使用 ====================
// MM1 Fixpipe 输出：loop % PRELOAD_NUM 选择写入槽位
Fixpipe(mm1ResGm[(info.loop % constInfo.preLoadNum) * constInfo.mmResUbSize + ...],
        cL0Tensor, fixParams);

// MM2 Fixpipe 输出：
// SFA 使用 loop % PRELOAD_NUM
// FIA MLA 使用 bn2IdxInCurCore % PRELOAD_NUM（因 MLA 的 head 维度处理不同）
Fixpipe(mm2ResGm[(info.bn2IdxInCurCore % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + ...],
        cL0Tensor, fixParams);

// Vec1 读取 MM1 结果：使用相同的 loop % PRELOAD_NUM 索引
auto srcGm = mm1ResGm[(info.loop % constInfo.preLoadNum) * constInfo.mmResUbSize + ...];
```

**vs. baseline (lingxi-code):**
```cpp
// 单缓冲：Cube 必须等 Vector 读完才能写入下一轮结果
static constexpr uint32_t SINGLE_BUFFER = 1;

mm1ResGm.SetGlobalBuffer(
    (__gm__ MM1_OUT_T *)(workspace + offset +
        aiCoreIdx * SINGLE_BUFFER * constInfo.mmResUbSize * sizeof(MM1_OUT_T)));
// ... 其他缓冲区同理，均为单份
// 结果：Cube 写入和 Vector 读取必须严格串行，无法流水
```

Benefit: 消除 Cube 写入与 Vector 读取之间的 GM 访问冲突，使流水交错执行成为可能
Trade-off: Workspace 内存占用翻倍；需要精确计算每个 core 的 offset 避免地址重叠；SFA 和 FIA 的索引策略略有差异需注意
