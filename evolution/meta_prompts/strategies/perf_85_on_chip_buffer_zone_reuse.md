# P85 On-Chip Buffer Zone Reuse (片上 Buffer 分区与分时复用)
## Overview
当算子包含多个串行计算阶段，且各阶段的临时 buffer 生命周期不重叠时，可将它们映射到同一块 UB TBuf 或 GM workspace 的不同逻辑区域，或直接共享同一物理区间，从 `sum(stage buffers)` 压缩到 `max(concurrent buffers)`。该策略的核心不是常驻数据跨循环复用，而是**同一次 `Process` 或单个 kernel 内多个阶段的分时复用**：前一阶段结束后，后一阶段覆盖其临时空间。它适用于 RmsNorm/RoPE、SoftMax+后处理、IFA 多阶段中间结果编排等串行阶段明确的场景。它不依赖 `pipe->Reset()` 做阶段级资源重建，也不处理跨迭代状态常驻；主要收益是释放片上或 workspace 容量给更大的 tile、双缓冲或额外流水深度。

## When to Use
- 算子包含两个或多个串行计算阶段，且这些阶段的临时 buffer 生命周期明确不重叠
- UB 或 workspace 容量紧张，希望通过共享物理空间减少总 buffer 占用
- 某阶段结束后其中间结果不再被后续阶段读取，可安全被覆盖
- 高阶 API（如 `SoftMax`）的临时 buffer 与其他计算阶段的临时空间不会同时存活
- 适用于单个 kernel 内按阶段推进的空间规划；若需要跨 loop/batch/iteration 保留状态，应优先参考 P67
- 若需要通过 `SyncAll() + pipe->Reset()` 在阶段间释放并重建整套资源，应优先参考 P79，而不是本策略

## Trade-off
- 各阶段必须具备严格的串行边界，错误判断生命周期会导致后续阶段覆盖仍在使用的数据
- 共享区通常依赖手工 offset、MAX 大小估算或 alias 规划，维护成本高于独立分配
- 当后续优化把原本串行阶段改造成重叠流水时，原有 zone reuse 方案可能失效，需要重新规划
- GM workspace 复用虽然减少总量，但不会像纯片上复用那样直接减少搬运次数；收益更多来自容量释放和整体布局简化

**Source operators**: ai_infra_kv_rms_norm_rope_cache, SIMD算子性能优化/内存访问, IFA, MLA 场景, sparse_lightning_indexer_grad_kl_loss_enhance

---

## Variant A: UB 内多区域分时复用
Source: ai_infra_kv_rms_norm_rope_cache

将一块大的 `TBuf<VECCALC>` 划分为多个逻辑 zone，前一阶段使用全部或部分 zone，后一阶段复用其中若干 zone 存放自己的临时变量。适合 RmsNorm、RoPE 等前后串行、局部临时量可覆盖的融合 Vector 场景。

**Expert implementation:**
```cpp
int64_t xLocalFp32Offset = 0;                     // zone0: RmsNorm fp32
int64_t xSquareLocalOffset = rows * headSize;    // zone1: RmsNorm square / RoPE cos
int64_t xSumLocalOffset = rows * headSize * 2;   // zone2: RmsNorm sum / RoPE sin
LocalTensor<float> xLocalFp32 = wsLocal[xLocalFp32Offset];
LocalTensor<float> xSquareLocal = wsLocal[xSquareLocalOffset];
LocalTensor<float> xSumLocal = wsLocal[xSumLocalOffset];
```

**vs. baseline (lingxi-code):**
```cpp
pipe.InitBuffer(rmsBuf0, 1, zoneSize);
pipe.InitBuffer(rmsBuf1, 1, zoneSize);
pipe.InitBuffer(rmsBuf2, 1, zoneSize);
pipe.InitBuffer(ropeCosBuf, 1, zoneSize);
pipe.InitBuffer(ropeSinBuf, 1, zoneSize);
// RmsNorm 和 RoPE 各自独立分配临时空间，无法复用同一块 wsLocal
```

Benefit: 让一块 UB workspace 覆盖多个串行阶段，显著降低临时 buffer 总量。
Trade-off: 复用依赖严格阶段顺序，后续若引入阶段重叠则必须重算 zone 划分。

---

## Variant B: 高阶 API 临时 Buffer 与计算 Buffer 共享
Source: SIMD算子性能优化/内存访问/算子与高阶API共享临时Buffer.md

当 `SoftMax` 等高阶 API 需要临时 buffer，而其他计算阶段也需要一块大小接近的 VECCALC 空间时，可按 `max(softmaxBufSize, computeBufSize)` 只分配一块共享 buffer，在两个阶段轮流使用。

**Expert implementation:**
```cpp
// 反例：独立分配，UB 空间浪费
pipe.InitBuffer(softmaxBuf, 1, softmaxBufSize);  // SoftMax 临时空间
pipe.InitBuffer(sumBuf, 1, sumBufSize);          // Add 临时空间

// 正例：共享分配，取 MAX 大小
uint32_t sharedSize = std::max(softmaxBufSize, sumBufSize);
pipe.InitBuffer(sharedBuf, 1, sharedSize);
// SoftMax 阶段使用 sharedBuf
// Add 阶段复用 sharedBuf
```

**vs. baseline (lingxi-code):**
```cpp
pipe.InitBuffer(softmaxTmp, 1, softmaxBufSize);
pipe.InitBuffer(postProcessTmp, 1, sumBufSize);
RunSoftmax(softmaxTmp);
RunPostProcess(postProcessTmp);
```

Benefit: 用一块共享 buffer 覆盖高阶 API 和普通计算阶段的临时空间需求，释放更多 UB 给主数据 tile。
Trade-off: 需要确认 API 内部临时空间在阶段结束后已完全失效，不能与后续计算并发使用。

---

## Variant C: GM Workspace 多阶段结果复用
Source: 【案例总结】DeepSeek V3网络IFA性能优化.md

对于 IFA、MLA 等需要将多个中间结果写入 GM workspace 的场景，可根据 MM1、Softmax、MM2、归约等阶段的生命周期，把不同阶段输出映射到同一段 workspace 区间，整体 workspace 大小取各阶段峰值而非求和。

**Expert implementation:**
```cpp
// Workspace 复用矩阵
// mm1Res 与 softmaxRes 生命周期不重叠 → 可复用
// softmaxRes 与 mm2Res 生命周期不重叠 → 可复用
// mm2Res 与 fdRes 生命周期不重叠 → 可复用

struct WorkspaceLayout {
    uint64_t phase1Start;  // MM1 输出
    uint64_t phase2Start;  // Softmax 输出（复用 phase1）
    uint64_t phase3Start;  // MM2 输出（复用 phase2）
    uint64_t phase4Start;  // FD 归约（复用 phase3）
    uint64_t totalSize;    // = max(phase sizes)
};
```

**vs. baseline (lingxi-code):**
```cpp
WorkspaceLayout layout;
layout.mm1Start = 0;
layout.softmaxStart = mm1ResSize;
layout.mm2Start = mm1ResSize + softmaxResSize;
layout.fdStart = mm1ResSize + softmaxResSize + mm2ResSize;
// 各阶段独立占用 workspace，整体大小为各阶段之和
```

Benefit: 将多阶段 workspace 需求从求和压缩为峰值，特别适合中间结果多、HBM 预算受限的复杂 attention/IFA 场景。
Trade-off: 需要显式维护阶段图和生命周期矩阵，任何新增中间结果都可能迫使整体布局重算。

---

## Variant D: 统一 workspace 编排下的局部 alias-safe 复用
Source: sparse_lightning_indexer_grad_kl_loss_enhance

当 per-core workspace 中存在多个中间张量且部分张量可在严格 fence 和消费顺序约束下共享物理区间时，可在统一 workspace 规划中加入局部 alias。例如 `reluGm` 直接复用 `bmm2Res` 的物理区间，同时其他区域仍保持独立 offset。

**Expert implementation:**
```cpp
workspacePerCoreSize = gatherPSize + gatherSYSize + bmm1Size + bmm2Size +
    reluGradSize + psySyncSize + bmm3Size + scatterAddSize + lossResSize;

gatherPResGm.SetGlobalBuffer(workspace + gatherPOffset);
gatherSYResGm.SetGlobalBuffer(workspace + gatherSYOffset);
bmm1ResGm.SetGlobalBuffer(workspace + bmm1Offset);
bmm2ResGm.SetGlobalBuffer(workspace + bmm2Offset);
reluGm.SetGlobalBuffer((__gm__ RELU_T *)(workspace + bmm2Offset));
psySyncGm.SetGlobalBuffer(workspace + psySyncOffset);
```

**vs. baseline (lingxi-code):**
```cpp
bmm2ResGm.SetGlobalBuffer(workspace + bmm2Offset);
reluGm.SetGlobalBuffer(workspace + reluOffset);
// 每个中间结果单独保留物理区间，workspace 规划更保守
```

Benefit: 在统一 workspace 编排中进一步压缩局部峰值占用，减少碎片化并为额外 ping-pong 或 per-core 切片腾出空间。
Trade-off: alias 安全性高度依赖 fence、消费顺序和跨阶段 handoff，误判会直接造成数据踩踏；这种复用不能泛化为无条件模式。
