# P73 CV 并行 AIC:AIV 比例与多 Workspace 流水 (CV Parallel AIC:AIV Ratio & Multi-Workspace Pipeline)
## Overview
GroupedMatmul 等 Cube+Vector 融合算子中，当 Vector 计算为主要瓶颈时，可将 AIC:AIV 启动比例从默认 1:1 调整为 1:2，让更多 AIV 核分担 Vector 计算。同时，Cube 与 Vector 之间通过 workspace 传递数据存在依赖等待；默认 2 份 workspace 的 pingpong 方案仍有互等间隙，扩展到 4 份 workspace 可进一步消除 Cube-Vector 流水气泡。最后，Vector 侧开启 double buffer 进一步提升搬运与计算重叠度。

## When to Use
- Cube+Vector 融合算子（如 GroupedMatmul + 后处理量化）
- Profiling 显示 Vector Bound（Vector 计算耗时远大于 Cube）
- Cube 与 Vector 之间通过 GM workspace 传递中间结果
- 2 份 workspace pingpong 方案下 Cube/Vector 仍有明显互等间隙
- MIX 场景（AIC+AIV 混合编程）

## Trade-off
- 1:2 AIC:AIV 比例减少了可用 AIC 核数，Cube 吞吐降低
- 4 份 workspace 需要 4 倍 GM workspace 空间
- 仅适用于 MIX 场景的 Cube+Vector 融合算子
- Vector double buffer 需要额外 UB 空间

**Source operators**: 优秀实践/GroupedMatmul算子性能调优案例

---
## Variant A: AIC:AIV 1:2 比例缓解 Vector Bound
Source: 优秀实践/GroupedMatmul算子性能调优案例.md

Vector 计算为瓶颈时，将 AIC:AIV 比例设为 1:2，让 2 个 AIV 核分担 Vector 后处理。

**Expert implementation:**
```cpp
// Tiling 侧：设置 AIC:AIV 比例为 1:2
// 通过 SetAicAivRatio 或在 kernel launch 配置中设置 mixMode
// 使得每个 AIC 核对应 2 个 AIV 核处理 Vector 后处理
optiling::MatmulConfigParams matmulConfigParams;
matmulConfigParams.aicAivRatio = {1, 2};  // AIC:AIV = 1:2
cubeTiling.SetMatmulConfigParams(matmulConfigParams);

// Kernel 侧：AIV 根据 subBlockIdx 分配工作
uint32_t aivIdx = GetSubBlockIdx();  // 0 或 1
uint32_t aivNum = GetSubBlockNum();  // 2
// 按 aivIdx 将 Vector 后处理工作量均分到 2 个 AIV 核
uint32_t vecStart = aivIdx * totalVecWork / aivNum;
uint32_t vecEnd = (aivIdx + 1) * totalVecWork / aivNum;
```

Benefit: 总耗时从 218.1us 降至 154.2us（提升 29.3%），Cube 计算间等待明显减小
Trade-off: AIC 核数减半，Cube 吞吐降低

---
## Variant B: 4 份 Workspace 消除 Cube-Vector 互等
Source: 优秀实践/GroupedMatmul算子性能调优案例.md

默认 2 份 workspace pingpong 下 Cube 需等 Vector 释放 workspace，扩展到 4 份后 Cube 可连续写入不同 workspace 而无需等待。

**Expert implementation:**
```cpp
// 分配 4 份 workspace 空间
constexpr int32_t WORKSPACE_NUM = 4;
__gm__ uint8_t* workspace[WORKSPACE_NUM];
for (int32_t i = 0; i < WORKSPACE_NUM; i++) {
    workspaceOffset = i * singleWorkspaceSize;
    workspace[i] = workspaceGm + workspaceOffset;
}

// Cube 侧：轮转写入 4 份 workspace
int32_t wsIdx = cubeCount % WORKSPACE_NUM;
matmulObj.IterateAll(workspace[wsIdx], ...);

// Vector 侧：按序读取对应 workspace
int32_t wsIdx = vecCount % WORKSPACE_NUM;
// 从 workspace[wsIdx] 搬入数据进行 Vector 后处理
```

Benefit: 总耗时从 154.2us 降至 131.8us（提升 14.5%），Cube/Vector 间隙明显减小
Trade-off: 需要 4 倍 GM workspace 空间

---
## Variant C: Vector Double Buffer 提升搬运计算重叠
Source: 优秀实践/GroupedMatmul算子性能调优案例.md

在 Variant A+B 基础上，Vector 侧 InitBuffer 指定 buffer 数为 2，实现搬运与计算重叠。

**Expert implementation:**
```cpp
// Vector 侧 InitBuffer 指定 double buffer
pipe->InitBuffer(scaleInQueue, 2, tiling->mmTilingData.baseN * sizeof(DTYPE_SCALE));
pipe->InitBuffer(perTokenScaleInQueue, 2, tiling->mmTilingData.baseM * sizeof(float));
pipe->InitBuffer(vecInQueue, 2, tiling->ubCalSize * sizeof(cT::T));
pipe->InitBuffer(vecOutQueue, 2, tiling->ubCalSize * sizeof(DTYPE_Y));
```

Benefit: 总耗时从 131.8us 降至 128.1us（提升 2.8%），三层优化累计从 218.1us 降至 128.1us（提升 41.3%）
Trade-off: 需要额外 UB 空间用于 double buffer。本 Variant 为 P1（Double Buffering）在 MIX 场景 Vector 侧的特化应用，参见 P1 了解通用双缓冲机制。CV 并行块大小选择参见 P16
