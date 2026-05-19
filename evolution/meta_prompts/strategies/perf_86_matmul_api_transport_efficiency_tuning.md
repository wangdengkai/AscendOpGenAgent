# P86 Matmul Internal Data-Path Efficiency Tuning (Matmul 高阶 API 内部数据通路效率调优)
## Overview
该策略统一覆盖四类同源优化：选择更合适的调度模板（MDL/NBuffer33/Preload）、放大 base block 提升计算访存比、使能 UnitFlag 细粒度 MMAD-FIXPIPE 同步，以及开启 K 轴错峰访问缓解多核 GM 冲突。四者都作用于 Matmul 库内部的 GM→L1→L0→Cube 执行路径，目标是一致的：减少 MTE2 次数、缩短搬运空泡、提升 Cube 与搬运/搬出的重叠效率。它不包含手动 Mmad 流水、不包含跨核共享 L1 数据，也不改变分核拓扑。

## When to Use
- 使用 AscendC Matmul 高阶 API，且 profiling 显示主要瓶颈在 MTE2 搬运、MMAD/FIXPIPE 串行或多核 K 轴冲突
- 大 shape Matmul 中默认模板或默认 base block 无法充分利用带宽与 Cube 计算能力
- 希望通过 Matmul 库现有配置项优化性能，而不是改写为手动 Mmad 实现
- 多核 Matmul 的 K 轴较大、GM→L1 带宽利用率低，或 MTE2 耗时远高于 Cube 耗时
- 可接受模板、base block 和同步模式都依赖 shape/profiling 进行调参
- 若需要共享 L1 上的矩阵给多个 AIV，应优先参考 P57
- 若需要改变多核并行拓扑（如切 K 轴）并使用 AtomicAdd 归约，应优先参考 P58
- 若需要手动控制 L0A/L0B/L0C 与 Mmad/Fixpipe 时序，应优先参考 P73/P81
- 若需要通用 GM 地址冲突规避（非 Matmul K 轴专用），应优先参考 P50
- 若需要把输入矩阵预加载到 TSCM/L1 再让 Matmul 直接读取，应优先参考 P64

## Trade-off
- 不同模板、base block 和开关之间存在前置条件与互斥关系，错误组合可能无收益甚至退化
- 大 block、MDL、大包搬运或 preload 往往依赖更大的 L1/L0 预算，小 shape 容易被头开销抵消
- UnitFlag、Kdim reorder 等配置受模板能力约束，并非所有 Matmul 场景都支持
- 该策略高度依赖 profiling 结果；若真实瓶颈在分核、片上缓存全载或跨核共享，需切换到其他策略而非继续调 MatmulConfig

**Source operators**: 优秀实践/Matmul性能调优案例

---

## Variant A: 调度模板选择（MDL / NBuffer33 / Preload）
Source: 优秀实践/Matmul性能调优案例/Matmul高阶API使能MDL模板.md, 优秀实践/Matmul性能调优案例/Matmul高阶API使能NBuffer33模板.md, 优秀实践/Matmul性能调优案例/Matmul高阶API使能MTE2_Preload.md

当默认 Norm 模板的 MTE2 次数、间隙或搬运并行度不足时，可根据 profiling 信号选择更合适的调度模板：MDL 用大包搬运减少 MTE2 次数，NBuffer33 用错开搬运平衡计算与带宽，Preload 在间隙中预取下一轮数据。三者都属于 Matmul 库内部的调度模板切换。

**Expert implementation:**
```cpp
// MDL：一次搬多个基本块
AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL> matmulMdl;

// NBuffer33：基于 MDL 的 3x3 错开搬运
matmul_tiling::MatmulConfigParams matmulConfigParams(
    1, false, matmul_tiling::ScheduleType::N_BUFFER_33, ...);
cubeTiling.SetMatmulConfigParams(matmulConfigParams);

// Preload：在 MTE2 间隙预取下一轮
static constexpr MatmulConfig MM_CFG = GetMDLConfig(false, false, preloadMode);
AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG> matmulPreload;
```

**vs. baseline (lingxi-code):**
```cpp
// 基线：默认 Norm 模板
AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM> matmulObj;
```

Benefit: 根据 bottleneck 选择最优调度模板，减少 MTE2 次数、缩短间隙并提升搬运与计算重叠度。
Trade-off: 模板互斥且前提不同；MDL 头开销大，NBuffer33 依赖 MDL 且只适合纯 Cube，Preload 依赖 K 全载和对应方向 DoubleBuffer。

---

## Variant B: 基本块计算访存比优化
Source: 优秀实践/Matmul性能调优案例/Matmul算子优化Tiling策略.md

当 Matmul MTE2 Bound 且默认 `[baseM, baseN, baseK]` 组合导致计算密度低、搬出地址不对齐或总搬运量过高时，可放大 base block，使 Cube 固定计算量对应更少搬运数据，从而提高计算访存比。该 Variant 聚焦 block 参数本身，不再重复讲 MDL 模板细节。

**Expert implementation:**
```cpp
// 反例：小基本块，计算访存比低
int32_t baseM = 64;
int32_t baseN = 64;
tilingApi.SetFixSplit(baseM, baseN, -1);

// 正例：更大的基本块，提升计算访存比并改善对齐
int32_t baseM = 128;
int32_t baseN = 256;
tilingApi.SetFixSplit(baseM, baseN, -1);
// baseK 由 tiling 自动推导
```

**vs. baseline (lingxi-code):**
```cpp
// 基线：沿用默认或保守的 base block
int32_t baseM = 64;
int32_t baseN = 64;
tilingApi.SetFixSplit(baseM, baseN, -1);
```

Benefit: 通过更高的计算访存比减少单位计算量对应的搬运压力，为后续模板优化创造更好的带宽基础。
Trade-off: 更大基本块需要更高的片上资源预算，仅在 shape 足够大时收益稳定。

---

## Variant C: UnitFlag 细粒度 MMAD-FIXPIPE 同步
Source: 优秀实践/Matmul性能调优案例/Matmul高阶API使能UnitFlag.md

若 profiling 显示 MMAD 结束后 FIXPIPE 才启动，二者在 Matmul 库内部形成串行等待，可通过 `enUnitFlag=true` 把同步粒度从整条 MMAD 指令收缩到 512B 数据块，让 FIXPIPE 更早启动并与 MMAD 并行。

**Expert implementation:**
```cpp
__aicore__ inline constexpr MatmulConfig GetCustomMDLCFG() {
    auto mmCfg = CFG_MDL;
    mmCfg.enUnitFlag = true;
    return mmCfg;
}
constexpr static MatmulConfig CUSTOM_CFG_MDL = GetCustomMDLCFG();

AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CUSTOM_CFG_MDL> matmulObj;
```

**vs. baseline (lingxi-code):**
```cpp
// 基线：默认关闭 UnitFlag，FIXPIPE 等待整条 MMAD 完成
AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL> matmulObj;
```

Benefit: 提前触发 FIXPIPE，减少 MMAD/FIXPIPE 串行等待，提升 Cube 后处理流水重叠。
Trade-off: 仅支持 Norm/IBShare/MDL 模板；它是 Matmul 库内部开关，不适用于 P73 那类手动 Mmad unitFlag 时序控制。

---

## Variant D: K 轴错峰访问缓解多核 GM 冲突
Source: 优秀实践/Matmul性能调优案例/Matmul高阶API使能多核K轴错峰访问内存.md

当多核执行 Matmul 且各核会在同一时刻访问相同 GM K 轴地址时，可在 MDL 模板下使能 `enableKdimReorderLoad`，让各核从不同 K 偏移起始位置搬运，减少地址冲突导致的 GM→L1 带宽下降。

**Expert implementation:**
```cpp
constexpr MatmulConfig GetMDLKDimReorderConfig() {
    auto CFG = CFG_MDL;
    CFG.enableKdimReorderLoad = true;
    return CFG;
}
constexpr static MatmulConfig MM_CFG = GetMDLKDimReorderConfig();

AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG> matmulObj;
```

**vs. baseline (lingxi-code):**
```cpp
// 基线：各核从相同 K 轴起点开始搬运，易出现 GM 地址冲突
AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL> matmulObj;
```

Benefit: 在多核大 K 场景下缓解 GM 地址热点，提升 GM→L1 带宽利用率并降低 MTE2 耗时。
Trade-off: 仅支持 MDL 模板且要求 K 轴非全载；若瓶颈是通用 Vector/搬运冲突而非 Matmul K 轴冲突，应参考 P50 而不是本策略。
