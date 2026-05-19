# P71 Matmul IBShare L1 共享矩阵 (Matmul IBShare L1 Sharing)
## Overview
MIX 场景下多个 AIV 核使用相同的 A 或 B 矩阵数据时，默认每个 AIV 都独立从 GM 搬运到 L1。使能 IBShare 功能后，第一个 AIV 搬入的矩阵数据缓存在 L1 Buffer 上供其他 AIV 复用，避免重复 MTE2 搬运，减少搬运开销。

## When to Use
- MIX 场景（AIC+AIV 混合编程）
- 多个 AIV 的 A 或 B 矩阵 GM 地址相同
- 共享矩阵在 L1 Buffer 上可全载（singleCoreN=baseN*stepN, singleCoreK=baseK*stepKb）
- Profiling 显示 MTE2 搬运中存在重复数据搬运

## Trade-off
- 仅适用于 MIX 场景
- A 和 B 同时使能 IBShare 时只支持 IterateAll 输出到 GM
- 共享矩阵必须在 L1 上全载

**Source operators**: 优秀实践/Matmul性能调优案例

---
## Variant A: B 矩阵 IBShare 共享
Source: 优秀实践/Matmul性能调优案例/Matmul高阶API使能IBShare模板共享B矩阵数据.md

多个 AIV 共享 L1 上的 B 矩阵数据，避免 B 矩阵的重复 MTE2 搬运。

**Expert implementation:**
```cpp
// 设置 B 矩阵的 IBSHARE 参数为 true
using B_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND,
    BType, false, LayoutMode::NONE, true>;  // 最后一个参数 true 使能 IBShare

AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_IBSHARE_NORM> matmulObj;
```

Benefit: MTE2 搬运次数减半（AIV1 从 12 次降至 6 次），aic_mte2_time 从 5.56us 降至 4.71us（提升 15.46%）
Trade-off: 需要共享矩阵在 L1 上全载
