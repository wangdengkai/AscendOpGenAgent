# P46: MatmulImpl 高阶矩阵乘 API (MatmulImpl High-Level API)

## Overview
用 `matmul::MatmulImpl` 高阶 API + MDL 配置替代手写 L0/L1 缓冲管理，将 ~160 行手写 matmul 流水线代码压缩到 ~10 行，同时获得框架自动优化的 L1 多缓冲、K 维重排等能力。

## When to Use
- 任何包含 Cube 矩阵乘的算子（Cube+Vector 融合算子）
- 需要 L1 多缓冲预取优化的大 K 场景
- 需要支持多种量化模式（A8W8/A4W4/A16W8）的 matmul
- 手写 matmul 代码量过大、维护困难时

## Trade-off
- 依赖框架 MatmulImpl 版本，API 变更需跟进
- MDL 配置参数（depthA1/depthB1/stepKa/stepKb）需要根据 K 维大小调优
- 部分极端场景（非标准 layout）可能需要回退到手写

**Source operators**: grouped_matmul, ffn, quant_matmul_reduce_sum, grouped_matmul_swiglu_quant_v2, grouped_matmul_finalize_routing

---

## Variant A: 基础 MatmulImpl 替代手写 matmul
Source: grouped_matmul

用 MatmulImpl 模板替代手写的 GM→L1→L0A/L0B→Cube→L0C→GM 流水线。

**Expert implementation:**
```cpp
matmul::MatmulImpl<aT, bT, cT, BiasT, CFG_MDL> mm;
REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), mm, &mmTilingData_);
mm.SetOrgShape(m, n, k);
mm.SetSingleShape(curSingleM, curSingleN, k);
mm.SetTensorA(xGm[xOffset]);
mm.SetTensorB(weightGm[wOffset]);
mm.IterateAll<true>(yGm[yOffset], false);
```

**vs. baseline:**
```cpp
// 手写 ~160 行: LoadNdGmToNzL1, LoadNzL1ToZzL0A, Mmad, FixpipeNzL0cToNdGm
// 静态 L1_PREFETCH=3, 无动态优化, 无 K 维重排
```

Benefit: 代码量减少 70-80%，框架自动管理 L1/L0 缓冲
Trade-off: 依赖框架版本

## Variant B: MDL 硬件量化卸载
Source: grouped_matmul_swiglu_quant_v2

通过 MDL 配置启用硬件级量化融合，SetQuantVector 将反量化 scale 直接注入 Cube 流水线。

```cpp
mm.SetQuantVector(scaleGM[scaleOffset]);  // 硬件级反量化融合
auto cfg = GetMDLConfig();                 // Multi-Data-Load 配置
// CFG_MDL 模式: 自动 L1 预取 + K 维重排
```

Benefit: 反量化零 Vector 开销，完全由 Cube 硬件完成
Trade-off: 仅支持特定量化模式（A8W8/A8W4）

## Variant C: 深度优化参数配置（大 K 场景）
Source: quant_matmul_reduce_sum

针对 K≥1024 的大 K 场景，配置深度 L1 缓冲和 K 维重排加载。

```cpp
// 大 K 优化配置
// depthA1=8, depthB1=8: L1 八缓冲预取
// stepKa=4, stepKb=4: K 维分步加载
// enableKdimReorderLoad=true: DDR 访问优化
mm.SetOrgShape(m, n, k);
mm.SetSingleShape(curSingleM, curSingleN, k);
mm.IterateAll<sync>(output, accumulate_flag);
```

Benefit: L1 预取优化 30-50% UB 利用率，K 维重排减少 20-30% DDR 带宽
Trade-off: 深度缓冲增加 L1 占用

## Variant D: REGIST_MATMUL_OBJ 注册模式
Source: ffn

通过宏注册 matmul 对象，自动绑定 pipe 和 workspace。

```cpp
REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), mm, &mmTilingData_);
// 自动完成: pipe 绑定、workspace 分配、tiling 数据关联
// 支持多个 matmul 对象共享同一 pipe
```

Benefit: 简化初始化流程，支持多 matmul 对象管理
Trade-off: 宏展开增加编译时间
