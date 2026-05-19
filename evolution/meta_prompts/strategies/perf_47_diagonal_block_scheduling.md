# P47: 对角线块调度 (Diagonal Block Scheduling)

## Overview
用 MNBlockIdxCompute 对角线分配替代线性行优先分块，优化多核场景下的 L2 cache 复用和负载均衡。当 blockDimM > 5 时启用对角线映射，小 shape 自动回退到线性调度。

## When to Use
- 多核 M×N 分块的 Cube 矩阵乘场景
- blockDimM > 5（足够多的 M 维分块才有对角线收益）
- 多核竞争同一列权重数据导致 L2 cache 抖动
- M 维分布不均匀导致负载不均衡

## Trade-off
- 对角线映射增加少量地址计算开销（LCM 计算）
- 小 shape（blockDimM ≤ 5）无收益，需 threshold 控制
- 与 P51（动态核配比）互补：P47 优化块分配，P51 优化核分配

**Source operators**: grouped_matmul, quant_matmul_reduce_sum, grouped_matmul_finalize_routing

---

## Variant A: 8×8 网格对角线调度
Source: grouped_matmul

thresholdBlockNum=8 的对角线映射，在 8×8 网格内交错分配 M/N 维块。

**Expert implementation:**
```cpp
void MNBlockIdxCompute(uint32_t curBlock, ...) {
    if (blockDimM > thresholdDimM) {  // thresholdDimM = 5
        uint32_t thresholdBlockNum = 8;
        uint32_t curThresholdM = min(blockDimM, thresholdBlockNum);
        uint32_t curThresholdN = min(blockDimN, thresholdBlockNum);
        uint32_t thresholdM_dimN = curThresholdM * blockDimN;
        uint32_t curThresholdM_thresholdN = curThresholdM * curThresholdN;
        uint32_t localRelativeBlock = relativeBlock % thresholdM_dimN % curThresholdM_thresholdN;
        mIdx = localRelativeBlock % curThresholdM + relativeBlock / thresholdM_dimN * thresholdBlockNum;
        nIdx = (localRelativeBlock + localRelativeBlock / LeastCommonMultiple(curThresholdM, curThresholdN))
               % curThresholdN + relativeBlock % thresholdM_dimN / curThresholdM_thresholdN * curThresholdN;
    } else {
        // 线性回退
        mIdx = (curBlock - count) / blockDimN;
        nIdx = (curBlock - count) % blockDimN;
    }
}
```

**vs. baseline (线性调度):**
```cpp
mIdx = (curBlock - count) / blockDimN;
nIdx = (curBlock - count) % blockDimN;
// 同列块被不同核加载 → L2 cache 抖动
```

Benefit: L2 cache 命中率提升 10-20%，负载均衡改善 10-30%
Trade-off: 地址计算增加 ~10 条指令

## Variant B: LeastCommonMultiple 错位步长
Source: quant_matmul_reduce_sum

使用最小公倍数计算 N 维错位步长，确保对角线映射在非方阵网格中均匀分布。

```cpp
uint32_t lcm = LeastCommonMultiple(curThresholdM, curThresholdN);
nIdx = (localRelativeBlock + localRelativeBlock / lcm) % curThresholdN + nOffset;
```

Benefit: 非方阵网格（M≠N）下仍保持均匀的 L2 访问模式
Trade-off: LCM 计算增加少量标量开销

## Variant C: 与 parallNum 流水线结合
Source: grouped_matmul_finalize_routing

对角线调度与 parallNum（流水线并行度）结合，在多 group matmul 场景下同时优化组间和组内调度。

```cpp
// parallNum 控制组间流水线深度
// MNBlockIdxCompute 控制组内块分配
// 两者正交组合
```

Benefit: 组间流水 + 组内对角线，双层优化
Trade-off: 调度逻辑复杂度增加
