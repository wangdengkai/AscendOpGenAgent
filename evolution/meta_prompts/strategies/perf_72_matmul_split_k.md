# P72 Matmul 多核切 K (Matmul Multi-Core Split-K)
## Overview
当矩阵 M 和 N 较小无法有效切分多核时，可沿 K 轴切分实现多核并行。各核计算部分 K 的矩阵乘结果，最终通过 AtomicAdd 在 GM 上累加合并。需在首次写入 GM 前对 C 矩阵清零。

## When to Use
- M 和 N 较小，切分 M/N 轴无法充分利用多核
- K 轴较大（如 K=1024，M=N=16）
- M/N/K 均较大时，切 K 可更好平衡输入输出带宽

## Trade-off
- 需要 GM 清零 + AtomicAdd，有额外开销
- 不支持 Bias 参与矩阵乘
- 仅支持输出到 GM
- AtomicAdd 可能引入精度影响

**Source operators**: 优秀实践/Matmul性能调优案例

---
## Variant A: EnableMultiCoreSplitK + AtomicAdd
Source: 优秀实践/Matmul性能调优案例/Matmul高阶API使能多核切K.md

Tiling 侧使能多核切 K，Kernel 侧清零 GM 后开启 AtomicAdd 累加。

**Expert implementation:**
```cpp
// Tiling 侧：使能多核切 K
cubeTiling.SetOrgShape(M, N, K);
cubeTiling.SetShape(M, N, K);
cubeTiling.EnableMultiCoreSplitK(true);
cubeTiling.GetTiling(tilingData);

// Kernel 侧：GM 清零 + AtomicAdd
cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType*>(c), tiling.M * tiling.N);
Fill(cGlobal, tiling.M * tiling.N, (cType)0);  // GM 清零

uint8_t enAtomic = 1;
matmulObj.IterateAll(cGlobal, enAtomic);  // AtomicAdd 累加
```

Benefit: aic_time 从 19.60us 降至 13.70us（单核→双核并行，提升 30%）
Trade-off: 需 GM 清零 + AtomicAdd 开销，不支持 Bias
