# P44: Transpose Buffer 常驻复用
## Overview
gradTransposeBuf_ 作为转置后的梯度数据常驻 UB，在整个 numIters 迭代循环中被 colNormGrad/rowNormGrad 反复原地读写，只在首尾各做一次转置和 GM 搬运。

## When to Use
- 迭代式算法（如 Sinkhorn）中梯度数据需要多次原地更新
- 数据在迭代过程中需要行列交替访问（colNormGrad/rowNormGrad），转置操作仅首尾各一次

## Trade-off
- 占用 tAlign_×n_×n_×4 字节，n 较大时成为 tiling 瓶颈
- TransposeXIn/Out 本身有开销，仅在迭代次数较多（≥3）时收益明显

**Source operators**: ai_infra_sinkhorn_grad

---

## Variant A: 迭代式梯度常驻 UB 原地更新
Source: ai_infra_sinkhorn_grad

分配 VECCALC buffer 常驻 UB，首次转置后在整个迭代循环中原地读写，仅在最后一次迭代后转置回并搬出。

**Expert implementation:**
```cpp
TBuf<TPosition::VECCALC> gradTransposeBuf_;
void Process() {
    TransposeXIn();
    for (int j = numIters_ - 1; j > 0; --j) {
        colNormGrad();
        rowNormGrad();
    }
    TransposeXOut();
    CopyOut(offset);
}
```

Benefit: 迭代过程中零 DMA 开销，所有计算在片上完成
Trade-off: buffer 大小随 n² 增长，n 较大时成为 tiling 瓶颈
