# P35: TBuf 常驻中间累加器

## Overview
使用 TBuf<VECCALC> 分配的 buffer 不参与队列管理，在 Init 中分配后在整个 Process 生命周期内常驻，用于存储跨迭代的递推状态或梯度累加器。

## When to Use
- 递推类算子（conv1d）、需要跨迭代累加的梯度计算
- 状态数据在多次 loop 迭代中持续更新，不能每次重新分配和初始化

## Trade-off
- TBuf 不受队列同步保护，需要程序员自行通过 PipeBarrier 保证一致性
- 常驻 buffer 在整个 Process 期间占用 UB，减少可用于数据 tile 的空间

**Source operators**: ai_infra_causal_conv1d_add, ai_infra_sinkhorn_grad

---

## Variant A: TBuf<VECCALC> 常驻递推状态 buffer
Source: ai_infra_causal_conv1d_add, ai_infra_sinkhorn_grad

使用 TBuf<TPosition::VECCALC> 声明常驻 buffer，在递推计算中持续更新 y0/y1/y2 状态。

```cpp
TBuf<TPosition::VECCALC> y0Buf_;
TBuf<TPosition::VECCALC> y1Buf_;
TBuf<TPosition::VECCALC> y2Buf_;

void ComputeAndUpdate() {
    LocalTensor<float> y0BufLocal = y0Buf_.Get<float>();
    LocalTensor<float> y1BufLocal = y1Buf_.Get<float>();
    Mul(y2BufLocal, x32BufLocal, weight2, curDim_);
    Add(y2BufLocal, y1BufLocal, y2BufLocal, curDim_);
    Mul(y1BufLocal, x32BufLocal, weight1, curDim_);
    Add(y1BufLocal, y0BufLocal, y1BufLocal, curDim_);
    Mul(y0BufLocal, x32BufLocal, weight0, curDim_);
}
```

Benefit: 常驻 buffer 避免每次迭代重新分配，支持跨迭代状态传递
Trade-off: TBuf 不受队列同步保护，需手动插入 PipeBarrier 保证一致性
