# P30: PipeBarrier 密集插入的纯 Vector 同步

## Overview
在递推类算子（y0→y1→y2 依赖链）中，每条 Vector 指令后插入 PipeBarrier<PIPE_V>，确保前一条指令结果写入 UB 后才被下一条读取。

## When to Use
- 递推类算子，Vector 指令之间存在严格数据依赖（如 y1=f(y0), y2=f(y1)）
- 不适用于无依赖的并行 Vector 计算，密集 barrier 会严重降低吞吐

## Trade-off
- 密集 barrier 阻止 Vector 流水线内部的指令级并行
- 每条 Vector 指令后插入 PipeBarrier 会增加约 1-2 个周期的同步开销

**Source operators**: ai_infra_causal_conv1d_add, ai_infra_sinkhorn_grad

---

## Variant A: 递推依赖链中密集 PipeBarrier<PIPE_V>
Source: ai_infra_causal_conv1d_add

每条 Mul/Add 指令后紧跟 PipeBarrier<PIPE_V>，保证 y2 依赖 y1、y1 依赖 y0 的递推链严格按序执行。

```cpp
Mul(y2BufLocal, x32BufLocal, weight2, curDim_);
PipeBarrier<PIPE_V>();
Add(y2BufLocal, y1BufLocal, y2BufLocal, curDim_);
PipeBarrier<PIPE_V>();
Mul(y1BufLocal, x32BufLocal, weight1, curDim_);
PipeBarrier<PIPE_V>();
Add(y1BufLocal, y0BufLocal, y1BufLocal, curDim_);
PipeBarrier<PIPE_V>();
```

Benefit: 保证递推依赖链的正确性，避免乱序执行导致的数据错误
Trade-off: 密集 barrier 阻止 Vector 流水线内部的指令级并行
