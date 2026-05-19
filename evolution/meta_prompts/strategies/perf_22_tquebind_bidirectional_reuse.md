# P22: TQueBind 双向复用队列
## Overview
对于既需要从 GM 搬入又需要搬出到 GM 的数据（如 conv_states），使用 TQueBind 将 VECIN 和 VECOUT 绑定到同一块物理 buffer，节省约 50% 的 buffer 占用。适用于"读-改-写回"的 in-place 操作场景。

## When to Use
- "读-改-写回"场景，如 conv_states 更新、in-place 操作
- buffer 资源紧张，需要节省片上存储的场景

## Trade-off
- depth 通常只能为 1，无法做双缓冲流水；搬入搬出之间必须显式同步
- 不适用于输入输出 shape 不同的场景

**Source operators**: ai_infra_causal_conv1d_add

---

## Variant A: TQueBind 绑定 VECIN/VECOUT 实现 buffer 复用
Source: ai_infra_causal_conv1d_add

将 VECIN 和 VECOUT 绑定到同一块物理 buffer，同一个 tensor 先从 GM 搬入、就地修改后直接搬出到 GM，省去额外的输出 buffer 分配。搬入搬出之间通过 HardEvent 显式同步。

**Expert implementation:**
```cpp
TQueBind<TPosition::VECIN, TPosition::VECOUT, NUM_ONE> xQueue_;
TQueBind<TPosition::VECIN, TPosition::VECOUT, NUM_ONE> convStatesQueue_;

x16Local_ = xQueue_.AllocTensor<DTYPE>();
DataCopyPad(x16Local_, this->xGm[xGmOffset], copyParams, padParams);
SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
xQueue_.EnQue(x16Local_);
x16Local_ = xQueue_.DeQue<DTYPE>();
// 同一个 tensor 直接搬出到 GM
DataCopyPad(this->convStateGm[offset], x16Local_, copyParams);
xQueue_.FreeTensor(x16Local_);
```

Benefit: 节省约 50% 的 buffer 占用，适合 UB 资源紧张的场景
Trade-off: depth 只能为 1，无法做双缓冲流水；搬入搬出之间必须显式同步
