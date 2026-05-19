# R8: Mutex 细粒度同步

## Overview
使用 Mutex 替代粗粒度 SetFlag/WaitFlag，减少流水线停顿。

## When to Use
- SetFlag/WaitFlag 导致过多流水线停顿
- 需要比 pipe_barrier 更细粒度的同步控制
- 多个异步流水之间有部分依赖（非全依赖）

## Trade-off
- **收益**: 减少不必要的流水线等待
- **风险**: 过细粒度的同步增加编程复杂度
- **复杂度**: 高

## Variant A: 替代粗粒度 SetFlag/WaitFlag

```cpp
// Baseline: 粗粒度同步（等待整个流水完成）
SetFlag<HardEvent::MTE2_V>(eventId);
WaitFlag<HardEvent::MTE2_V>(eventId);
// 所有 MTE2 操作都完成后才继续

// Evolved: Mutex 细粒度同步
Mutex::Lock(PIPE_MTE2);
// 仅锁定 MTE2 流水
DataCopy(ubTensor, gmTensor, count);
Mutex::Unlock(PIPE_MTE2);
// MTE2 解锁后其他流水可继续
```

## Variant B: VF 内部用 LocalMemBar

VF 函数内部不使用 Mutex，而是用 LocalMemBar：

```cpp
// VF 内部: 精确指定依赖方向
StoreAlign(addr, reg, mask);
LocalMemBar<VEC_STORE, VEC_LOAD>();  // 仅等待 Store→Load
LoadAlign(reg2, addr);
```

## 使用建议

| 场景 | 推荐机制 |
|------|---------|
| VF 内部同步 | LocalMemBar（编译器自动优化） |
| VF 外部、简单依赖 | SetFlag/WaitFlag（保持兼容） |
| VF 外部、复杂依赖 | Mutex（细粒度控制） |
| 核间同步 | CrossCore 增强模式 |
