# P29: 跨核同步（AIC-AIV CrossCore Sync）

## Overview
在 AIC（Cube 核）和 AIV（Vector 核）之间传递数据时，通过 CrossCoreSetFlag/CrossCoreWaitFlag 实现跨核同步。一个 AIC 对应两个 AIV，需要分别同步。

## When to Use
- AIC/AIV 混合核架构下的 Cube-Vector 数据交接
- 一个 AIC 对应两个 AIV（AIV0/AIV1），需要分别同步（id0_ 和 id0_+AIV0_AIV1_OFFSET）
- 注意: `AIV0_AIV1_OFFSET` 为平台相关常量，通常定义在 common/buffer.h 或硬件头文件中，表示 AIV0 与 AIV1 的事件 ID 偏移量

## Trade-off
- AIC 对应两个 AIV 需要两次 SetFlag/WaitFlag；跨核事件 ID 空间有限
- 跨核同步延迟高于核内同步，应尽量减少同步次数（批量数据交接优于逐 tile 交接）

**Source operators**: common/buffer.h, sparse_flash_attention_enhance

---

## Variant A: AIC-AIV 跨核 CrossCoreSetFlag/WaitFlag 同步
Source: common/buffer.h

AIC 侧等待两个 AIV 的 MTE1 完成信号（id0_ 和 id0_+offset），AIV 侧等待 AIC 的 MTE3 完成信号，通过 CROSS_CORE_SYNC_MODE 区分跨核事件。

```cpp
if constexpr (bufferType == BufferType::L1) {
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(id0_);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(id0_ + AIV0_AIV1_OFFSET);
    } else {
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(id1_);
    }
}
```

Benefit: 实现 AIC/AIV 异构核间精确数据交接，支持 Cube-Vector 融合流水线
Trade-off: AIC 对应两个 AIV 需要两次 SetFlag/WaitFlag；跨核事件 ID 空间有限
