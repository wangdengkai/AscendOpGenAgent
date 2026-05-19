# P28: 自定义 HardEvent 生产者-消费者同步

## Overview
用编译期确定的 HardEvent 类型（MTE2_MTE1, MTE1_M, M_FIX 等）建立精确的生产者-消费者关系，替代粗粒度 PipeBarrier。每个 Buffer 绑定一对事件 ID。

## When to Use
- 多级流水线（≥3 级）的 Cube/Vector 算子，PipeBarrier 粒度过粗导致流水气泡
- 需要精确控制 MTE2→L1、L1→L0、L0→Cube、Cube→Vector 等特定级间同步，而非全流水线屏障
- 简单的两级流水（如仅 MTE2+Vector）使用 PipeBarrier 即可，无需引入 HardEvent 复杂度

## Trade-off
- 事件 ID 资源有限（16 个），需轮转复用；Init/UnInit 必须严格配对
- 每种 BufferType 需要独立的 HardEvent 特化，buffer 类型多时模板代码膨胀

**Source operators**: common/buffer.h

---

## Variant A: 编译期 HardEvent 类型映射 + Buffer 绑定
Source: common/buffer.h

通过 constexpr 函数在编译期确定每种 BufferType 对应的 HardEvent，消费者 Wait 生产者 Set，实现精确的流水线级间同步。

```cpp
template<BufferType Type>
struct BufferInfo {
    __aicore__ const static constexpr HardEvent ConsWaitProdStatus() {
        if constexpr (Type == BufferType::L1)  return HardEvent::MTE2_MTE1;
        if constexpr (Type == BufferType::L0A) return HardEvent::MTE1_M;
        if constexpr (Type == BufferType::L0C) return HardEvent::M_FIX;
    }
    static constexpr HardEvent EventP2C = ConsWaitProdStatus();
};
buffer_.Wait<EventP2C>();
buffer_.Set<EventP2C>();

// 更多常见 TPosition 类型映射
template<> struct BufferInfo<TPosition::L0B> {
    static constexpr HardEvent WAIT = HardEvent::MTE2_MTE1;
    static constexpr HardEvent SET  = HardEvent::MTE1_M;
};
template<> struct BufferInfo<TPosition::VECIN> {
    static constexpr HardEvent WAIT = HardEvent::MTE2_V;
    static constexpr HardEvent SET  = HardEvent::V_MTE3;
};
template<> struct BufferInfo<TPosition::VECOUT> {
    static constexpr HardEvent WAIT = HardEvent::V_MTE3;
    static constexpr HardEvent SET  = HardEvent::MTE3_V;
};
```

Benefit: 编译期确定事件类型，零运行时开销；替代粗粒度 PipeBarrier 减少流水线气泡
Trade-off: 事件 ID 资源有限（16 个），需轮转复用；Init/UnInit 必须严格配对
