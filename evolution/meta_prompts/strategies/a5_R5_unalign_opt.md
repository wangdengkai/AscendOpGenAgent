# R5: 非对齐访问优化

## Overview
使用 UnalignRegForLoad/Store 优化非 32B 对齐地址的数据访问。

## When to Use
- 数据地址非 32B 对齐（如 stride 访问、非对齐 offset）
- 当前使用多次对齐访问 + 拼接来处理非对齐数据

## Trade-off
- **收益**: 减少非对齐场景的访问次数
- **风险**: UnalignReg 资源有限（每 VF 最多 4 Load + 4 Store）
- **复杂度**: 需要 Pre/Post 配对调用

## Variant A: 非对齐加载

```cpp
// Baseline: 对齐加载 + 手动拼接
// (需要两次 LoadAlign + Shift 操作)

// Evolved: 使用 UnalignReg
UnalignRegForLoad ureg;
LoadUnAlignPre<float>(ureg, srcAddr);  // 预初始化（循环外）

for (uint16_t i = 0; i < repTimes; ++i) {
    LoadUnAlign<float, POST_MODE_UPDATE>(srcReg, ureg, srcAddr, stride);
    // srcAddr 自动更新
    Adds(dstReg, srcReg, scalar, mask);
    StoreAlign(dstAddr + i * repSize, dstReg, mask);
}
```

## Variant B: 非对齐存储

```cpp
UnalignRegForStore ureg;

for (uint16_t i = 0; i < repTimes; ++i) {
    LoadAlign(srcReg, srcAddr + i * repSize);
    Adds(dstReg, srcReg, scalar, mask);
    StoreUnAlign<float, POST_MODE_UPDATE>(dstAddr, dstReg, ureg, stride);
}
StoreUnAlignPost<float, POST_MODE_UPDATE>(dstAddr, ureg, stride);  // 必须调用
```

**注意**: StoreUnAlignPost 必须在循环结束后调用，否则最后一块数据可能丢失。
