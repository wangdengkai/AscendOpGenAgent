# P65 UB Bank 冲突规避 (UB Bank Conflict Avoidance)
## Overview
UB（192KB）划分为 48 个 bank（16 个 bank group，每组 3 个 bank），每 bank 4KB。同一 bank group 的并发读写会导致 bank 冲突，使单 Repeat 从 1 拍退化到 8 拍。通过优化计算逻辑或地址分配规避冲突。

## When to Use
- Vector 计算密集型算子
- 多 src 操作数场景（双 src 在同一 bank group）
- blk_stride 为 16 倍数场景（8 个 DataBlock 全在同一 bank group）
- Profiling 显示 Vector 计算时间异常高

## Trade-off
- 地址优化方案需多申请 UB 空间（如 256 字节 padding）
- 计算逻辑优化可能增加代码复杂度
- 需要理解 UB bank 结构

**Source operators**: SIMD算子性能优化/内存访问

---
## Variant A: 优化计算逻辑（连续读跳写）
Source: SIMD算子性能优化/内存访问/避免Unified_Buffer的bank冲突.md

改跳读连续写为连续读跳写，避免写写冲突。

**Expert implementation:**
```cpp
// 反例：跳读连续写，blk_stride=16 导致 8 个 DataBlock 在同一 bank group
Adds(dst, src, scalar, MASK_PLACEHOLDER, 1, {1, 16, 1, 16});

// 正例：连续读跳写，读操作连续不冲突
Adds(dst, src, scalar, MASK_PLACEHOLDER, 1, {16, 1, 16, 1});
```

Benefit: 消除 bank 冲突，单 Repeat 从 8 拍降至 1-2 拍
Trade-off: 输出地址不连续，后续可能需要重排

---
## Variant B: 优化地址分配（padding 错开 bank group）
Source: SIMD算子性能优化/内存访问/避免Unified_Buffer的bank冲突.md

InitBuffer 时多申请 256 字节使 src0/src1 错开 bank group，确保 dst 不与 src 落入同一 bank。

**Expert implementation:**
```cpp
// 反例：x, y, z 连续分配，可能落入同一 bank group
pipe.InitBuffer(xBuf, 1, dataSize);
pipe.InitBuffer(yBuf, 1, dataSize);
pipe.InitBuffer(zBuf, 1, dataSize);

// 正例：x 多申请 256 字节，使 y 错开 bank group
pipe.InitBuffer(xBuf, 1, dataSize + 256);  // 额外 256B padding
pipe.InitBuffer(yBuf, 1, dataSize);
pipe.InitBuffer(zBuf, 1, dataSize);
// z = x + y 时，x/y/z 分布在不同 bank group
```

Benefit: 消除读读冲突和读写冲突
Trade-off: 需多申请 UB 空间（256 字节 padding）
