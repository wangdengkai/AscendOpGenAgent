# R4: Hardware Loop 规范化

## Overview
确保 VF 内的循环满足 Hardware Loop 编码要求，避免退化为 Software Loop。

## When to Use
- VF 函数中的循环未被编译器识别为 Hardware Loop
- 循环体内包含 if/else 或三元运算符
- 循环变量类型不是 uint16_t
- 循环起始值不为 0 或步长不为 1

## Trade-off
- **收益**: Hardware Loop 比 Software Loop 快 2-5x（减少分支预测开销）
- **风险**: 代码可读性略降（for(1) 替代 if）
- **复杂度**: 低（机械替换）

## Variant A: 循环变量类型修正

```cpp
// [BAD] Software Loop: uint32_t 类型
for (uint32_t i = 0; i < bound; i++) { ... }

// [OK] Hardware Loop: uint16_t 类型
for (uint16_t i = 0; i < (uint16_t)bound; i++) { ... }
```

## Variant B: if/else 消除

```cpp
// [BAD] Software Loop: 包含 if
for (uint16_t i = 0; i < repTimes; i++) {
    if (i == repTimes - 1 && tailCount > 0) {
        // 尾块处理
    } else {
        // 正常处理
    }
}

// [OK] Hardware Loop: for(1) 替代 if
for (uint16_t i = 0; i < mainRepTimes; i++) {
    // 正常处理
}
uint16_t hasTail = !!tailCount;
for (uint16_t i = 0; i < hasTail; i++) {
    // 尾块处理
}
```

## Variant C: 循环起始值/步长修正

```cpp
// [BAD] Software Loop: 起始值 2, 步长 2
for (uint16_t i = 2; i < bound * 3; i += 2) { ... }

// [OK] Hardware Loop: 起始值 0, 步长 1, 内部计算实际索引
uint16_t loopCount = (bound * 3 - 2 + 1) / 2;
for (uint16_t i = 0; i < loopCount; i++) {
    uint16_t actualIdx = i * 2 + 2;
    // 使用 actualIdx
}
```

## Variant D: if constexpr（编译期常量）

```cpp
// 当条件是编译期已知的
template <bool HAS_BIAS>
__simd_vf__ inline void ComputeVF(...) {
    for (uint16_t i = 0; i < repTimes; i++) {
        LoadAlign(srcReg, srcAddr + i * repSize);
        if constexpr (HAS_BIAS) {
            LoadAlign(biasReg, biasAddr + i * repSize);
            Add(srcReg, srcReg, biasReg, mask);
        }
        StoreAlign(dstAddr + i * repSize, srcReg, mask);
    }
}
```
