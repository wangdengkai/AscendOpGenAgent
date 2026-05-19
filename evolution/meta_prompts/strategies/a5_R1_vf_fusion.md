# R1: VF 函数融合

## Overview
将多个独立的 `__simd_vf__` 函数合并为一个，减少中间结果的 Load/Store 操作，最大化寄存器复用。

## When to Use
- 多个连续的 VF 函数操作相同数据
- 存在冗余的 Store→Load 序列（VF A 的输出 Store 到 UB，VF B 再 Load 回来）
- Profiling 显示 Load/Store 指令占比高

## Trade-off
- **收益**: 减少内存访问 2-5x（取决于融合 VF 数量）
- **风险**: RegTensor 数量增加，可能超过 32 限制
- **复杂度**: 需确保控制流等价，循环结构一致

## Variant A: 手动融合

将两个独立 VF 合并为一个。

### Baseline (Phase 1 翻译输出)
```cpp
// 两个独立 VF: Add 和 Mul
__simd_vf__ inline void AddVF(__ubuf__ float* dst, __ubuf__ float* s0, __ubuf__ float* s1,
    uint32_t count, uint32_t oneRepSize, uint16_t repTimes) {
    RegTensor<float> r0, r1, d;
    MaskReg mask;
    for (uint16_t i = 0; i < repTimes; ++i) {
        mask = UpdateMask<float>(count);
        LoadAlign(r0, s0 + i * oneRepSize);
        LoadAlign(r1, s1 + i * oneRepSize);
        Add(d, r0, r1, mask);
        StoreAlign(dst + i * oneRepSize, d, mask);  // Store 到 UB
    }
}

__simd_vf__ inline void MulVF(__ubuf__ float* dst, __ubuf__ float* s0, __ubuf__ float* s1,
    uint32_t count, uint32_t oneRepSize, uint16_t repTimes) {
    RegTensor<float> r0, r1, d;
    MaskReg mask;
    for (uint16_t i = 0; i < repTimes; ++i) {
        mask = UpdateMask<float>(count);
        LoadAlign(r0, s0 + i * oneRepSize);  // 从 UB Load 回来
        LoadAlign(r1, s1 + i * oneRepSize);
        Mul(d, r0, r1, mask);
        StoreAlign(dst + i * oneRepSize, d, mask);
    }
}

// 调用: Store→Load 冗余
VF_CALL<AddVF>(tmp, s0, s1, count, repSize, repTimes);
VF_CALL<MulVF>(dst, tmp, s2, count, repSize, repTimes);
```

### Evolved (手动融合)
```cpp
__simd_vf__ inline void FusedAddMulVF(__ubuf__ float* dst,
    __ubuf__ float* s0, __ubuf__ float* s1, __ubuf__ float* s2,
    uint32_t count, uint32_t oneRepSize, uint16_t repTimes) {
    RegTensor<float> r0, r1, r2, tmp, d;
    MaskReg mask;
    for (uint16_t i = 0; i < repTimes; ++i) {
        mask = UpdateMask<float>(count);
        LoadAlign(r0, s0 + i * oneRepSize);
        LoadAlign(r1, s1 + i * oneRepSize);
        LoadAlign(r2, s2 + i * oneRepSize);
        Add(tmp, r0, r1, mask);       // tmp 留在寄存器
        Mul(d, tmp, r2, mask);         // 直接使用 tmp
        StoreAlign(dst + i * oneRepSize, d, mask);  // 只 Store 最终结果
    }
}
// RegTensor: 5 (r0, r1, r2, tmp, d)
// 减少: 1x Store + 1x Load per iteration
```

**Benefit**: 减少 2 次内存访问/iteration
**Trade-off**: RegTensor 从 3 增加到 5

## Variant B: 编译器自动融合

确保两个 VF 的控制流等价，让编译器自动融合。

```cpp
// 使用连续计算 API（编译器识别后自动融合）
Add(yLocal, xLocal, xLocal, count);
Mul(yLocal, yLocal, xLocal, count);
// 编译器将 Add 和 Mul 对应的 VF 自动融合
```

**条件**: 必须使用连续计算 API 模式（非高维切分模式）

## Variant C: 多步融合

融合 3 个以上操作（如 GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))）

**注意**: 需确保 RegTensor 总数 ≤ 32，否则拆分为多个 VF
