# P69 UB 融合连续 Vector 计算 (UB Fused Vector Chain)
## Overview
多次 Vector 计算且前一次输出是后一次输入时，反例做法是每次计算后将结果从 UB 搬到 GM 再搬回 UB。正例做法是使用 TQue<VECCALC> 将中间结果暂存在 UB 上直接作为下一次计算输入，仅在首次搬入和最终搬出时访问 GM。n 次 Vector 计算从 2n 次 GM 搬运减少到 2 次。

## When to Use
- 多步 Vector 计算链（如 Exp → Abs → Mul）
- 前一步输出直接作为后一步输入的算子
- Profiling 显示 MTE2/MTE3 搬运时间占比高

## Trade-off
- 需要额外的 VECCALC buffer 空间
- 中间结果链过长时可能受限于 UB 容量

**Source operators**: SIMD算子性能优化/矢量计算

---
## Variant A: TQue<VECCALC> 中间结果直通
Source: SIMD算子性能优化/矢量计算/通过Unified_Buffer融合实现连续vector计算.md

使用 VECIN→VECCALC→VECOUT 队列链实现 UB 内连续计算，消除中间 GM 搬运。

**Expert implementation:**
```cpp
// 反例：Exp 后搬出 GM，再搬入做 Abs（4 次 GM 搬运）
void Process() {
    CopyIn(); Compute_Exp(); CopyOut();   // GM→UB→Exp→UB→GM
    CopyIn1(); Compute_Abs(); CopyOut1(); // GM→UB→Abs→UB→GM
}

// 正例：UB 融合，仅 2 次 GM 搬运
void Compute() {
    LocalTensor<float> src0Local = inQueueSrc0.DeQue<float>();
    LocalTensor<float> dstLocal = outQueueDst.AllocTensor<float>();
    Exp(dstLocal, src0Local, 1024);  // VECIN → VECCALC 中间结果
    Abs(dstLocal, dstLocal, 1024);   // 直接在 UB 上继续计算
    outQueueDst.EnQue<float>(dstLocal);
    inQueueSrc0.FreeTensor(src0Local);
}
```

Benefit: GM 搬运次数从 2n 降至 2 次，显著减少 MTE2/MTE3 时间
Trade-off: 需要 VECCALC buffer 空间存放中间结果
