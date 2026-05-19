# P42: workspace buffer 多区域分时复用
## Overview
一块大的 TBuf<VECCALC> 被划分为多个逻辑区域，供 RmsNorm 和 RoPE 交替使用。RmsNorm 使用 zone0/1/2，RoPE 复用 zone1/2 做 cos/sin 和中间变量。

## When to Use
- 融合算子中多个计算阶段需要大量 UB 临时空间但不同时执行
- 各阶段的临时数据无需跨阶段保留，可安全覆盖（如 RmsNorm 结果用完后 RoPE 可复用同一区域）

## Trade-off
- 各阶段不能同时执行，zone 复用要求严格的阶段串行顺序
- wsBuffer 大小 = zone 数 × rows × headSize × sizeof(float)，headSize 较大时占用显著

**Source operators**: ai_infra_kv_rms_norm_rope_cache

---

## Variant A: 三区域分时复用 RmsNorm/RoPE
Source: ai_infra_kv_rms_norm_rope_cache

将 wsBuffer 按 rows×headSize 为单位划分为 zone0/1/2，RmsNorm 阶段使用全部三个区域，RoPE 阶段复用 zone1/2 存放 cos/sin。

**Expert implementation:**
```cpp
int64_t xLocalFp32Offset = 0;                  // zone0: RmsNorm fp32
int64_t xSquareLocalOffset = rows * headSize;   // zone1: RmsNorm square / RoPE cos
int64_t xSumLocalOffset = rows * headSize * 2;  // zone2: RmsNorm sum / RoPE sin
LocalTensor<float> xLocalFp32 = wsLocal[xLocalFp32Offset];
LocalTensor<float> xSquareLocal = wsLocal[xSquareLocalOffset];
```

Benefit: 一块 buffer 服务多个计算阶段，最大化 UB 利用率
Trade-off: 各阶段必须串行执行，无法流水重叠
