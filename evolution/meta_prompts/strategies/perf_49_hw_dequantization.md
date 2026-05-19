# P49: 硬件加速反量化 (Hardware-Accelerated Dequantization)

## Overview
用 AscendDequant 硬件融合指令（int32→float32 * scale 一步完成）+ Brcb 广播指令替代软件循环反量化，将 O(M) 次 Vector 操作压缩到 O(1) 次硬件指令。

## When to Use
- 任何 matmul 输出需要反量化的算子（int32→float 并乘 scale）
- 存在 per-token 或 per-channel scale 的量化算子
- 基线使用循环 Cast+Mul+Muls 实现反量化
- 需要减少 Vector 指令数量以释放 Vector 流水线

## Trade-off
- AscendDequant 需要特定硬件支持（910B+）
- scale 编码格式（UINT64）需要预处理
- SetQuantVector 仅支持 Cube 侧融合，部分场景仍需 Vector 后处理

**Source operators**: grouped_matmul, ffn, quant_matmul, quant_matmul_reduce_sum, grouped_matmul_swiglu_quant_v2, grouped_matmul_finalize_routing

---

## Variant A: AscendDequant + Brcb per-token scale 广播
Source: quant_matmul_reduce_sum

AscendDequant 完成 int32→float32*scale，Brcb 广播 per-token scale 到所有列。

**Expert implementation:**
```cpp
// 单条硬件指令完成反量化
AscendDequant(dequantResult, mmOut, scale, tmpLocal,
              {curVecBaseM, alignBaseN, curVecBaseN});

// Brcb 广播 per-token scale（替代逐行 Muls 循环）
Brcb(scaleLocal, perTokenScaleGm[offset], 1, 1, 0, 0);
Mul(output, dequantResult, scaleLocal, totalSize);
```

**vs. baseline (软件反量化):**
```cpp
Cast(fp32Local, inLocal_, CAST_NONE, tileSize_);  // int32→float32
for (int i = 0; i < subBlockM_; i++) {
    Mul(fp32Local[i*N], fp32Local[i*N], scale[i*N], N);   // per-channel scale
    Muls(fp32Local[i*N], fp32Local[i*N], tokenScale[i], N); // per-token scale
}
// 总计: 1 + M + M = 2M+1 次 Vector 操作
```

Benefit: Vector 指令从 O(M) 降到 O(1)，延迟减少 20-40%
Trade-off: 需要 AscendDequant 硬件支持

## Variant B: Fixpipe UINT64 编码硬件级反量化
Source: quant_matmul（框架级）

通过 Fixpipe 在 L0C→GM 写回路径上完成反量化，scale 编码为 UINT64 格式。

```cpp
// Scale 预编码为硬件格式
TransDequantScaleToM1(deqScale);  // float → UINT64 编码
// Fixpipe 在写回时自动完成 int32 * scale → float
FixpipeNzL0cToNdGm(output, l0cBuf, deqScale, ...);
```

Benefit: 反量化完全在写回路径完成，零 Vector 开销
Trade-off: UINT64 编码格式复杂，仅支持 Fixpipe 路径

## Variant C: BroadCast API 标量广播替代逐行 Muls
Source: grouped_matmul_finalize_routing

用 BroadCast API 将标量 scale 广播到整行，替代逐行 Muls 循环。

```cpp
// 替代: for (i=0; i<M; i++) Muls(out[i*N], out[i*N], scale[i], N)
BroadCast(scaleExpanded, scaleLocal, {M, N}, {M, 1});
Mul(output, input, scaleExpanded, M * N);
```

Benefit: 消除标量循环，单次向量化 Mul 完成
Trade-off: 需要额外 buffer 存储广播后的 scale

## Variant D: SetQuantVector 配置
Source: ffn

在 MatmulImpl 上配置 SetQuantVector，将反量化 scale 直接注入 Cube 流水线。

```cpp
mm.SetQuantVector(scaleGM[scaleOffset]);
// Cube 硬件自动完成: matmul_result * scale
// 无需 Vector 侧任何反量化操作
```

Benefit: 反量化完全由 Cube 硬件完成，Vector 流水线完全释放
Trade-off: 仅支持 per-channel scale，不支持 per-token scale
