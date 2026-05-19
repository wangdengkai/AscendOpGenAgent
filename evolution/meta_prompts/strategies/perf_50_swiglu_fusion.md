# P50: SwiGLU 融合流水线 (SwiGLU Fusion Pipeline)

## Overview
用高阶 SwiGLU API + Ping-Pong 双缓冲实现 AIC-AIV 流水线，将 5 步手动 sigmoid+gate 操作压缩为单条 API 调用，减少 80% 指令数和 75% PipeBarrier 同步。

## When to Use
- MoE FFN 融合算子中的 SwiGLU 激活（适用范围较窄）
- 基线使用 Muls(-1)→Exp→Adds(1)→Div→Mul 五步实现 sigmoid gate
- 需要减少 Vector 流水线占用以释放给反量化等后处理
- Cube 输出需要经过 SwiGLU 激活后再进入下一阶段

## Trade-off
- SwiGLU API 可能不支持所有数据类型（需检查硬件版本）
- 适用范围较窄（narrow scope），仅限 MoE FFN 融合算子
- Ping-Pong 双缓冲增加 UB 占用

**Source operators**: ffn, grouped_matmul_swiglu_quant_v2

---

## Variant A: GLU Ping-Pong 双缓冲 + 库 API
Source: ffn

使用 SwiGLU 库 API 配合 Ping-Pong 双缓冲实现 AIC-AIV 流水线重叠。

**Expert implementation:**
```cpp
// 单条 API 调用完成 SwiGLU
SwiGLU<float, false>(workspace, src0, src1, beta, halfTokenLen);
PipeBarrier<PIPE_V>();
// Ping-Pong: 当前 tile 计算时，下一 tile 的 Cube 输出已搬入另一 buffer
```

**vs. baseline (手动 5 步):**
```cpp
// Step 1: sigmoid(-x)
Muls(tmpLocal, aLocal, -1.0f, tileSize_);
PipeBarrier<PIPE_V>();
// Step 2: exp(-x)
Exp<float, 0>(bLocal, tmpLocal, tileSize_);
PipeBarrier<PIPE_V>();
// Step 3: 1 + exp(-x)
Adds(bLocal, bLocal, 1.0f, tileSize_);
PipeBarrier<PIPE_V>();
// Step 4: sigmoid = 1 / (1 + exp(-x))
Duplicate(onesLocal, 1.0f, tileSize_);
Div(tmpLocal, onesLocal, bLocal, tileSize_);
PipeBarrier<PIPE_V>();
// Step 5: gate = x * sigmoid(x)
Mul(aLocal, aLocal, tmpLocal, tileSize_);
// 需要额外 onesLocal buffer
```

Benefit: 指令数 5→1（80% 减少），PipeBarrier 4→1（75% 减少），省去 onesLocal buffer
Trade-off: 依赖 SwiGLU 库 API 可用性

## Variant B: 高阶 SwiGLU API + 单趟计算避免重计算
Source: grouped_matmul_swiglu_quant_v2

在 Cube 输出的两半（gate_proj 和 up_proj）上直接调用 SwiGLU，避免分开计算再合并。

```cpp
// Cube 输出 [M, 2N]: 前半 gate_proj, 后半 up_proj
// 单趟 SwiGLU: swiglu(gate_proj, up_proj) → [M, N]
SwiGLU<float, false>(output, gateProj, upProj, beta, tokenLen);
// 避免: 分别计算 sigmoid(gate) 和 gate*up 再合并
```

Benefit: 单趟计算避免重计算和中间 buffer，减少 30-40% 激活阶段延迟
Trade-off: 要求 Cube 输出 layout 为 [M, 2N] 连续排列
