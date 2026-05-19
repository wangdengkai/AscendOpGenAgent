# P70 FP Buffer 随路量化 (FixPipe Buffer Inline Quantization)
## Overview
对矩阵乘结果进行量化时，反例做法是将 CO1 结果搬到 workspace 再搬到 UB 进行 Vector 量化计算。正例做法是将量化参数搬运到 C2PIPE2GM（Fixpipe Buffer）上，调用一次 Fixpipe 接口同时完成矩阵乘结果搬出和量化计算，省去中间 CO1→workspace→UB 的搬运和 Vector 量化计算。

## When to Use
- Matmul 结果需要量化/反量化的算子
- Atlas A2 训练/推理系列产品
- Profiling 显示 CO1→GM→UB 搬运和 Vector 量化耗时较长

## Trade-off
- 仅支持 Atlas A2 训练/推理系列产品
- 需要将量化参数预搬到 Fixpipe Buffer（C2PIPE2GM 位置）
- 量化模式受 Fixpipe 硬件支持的格式限制

**Source operators**: SIMD算子性能优化/矩阵计算

---
## Variant A: Fixpipe Buffer 存放量化参数一次完成量化搬出
Source: SIMD算子性能优化/矩阵计算/通过FP_Buffer存放量化参数实现高效随路量化.md

将量化参数从 GM→L1→C2PIPE2GM，Fixpipe 在搬出 CO1 结果时同步完成量化。

**Expert implementation:**
```cpp
// 反例：CO1→workspace→UB 量化→GM（多次搬运 + Vector 量化计算）
void CopyOut() {
    Fixpipe(xGm, c1Local, fixpipeParams);  // CO1→workspace
}
void CopyIn1() {
    DataCopy(src0Local, xGm, cSize);  // workspace→UB
}
void Compute1() {
    Cast(tmpLocal, src0Local, ...);  // UB 上 Vector 量化
    Mul(tmpHalfBuffer, tmpLocal, deqLocal, cSize);
    Cast(dstLocal, tmpHalfBuffer, ...);
}

// 正例：量化参数预搬到 FB，一次 Fixpipe 完成
TQue<QuePosition::C1, 1> inQueueDeq1;      // GM→L1
TQue<QuePosition::C2PIPE2GM, 1> inQueueDeq; // L1→FB

void SplitDeq() {
    DataCopy(deqLocal, deq1Local, {...});  // L1→FB
}
void CopyOut() {
    SetFixpipeNz2ndFlag(1, 0, 0);
    dataCopyParams.quantPre = QuantMode_t::VQF322B8_PRE;
    dataCopyParams.nz2ndEn = true;
    DataCopy(cGM, c1Local, dataCopyParams);  // CO1→GM 含量化
}
```

Benefit: 省去 CO1→workspace→UB 搬运和 Vector 量化计算，减少搬运次数和计算耗时
Trade-off: 仅 Atlas A2 系列产品支持，量化参数需预搬到 FB
