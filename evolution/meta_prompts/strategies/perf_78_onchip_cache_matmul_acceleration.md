# P78 片上缓存加速 Matmul (On-Chip Cache Matmul Acceleration)
## Overview
利用 TSCM（Tensor Shared Cache Memory）或 L1 Carry 路径将 Matmul 的输入矩阵预加载到片上缓存，避免 Matmul 内部的 GM→L1 搬运延迟。TSCM 路径适用于训练场景的中等规模矩阵，L1 Carry 路径适用于推理 decode 场景的小 M 大 N 矩阵。

## When to Use
- BMM1/BMM2 的输入矩阵可以放入 TSCM 或 L1（headDim <= 192）
- Decode 场景 S1=1，Q 矩阵极小可放入 L1
- Matmul 的 MTE2 搬运成为性能瓶颈

## Trade-off
- TSCM/L1 容量有限，大 headDim 场景可能放不下
- 需要额外的 Nd2Nz 预处理步骤
- 与标准 GM-based Matmul 共存，增加代码复杂度

**Source operators**: flash_attention_score_enhance, ai_infra_esa_select_topk

---
## Variant A: TSCM 矩阵缓存（训练 FAE）
Source: flash_attention_score_enhance op_kernel

FAE-SAB 变体定义 `FaTscmArray` 管理 TSCM buffer，当 `mmPolicyType == UNSPLITK` 时，BMM1/BMM2 输入从 GM 切换为 TSCM 位置。

**Expert implementation:**
```cpp
// TSCM 矩阵缓存
struct FaTscmArray {
    TBuf<TPosition::TSCM> tscmBuf[TSCM_BUF_NUM];  // 2 个 TSCM 槽
    void InitTscmBuffer(TPipe* pipe) {
        pipe->InitBuffer(tscmBuf[Q_VEC1_INDEX], tscmSize);
        pipe->InitBuffer(tscmBuf[K_V_INDEX], tscmSize);
    }
};
// Matmul 使用 TSCM 位置
if (mmPolicyType == MmPolicyType::UNSPLITK) {
    mm1.SetAType(TPosition::TSCM, CubeFormat::NZ);
    mm1.SetBType(TPosition::TSCM, CubeFormat::NZ);
    mm1LoadData(tscmBuf[Q_VEC1_INDEX], qGm);  // GM→TSCM 预加载
}
```

**vs. baseline (lingxi-code):**
```cpp
// 基线：Matmul 从 GM 读取，内部 MTE2 搬运
mm1.SetAType(TPosition::GM, CubeFormat::ND);
mm1.SetBType(TPosition::GM, CubeFormat::ND);
mm1.IterateAll(qGm, kGm, outGm);  // MTE2 搬运在 Matmul 内部
```

Benefit: TSCM 带宽远高于 GM→L1 路径；避免 L1 容量不足时的反复搬运
Trade-off: TSCM 容量有限；需要额外 LoadData 调用

---
## Variant B: L1 Carry Matmul（推理 Decode）
Source: ai_infra_esa_select_topk op_kernel

ESA 在 decode 场景使用 L1 Carry 路径：Q（8KB）和 K（500KB）预加载到 L1 并转为 NZ 格式，Matmul 直接从 L1 读取。

**Expert implementation:**
```cpp
// L1 Carry Matmul
constexpr int L1_Q_SIZE = 8 * 1024;   // 8KB
constexpr int L1_K_SIZE = 500 * 1024; // 500KB

// 预加载 Q/K 到 L1 并转 NZ
Nd2Nz(l1QBuf, qGm, nd2nzParams);
Nd2Nz(l1KBuf, kGm, nd2nzParams);

// mmL1Carry 从 TSCM/L1 读取
mmL1Carry.SetAType(TPosition::TSCM, CubeFormat::NZ);
mmL1Carry.SetBType(TPosition::TSCM, CubeFormat::NZ);
mmL1Carry.IterateAll(l1QBuf, l1KBuf, outGm);
```

Benefit: 消除 Matmul 内部 GM→L1 搬运延迟；适合小 M 大 N 的 decode 场景
Trade-off: L1 容量限制（Q=8KB, K=500KB）；仅 decode 场景有效
