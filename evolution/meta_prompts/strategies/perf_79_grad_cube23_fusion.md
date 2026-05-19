# P79 Grad Cube2+Cube3 融合计算 (Grad Cube2+Cube3 Fusion for d=64)
## Overview
在 Flash Attention 反向传播中，Cube2（dS*K→dQ）和 Cube3（dS^T*Q→dK, S^T*dY→dV）共享输入矩阵 dS/S。当 headDim=64 且 sparseMode=0 时，将 Cube2 和 Cube3 融合为 `Cube23Process`，利用 L0A 的 ping/pong 分别存放 matrixA 和 matrixA^T，L0B 的 ping/pong 分别存放 K 和 Q，减少一次完整的 L1→L0 搬运流程。

## When to Use
- Flash Attention Backward，headDim=64 且 sparseMode=0
- Cube2 和 Cube3 共享输入矩阵（dS 和 S）
- L0C 容量足够容纳 4 个 buffer（2 for dQ + 2 for dK）

## Trade-off
- 仅适用于 d=64 场景，其他 headDim 需要分离的 Cube2+Cube3
- L0C 需要 4 个 buffer（128K 全占满），容量压力大
- 代码复杂度显著增加

**Source operators**: flash_attention_score_grad_enhance

---
## Variant A: L0A Transpose 共享 + L0C 4-Buffer（训练 FAG d=64）
Source: flash_attention_score_grad_enhance op_kernel

利用 LoadData 的 transpose 参数实现 L1→L0A 的转置加载，L0A ping 存 matrixA，L0A pong 存 matrixA^T。

**Expert implementation:**
```cpp
// Cube23 融合: d=64 特化
bool cube23_mix_flag = (sparseMode == 0 && headDim == 64);

if (cube23_mix_flag) {
    // L0A ping: matrixA (dS for dQ)
    // L0A pong: matrixA^T (dS^T for dK, via transpose load)
    LoadData(l0a_ping, l1_dS, loadParams);           // 正常加载
    LoadData(l0a_pong, l1_dS, loadParamsTranspose);   // 转置加载

    // L0B ping: K (for dQ = dS * K)
    // L0B pong: Q (for dK = dS^T * Q)
    LoadData(l0b_ping, l1_K, loadParams);
    LoadData(l0b_pong, l1_Q, loadParams);

    // L0C 4-buffer: c1_ping/pong for dQ, c2_ping/pong for dK
    Mmad(l0c_c1, l0a_ping, l0b_ping, mmadParams);  // dQ += dS * K
    Mmad(l0c_c2, l0a_pong, l0b_pong, mmadParams);  // dK += dS^T * Q

    // dQ 使用 AtomicAdd 累加到 GM
    SetAtomicType<float>();
    Fixpipe(dqGm, l0c_c1, fixpipeParams);
}

**vs. baseline (lingxi-code):**
```cpp
// 基线：Cube2 和 Cube3 分离执行
// Cube2: dQ = dS * K
LoadData(l0a, l1_dS, params);
LoadData(l0b, l1_K, params);
Mmad(l0c, l0a, l0b, params);
Fixpipe(dqGm, l0c, params);

// Cube3: dK = dS^T * Q (需要重新加载 dS 并转置)
LoadData(l0a, l1_dS, paramsTranspose);  // 额外的 L1→L0 搬运
LoadData(l0b, l1_Q, params);
Mmad(l0c, l0a, l0b, params);
Fixpipe(dkGm, l0c, params);
```

Benefit: 减少一次完整的 L1→L0 搬运；L0A 的 matrixA 和 matrixA^T 共享同一份 L1 数据
Trade-off: L0C 需要 4 个 buffer；仅适用于 d=64
