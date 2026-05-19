# P87 Manual Mmad Pipeline Timing Control (手动 Mmad 流水线时序控制)
## Overview
当 AscendC 的 Matmul 高阶 API 无法满足特定时序需求时，可以完全绕过 Matmul 库，手动控制 L0A/L0B/L0C 的双缓冲、HardEvent 同步、`unitFlag` 驱动的 Mmad/Fixpipe 融合，以及基于矩阵规模的条件性 `PipeBarrier<PIPE_M>()`。这类优化的共同点不是改变数学逻辑或分核拓扑，而是**直接操控手动 Mmad 执行路径内部的 Cube 时序**：用 L0 双缓冲隐藏搬运延迟，用 `unitFlag` 把最后一次 Mmad 与 Fixpipe 融合，用条件性 `PipeBarrier` 避免小矩阵累加模式下的数据冒险。它适用于同一 kernel 内存在多个不同形状 Matmul、或需要精确安排 K 轴多次迭代时序的场景。它不包含 Matmul 库内部 `MatmulConfig` 调参，也不包含 Split-K/AtomicAdd 或 d=64 特化的算法级 Cube 融合。

## When to Use
- 需要完全绕过 AscendC Matmul 库，手动编排 L0A/L0B/L0C 的搬运、Mmad 和 Fixpipe 时序
- 同一 kernel 中存在多个不同形状 Matmul，或需要共享 L0 buffer、精确控制多段手动 Matmul 流水
- K 轴需要多次迭代，且 profiling 显示 Mmad/Fixpipe 串行等待或 Fixpipe 启动过晚
- L0C 处于累加模式，且矩阵规模跨度较大，需要按大小决定是否插入 `PipeBarrier<PIPE_M>()`
- 若希望继续使用 Matmul 高阶 API 并通过 `MatmulConfig`、模板或 `enUnitFlag` 调参优化，应优先参考 P72
- 若需要改变多核并行拓扑并引入 Split-K/AtomicAdd 归约，应优先参考 P58
- 若需要 d=64、sparseMode=0 下的 Cube2+Cube3 特化融合，应优先参考 P65

## Trade-off
- 代码复杂度和维护成本极高，需要手动管理 L0 buffer 生命周期、HardEvent、`unitFlag` 和同步点
- 错误的 `unitFlag` 或 `PipeBarrier` 条件会直接导致数据错误、死锁或数据竞争
- 不同 headDim、K 切分层级和硬件环境下，最佳的 kL1/kL0 参数与同步阈值可能不同，通常需要 profiling 验证
- 该策略关注的是手动 Mmad 流水线时序控制；若真实瓶颈在 Matmul 库模板、跨核并行拓扑或算法级数据复用，继续沿此方向优化会失焦

**Source operators**: lightning_indexer_enhance, sparse_flash_attention_grad_enhance, sparse_lightning_indexer_grad_kl_loss_enhance, sparse_flash_attention_enhance, ai_infra_sparse_flash_attention_gqa, ai_infra_sparse_flash_attention_pioneer

---

## Variant A: L0 双缓冲 + HardEvent 手动流水骨架
Source: sparse_lightning_indexer_grad_kl_loss_enhance, lightning_indexer_enhance op_kernel

当需要绕过 Matmul 库、手动组织单级 K 轴迭代的 Mmad 流水时，可为 L0A/L0B/L0C 各分配 ping-pong 双缓冲，并用 HardEvent 组织 L1→L0 搬运和 Mmad 的生产者-消费者同步。该 Variant 提供手动 Mmad 流水线的基础骨架，其他同步优化通常都建立在此之上。

**Expert implementation:**
```cpp
// 手动 Mmad 流水线
TBuf<TPosition::A1> l0aBuf;  // 32K * 2 (ping/pong)
TBuf<TPosition::B1> l0bBuf;  // 32K * 2
TBuf<TPosition::C1> l0cBuf;  // 64K * 2

for (int k = 0; k < kLoops; k++) {
    int pp = k % 2;
    WaitFlag(HardEvent::M_MTE1, L0AB_EVENT + pp);

    LoadData(l0a[pp], l1Key[kvIdx], load3DParams);
    LoadData(l0b[pp], l1Query[qpIdx], load2DParams);

    SetFlag(HardEvent::MTE1_M, L0AB_EVENT + pp);
    WaitFlag(HardEvent::MTE1_M, L0AB_EVENT + pp);

    MmadParams mmadParams;
    mmadParams.unitFlag = (k == kLoops - 1) ? 0b11 : 0b10;
    Mmad(l0c[pp], l0a[pp], l0b[pp], mmadParams);

    SetFlag(HardEvent::M_MTE1, L0AB_EVENT + pp);
}
```

**vs. baseline (lingxi-code):**
```cpp
// 基线：使用 Matmul 库 API
Matmul<...> mm;
mm.SetTensorA(queryGm);
mm.SetTensorB(keyGm);
mm.IterateAll(queryGm, keyGm, outGm);
// Matmul 库内部管理 L0 buffer 和同步
```

Benefit: 精确控制 L0 搬运与 Cube 执行时序；允许多个 Matmul 共享 L0 buffer，避免 Matmul 库调度限制。
Trade-off: 需要手动管理所有同步点和 buffer 生命周期，代码复杂度显著增加。

---

## Variant B: 多级 K 切分与 7-buffer L1 配合的手动 Mmad
Source: sparse_flash_attention_enhance op_kernel

当 MLA 等场景的 headDim 很大，单级 K 切分无法高效匹配片上资源时，可将 K 轴拆为 kL1/kL0 两级切分，并与 L1 上的 QP 4-buffer + KV 3-buffer 方案配合。该 Variant 仍属于手动 Mmad 流水线，只是把基础骨架扩展为更复杂的多级 K 迭代结构。

**Expert implementation:**
```cpp
constexpr int kSize = 576;
constexpr int kL1Size = 288;
constexpr int kL1Loops = 2;
constexpr int kL0Size = 96;
constexpr int kL0Loops = 3;

for (int kL1 = 0; kL1 < kL1Loops; kL1++) {
    DataCopy(bufKVL1[kvL1BufIdx % 3], kvGm[offset], params);

    for (int kL0 = 0; kL0 < kL0Loops; kL0++) {
        LoadData(l0a[pp], bufKVL1[idx], load3DParams);
        Mmad(l0c[pp], l0a[pp], l0b[pp], mmadParams);
    }
}
```

**vs. baseline (lingxi-code):**
```cpp
// 基线：单级 Matmul 库调度
Matmul<...> mm;
mm.IterateAll(queryGm, keyGm, outGm);
// 无法显式控制 kL1/kL0 两级切分与 L1 7-buffer 配合
```

Benefit: 让 K 轴切分与 L1/L0 资源布局更精确匹配，适合大 headDim 的手动 Cube 流水。
Trade-off: 高度依赖特定 headDim 与 buffer 布局；错误迁移到非 MLA 场景容易失效甚至出错。

---

## Variant C: UnitFlag 驱动的 Mmad-Fixpipe 融合同步
Source: ai_infra_sparse_flash_attention_pioneer op_kernel, ai_infra_sparse_flash_attention_gqa op_kernel

在手动 Mmad 流水线的 K 轴多次迭代场景中，中间迭代只做 Mmad 累加，最后一次迭代再同时触发 Mmad 与 Fixpipe，可用 `unitFlag` 把同步粒度精确压缩到最后一个有效累加点。该 Variant 必须依赖 Variant A 或 B 这样的手动 Mmad 骨架，不适用于 Matmul 库内部的 `enUnitFlag` 配置。

**Expert implementation:**
```cpp
MmadParams mmadParams;
mmadParams.unitFlag = (kL1 == 1 && kL0 == (kL0Loops - 1)) ? 0b11 : 0b10;
// 0b10: 仅 Mmad 累加
// 0b11: Mmad + Fixpipe 同时启动
Mmad(l0cTensor, l0aTensor, l0bTensor, mmadParams);

FixpipeParamsV220 fixpipeParams;
fixpipeParams.unitFlag = mmadParams.unitFlag;
Fixpipe(outTensor, l0cTensor, fixpipeParams);
```

**vs. baseline (lingxi-code):**
```cpp
for (int k = 0; k < kLoops; k++) {
    Mmad(l0c, l0a, l0b, params);  // 默认每次都等价于完整收尾
    // Fixpipe 与下一次 Mmad 串行，无法形成更细粒度重叠
}
```

Benefit: 让最后一次有效累加与 Fixpipe 精确衔接，减少 Mmad/Fixpipe 串行等待，提升手动 Cube 流水重叠度。
Trade-off: `unitFlag` 条件必须与 K 轴切分层级严格匹配；单级 K 和多级 K 场景的判断条件不同，误设会直接导致数据错误。

---

## Variant D: 基于矩阵规模的条件性 PipeBarrier
Source: sparse_lightning_indexer_grad_kl_loss_enhance, lightning_indexer_enhance op_kernel

在手动 Matmul 的 L0C 累加模式下，小矩阵的 Mmad 执行时间不足以完全覆盖后续搬运，如果仍像大矩阵那样省略同步，会引入数据冒险。此时可根据 fractal 数量阈值，只在小矩阵场景下插入 `PipeBarrier<PIPE_M>()`，而在大矩阵场景保持无额外 barrier 的更深流水。

**Expert implementation:**
```cpp
void ManualMmad(const MmParam& mmParam, const MmadParams& mmadParams) {
    LoadData(l0a, l1Src, loadParams);
    Mmad(l0c, l0a, l0b, mmadParams);

    if (mmParam.isL0CAccum &&
        ((mmadParams.m / 16) * (mmadParams.n / 16) < 10)) {
        PipeBarrier<PIPE_M>();
    }
}
```

**vs. baseline (lingxi-code):**
```cpp
Mmad(l0c, l0a, l0b, mmadParams);
PipeBarrier<PIPE_M>();  // 每次无条件等待
```

Benefit: 小矩阵保证正确性，大矩阵避免不必要的同步开销，在正确性与性能之间取得更好的平衡。
Trade-off: 阈值是经验值，迁移到不同硬件、不同 CANN 版本或不同手动 Mmad 结构时需要 profiling 复核。
