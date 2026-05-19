# P88 Multi-Phase Compute Phase Decomposition (多阶段计算拆分与资源重分配)
## Overview
当算子内部存在前置预处理、主计算、后处理或某个会被主循环重复引用的子计算时，可将其拆成多个彼此串行的独立阶段，并在阶段边界通过 `SyncAll()`、独立 `TPipe` 实例或 `pipe->Reset()` 重新分配 UB/L1/Workspace 等片上资源。该策略的核心不是单纯“多阶段”命名，而是**通过阶段化执行把互相竞争片上资源的子任务拆开**：要么让不同阶段拥有各自最合适的 buffer/TPipe 布局，要么把原本放在 Main 循环中的重复子计算前移为独立阶段并写入 workspace，供后续阶段直接消费。它适用于主阶段与前后处理阶段资源需求差异大、或某个子计算在主循环中被反复引用且值得前置抽取的场景。它不包含单纯的阶段内 buffer alias/zone reuse，也不包含跨 loop/batch/iteration 的常驻状态保留。

## When to Use
- 算子可自然拆成 Pre/Main/Post 或“主计算 + 后处理 + 类型转换”等两个及以上串行阶段
- 不同阶段的 UB/L1/buffer 布局差异显著，若强行共用同一套资源会导致片上利用率低或 buffer 冲突
- 主循环内存在会被多次重复引用的子计算，且可抽成独立阶段先算完并写入 workspace 供后续阶段直接读取
- 可接受阶段间使用 `SyncAll()`、独立 `TPipe` 或 `pipe->Reset()` 带来的同步与初始化开销
- 反向传播或复杂融合算子中，前置阶段、主阶段、后置阶段的职责清晰，可通过拆分降低主循环的 Vector/Cube 负担
- 若目标是阶段内复用同一块 UB/Workspace 物理空间，而不是在阶段边界重建资源，应优先参考 P71
- 若目标是跨 loop/batch/iteration 保留小参数、状态或累加器常驻 UB/TBuf，应优先参考 P67

## Trade-off
- 阶段边界的 `SyncAll()`、独立 `TPipe` 初始化/销毁或 `pipe->Reset()` 会引入额外固定开销
- 多阶段之间通常无法像单一深流水那样完全重叠，阶段切换过多时可能抵消收益
- 若阶段拆分粒度过细，workspace 读写、控制流和初始化成本会变成新的瓶颈
- 一旦后续优化把原本串行阶段改成重叠流水，原有阶段划分与资源布局可能需要重构

**Source operators**: lightning_indexer_enhance, sparse_lightning_indexer_grad_kl_loss_enhance, flash_attention_score_grad_enhance

---
## Variant A: 两阶段 `pipe->Reset()` 资源重建
Source: lightning_indexer_enhance, sparse_lightning_indexer_grad_kl_loss_enhance op_kernel

当主计算阶段与后处理阶段的 UB 需求差异很大，且两者严格串行时，可在第一阶段结束后执行 `SyncAll() + pipe->Reset()`，释放主阶段的 buffer，并按第二阶段需求重新初始化资源。该 Variant 适用于“主计算 + ScatterAdd/后处理”这类两阶段结构。

**Expert implementation:**
```cpp
void Process() {
    InitBuffersPhase1(pipe);
    MainLoop();

    SyncAll();
    pipe->Reset();

    vector2Service.InitBuffers(pipe);
    vector2Service.Process();
}
```

**vs. baseline (lingxi-code):**
```cpp
void Process() {
    InitAllBuffers(pipe);
    MainLoop();
    PostProcess();
}
```

Benefit: 每个阶段可独立最大化片上资源利用率，避免整条 kernel 生命周期内维持一套折中的 buffer 布局。
Trade-off: 阶段边界引入全核同步和整套 buffer 重建开销，且两阶段无法重叠执行。

---
## Variant B: 多阶段独立 `TPipe` 分离主计算与前后处理
Source: flash_attention_score_grad_enhance op_kernel

当算子天然包含 Pre、子预计算、Main、Post 等多个职责清晰的阶段时，可为每个阶段分配独立 `TPipe` 实例和各自 buffer 布局，通过多次 `SyncAll()` 串联执行。该 Variant 适用于阶段之间资源需求显著不同、且无需在阶段边界做整套 `pipe->Reset()` 的复杂训练算子。

**Expert implementation:**
```cpp
void VectorProcess() {
    { TPipe pipePre; VecPreProcess(pipePre); }
    SyncAll();

    { TPipe pipeSfmg; VecSftgProcess(pipeSfmg); }
    SyncAll();

    { TPipe pipeVec; VecMainProcess(pipeVec); }
    SyncAll();

    { TPipe pipePost; VecPostProcess(pipePost); }
}
```

**vs. baseline (lingxi-code):**
```cpp
void VectorProcess() {
    TPipe pipe;
    InitSharedBuffers(pipe);
    VecPreProcess(pipe);
    VecMainProcess(pipe);
    VecPostProcess(pipe);
}
```

Benefit: 各阶段可拥有最适合自己的 TPipe 与 buffer 布局，避免主阶段与前后处理互相挤占片上资源。
Trade-off: 多个独立阶段和多次同步会带来固定控制开销，阶段过多时可能削弱收益。

---
## Variant C: 重复子计算前置为独立预计算阶段
Source: flash_attention_score_grad_enhance op_kernel

当某个子计算结果会在 Main 循环中被多次引用时，可将其前置为独立阶段，在主阶段之前算完并写入 workspace，后续 Main 阶段只负责读取结果并继续主路径计算。该 Variant 的典型例子是将 softmax gradient `sum(dY * O)` 抽成独立 VecSfmg 阶段，从而减少 Main 循环内部的重复 Vector 计算。

**Expert implementation:**
```cpp
class VectorSoftmaxGrad {
    TPipe pipeSfmg;
    TBuf<> inputBuf;
    TBuf<> castBuf;

    void Process() {
        for (int s1 = 0; s1 < s1Loops; s1++) {
            CopyInSfmg(inputBuf, dyGm, oGm, s1);
            Cast(castBuf, inputBuf, RoundMode::CAST_NONE);
            Mul(castBuf, castDyBuf, castOBuf);
            ReduceSum(outputBuf, castBuf, reduceParams);
            DataCopy(sfmgWorkspaceGm[s1 * headNum], outputBuf, copyParams);
        }
    }
};

void VecMainProcess() {
    DataCopy(sfmgUb, sfmgWorkspaceGm[offset], params);
    Sub(dsUb, dyvUb, sfmgUb);
}
```

**vs. baseline (lingxi-code):**
```cpp
for (int s2 = 0; s2 < s2Loops; s2++) {
    Mul(tmpBuf, dyBuf, oBuf);
    ReduceSum(sfmgBuf, tmpBuf, params);
}
```

Benefit: 把会被 Main 多次重复引用的子计算移出主循环，降低主阶段 Vector 负担，并让预计算阶段使用独立 buffer 布局。
Trade-off: 需要额外 workspace 存储预计算结果，且预计算阶段与 Main 阶段之间需要显式同步。
