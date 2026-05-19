# P62 Draft: Sparse/Dense Dual-Path Dispatch by Reachable Coverage（按可达覆盖度切换 sparse/dense 双路径，草稿）

## Status
- **Draft / pending cross-family validation**
- 本文件是候选草稿，不属于本轮正式入库策略
- 本轮**不更新** `evolution/meta_prompts/strategy-index.md`
- 只有在再找到至少一个独立 sibling / 非同血统 sparse attention 算子复现后，才考虑转正入库

## Overview
部分 sparse attention 训练算子并不会无条件执行 gather-merge sparse path。更优实现会先比较“当前 query 可达的有效稀疏跨度”与“已选 sparse/topK 预算是否已经足够覆盖”，再在运行时切换两条不同的数据流：
- 若 sparse/topK 覆盖已经足够，则直接走 dense GM path，避免额外 gather/merge
- 若可达跨度超出已选 sparse 预算，则保留 sparse gather/merge path，把不连续 KV 先整理到 workspace 再供后续阶段消费

这不是普通的 tiling 调节，而是基于可达覆盖度的运行时 dataflow dispatch：两条分支在输入来源、搬运模式、workspace 依赖以及后续流水结构上都显著不同。

## Why This Is Still a Draft
- 目前证据主要来自同一 sparse attention 家族的两个近邻实现：`sparse_flash_attention_grad_enhance` 与 `sparse_lightning_indexer_grad_kl_loss_enhance`
- 虽然它们的命名面不同（`isSmallS2` vs `mergeKv`），但抽象语义接近：都在回答“当前可达 S2 / 稀疏覆盖是否已足够，是否还值得走 gather sparse path”
- 现阶段仍缺少更宽的跨 family 复现，因此本轮只保留为 `_draft`

## When to Use
- sparse attention / long-sequence attention 训练算子，存在显式 topK 或 sparseIndices 驱动的稀疏访问
- sparse path 与 dense path 在 memory movement、workspace 依赖、后续流水结构上差异显著
- 可在 task / tile 级计算“有效可达跨度”或“已选稀疏覆盖预算”，并以此驱动路径切换

## Trade-off
- 需要额外维护两套有效的数据流实现，而不仅是同一路径上的参数微调
- 运行时判定变量若定义不稳，会导致 dispatch 语义含糊，难以复用到其他算子
- 很容易被误写成“已经通用化的主策略”；当前证据宽度还不足，必须保留 draft 边界

---

## Variant A: `isSmallS2` 形式的有效覆盖判定
Source: sparse_flash_attention_grad_enhance

该实现根据“当前可达 S2 范围”与“actualSelectedBlockCount × selectedBlockSize”比较，得到 `isSmallS2`。当可达范围已被 selected sparse coverage 覆盖时，后续阶段可以直接走 dense/direct path；否则继续走 sparse gather path。

**Mechanism sketch:**
```cpp
runInfo[mmPingPongIdx].isSmallS2 =
    (curMaxS2 <= actualSelectedBlockCount * selectedBlockSize);

if (runInfo.isSmallS2) {
    DensePath();
} else {
    SparseGatherPath();
}
```

Benefit: 当 selected sparse coverage 已足够覆盖可达范围时，跳过不必要 gather/merge，减少 workspace 压力与额外搬运
Trade-off: 判定依赖 causal horizon 与 selected block coverage 的协同计算，若 coverage 估计不准确，会削弱 dispatch 收益

When to Use: selected sparse block 的覆盖语义清晰，且 `curMaxS2` / `actualSelectedBlockCount` 能稳定表达“有效可达范围是否已被覆盖”

---

## Variant B: `mergeKv` 形式的 reachable-horizon vs TopK-budget 判定
Source: sparse_lightning_indexer_grad_kl_loss_enhance

该实现先计算 `s2SparseLen`（当前 query 在 sparse mode 下可达的 S2 horizon），再与 `kSize` 比较，得到 `mergeKv = (s2SparseLen > kSize)`。这个布尔值不只控制 mm1 的输入来源，还继续传播到 mm2、scatter-add 等后续阶段：
- `mergeKv = true`：先 gather/merge sparse KV 到 workspace，再供多阶段消费
- `mergeKv = false`：直接走 dense GM path，避免 gather workspace 依赖

**Mechanism sketch:**
```cpp
runInfo.s2SparseLen = GetS2SparseLen(...);
runInfo.mergeKv = (runInfo.s2SparseLen > constInfo.kSize);

if (runInfo.mergeKv) {
    ComputeMm1Sparse();
    ProcessScatterAddMergeKv();
} else {
    ComputeMm1Dense();
    ProcessScatterAddNoMergeKv();
}
```

Benefit: 把一次 reachable-horizon 判定复用到多阶段数据流切换，避免每个 stage 单独重算稀疏/稠密决策
Trade-off: 更强依赖 TopK gather 语义，抽象上比 `isSmallS2` 形式更 attention-specific，暂不宜直接泛化为主库正式策略

When to Use: dispatch 决策需要跨多个 stage 复用，且 sparse path 与 dense path 在后续 scatter / reduction 行为上也有分叉

---

## Promotion Gate
仅当满足以下条件时，才考虑从 draft 转正：
1. 再找到至少一个独立 sibling 或非同血统 sparse attention 算子复现相同抽象；
2. 能把判定变量统一抽象为“reachable coverage vs sparse budget”，而不是绑定某个单算子的专属命名；
3. 能证明该机制确实区别于普通 adaptive tiling / multi-algorithm dispatch，而是稳定的 runtime dataflow dual-path selection。
