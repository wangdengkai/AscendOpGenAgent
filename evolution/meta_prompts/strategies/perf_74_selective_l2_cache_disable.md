# P74 L2 Cache 选择性禁用 (Selective L2 Cache Disable)
## Overview
在 KV cache 数据量远大于 L2 容量的推理场景中，通过 `SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE)` 选择性禁用 L2 Cache，避免 L2 cache thrashing 导致的性能抖动。Host 端 tiling 根据 KV layout 和数据总量动态决定是否禁用。

## When to Use
- KV cache 数据总量超过 L2 容量的 1.2 倍（长序列、大 batch 推理）
- KV layout 为 BNSD 或 BnNBsD（连续访存，不依赖 L2 预取）
- PageAttention 场景下 KV 访问模式不规则

## Trade-off
- 禁用后所有 DMA 走 DDR→L1 路径，短序列场景可能损失 L2 命中收益
- 需要 Host 端准确估算 KV 数据总量与 L2 容量的比值
- 与 P47 (L2 Cache 优化) 互补：P47 侧重利用 L2，本策略侧重规避 L2 thrashing

**Source operators**: ai_infra_sparse_flash_attention_gqa

---
## Variant A: Host 端 L2 Cache 禁用判断
Source: ai_infra_sparse_flash_attention_gqa op_host

Host 端 tiling 中 `GetL2CacheOffFlag()` 根据 KV layout 和数据量决定是否禁用 L2 Cache。

**Expert implementation:**
```cpp
// Host 端判断逻辑
bool GetL2CacheOffFlag() {
    // 条件1: BNSD/BnNBsD layout 连续访存，不需要 L2 预取
    if (kvLayout == BNSD || kvLayout == BnNBsD) {
        return true;  // 禁用 L2
    }
    // 条件2: KV 数据总量超过 L2 容量的 1.2 倍
    uint64_t kvDataSize = batchSize * kvHeadNum * seqLen * headDim * sizeof(half);
    if (kvDataSize > L2_CAPACITY * 1.2) {
        return true;  // 禁用 L2，避免 thrashing
    }
    return false;
}
```

**vs. baseline (lingxi-code):**
```cpp
// 基线：默认启用 L2 Cache，长序列场景可能 thrashing
// 无 L2 Cache 控制逻辑
```

Benefit: 避免 L2 cache thrashing 导致的性能抖动；减少 L2 cache line 替换开销
Trade-off: 禁用后短序列场景可能损失 L2 命中收益

---
## Variant B: Kernel 端 L2 Cache Hint 设置
Source: ai_infra_sparse_flash_attention_gqa op_kernel

Kernel 端在 K/V 数据搬运前设置 L2 Cache Hint。

**Expert implementation:**
```cpp
// Kernel 端设置 L2 Cache Hint
if (l2CacheOff) {
    SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);  // K 搬运
}
DataCopy(keyL1, keyGm, copyParams);

if (l2CacheOff) {
    SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);  // V 搬运
}
DataCopy(valueL1, valueGm, copyParams);
```

Benefit: 精确控制每次 DMA 搬运的 L2 Cache 行为
Trade-off: 需要在每次搬运前设置 Hint，增加少量指令开销
