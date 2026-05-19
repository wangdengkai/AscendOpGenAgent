# P61 L2 Cache 优化 (L2 Cache Optimization)
## Overview
通过关闭双页表功能（开启 KV 的 L2 Cache），减少 GM 访问延迟。在 PageAttention 场景下，KV 数据通过 L2 Cache 缓存，显著提升数据复用效率。

## When to Use
- PageAttention 场景，KV 数据访问频繁
- 多轮迭代复用同一 KV 数据
- L2 Cache 空间足够容纳热数据

## Trade-off
- 需要硬件支持 L2 Cache 功能
- 可能影响其他算子的 L2 Cache 使用
- 需要权衡 L2 Cache 与双页表功能

**Source operators**: IFA PageAttention 场景

---

## Variant A: 关闭双页表开启 L2 Cache
Source: 【案例总结】DeepSeek V3网络IFA性能优化.md

通过关闭 l2CacheOffFlag，使 KV 数据能够进入 L2 Cache，减少 GM 访问延迟。

**Expert implementation:**
```cpp
// 关闭双页表功能，开启 KV 的 L2 Cache
// 修改位置：ops_adv 代码

// 原始配置：l2CacheOffFlag = true
// KV 数据直接访问 GM，无 L2 缓存

// 优化配置：l2CacheOffFlag = false
// KV 数据进入 L2 Cache，后续访问命中缓存

// 性能收益
// 250us → 210us（约 16% 提升）
```

**vs. baseline (lingxi-code):**
```cpp
// 基线：双页表开启，KV 不走 L2 Cache
// 每次 KV 访问都需要 GM 读取
```

Benefit: 开启 L2 Cache 后，KV 数据复用率提升，GM 访问延迟降低
Trade-off: 关闭双页表可能影响其他功能；需要硬件支持

---

## Variant B: L2 Cache 复用策略
Source: 【案例总结】DeepSeek V3网络IFA性能优化.md

在多轮 KV 迭代中，利用 L2 Cache 缓存热数据，减少重复 GM 访问。

**Expert implementation:**
```cpp
// L2 Cache 复用策略
// 1. 首轮迭代：KV 数据从 GM 加载到 L2
// 2. 后续迭代：KV 数据从 L2 命中，延迟降低

// 分块策略优化 L2 命中率
for (uint32_t s2 = 0; s2 < s2Loops; s2++) {
    // 相邻 s2 块访问相邻 KV 数据，L2 局部性好
    ProcessS2Block(s2 * s2Size, s2Size);
}

// 核间交错执行提升 L2 复用
// 相邻核处理相邻 S2 块，共享 L2 中的 KV 数据
```

Benefit: 提升核间 L2 Cache 复用率；减少 GM 带宽压力
Trade-off: 需要调整分块和分核策略

---

## Variant C: L2 Cache 预取
Source: 【案例总结】DeepSeek V3网络IFA性能优化.md

在计算当前块时，预取下一块的 KV 数据到 L2 Cache，隐藏访存延迟。

**Expert implementation:**
```cpp
// L2 预取策略
// 当前块计算时，预取下一块 KV

void ProcessBlock(uint32_t blockIdx) {
    // 计算当前块
    Compute(blockIdx);
    
    // 预取下一块 KV 到 L2
    if (blockIdx + 1 < totalBlocks) {
        PrefetchKVToL2((blockIdx + 1) * blockSize, blockSize);
    }
}

// 注意：TilingData 预取收益有限（COPY_TILING 本身 <1us）
// 重点优化 KV 数据预取
```

Benefit: 隐藏 KV 访存延迟，提升计算流水效率
Trade-off: 预取指令有额外开销；需要精确控制预取时机

---

## Variant D: Matmul L2 Cache 切分与错位分核
Source: 优秀实践/Matmul性能调优案例/Matmul高阶API使能L2_Cache切分.md

将 Matmul 数据切分为适合 L2 Cache 容量的分块，配合错位分核（对角线映射）策略，使相邻核处理相邻数据块，提升 L2 Cache 命中率。通过 LCM 计算对角线映射关系。

**Expert implementation:**
```cpp
// L2 Cache 切分判断
if (GetTotalSize() <= L2_TILE_THRESHOLD) {
    EnableL2CacheSplit();
}

// 错位分核：对角线映射
void GetBlockCoord(uint32_t blockIdx, uint32_t &mIdx, uint32_t &nIdx) {
    uint32_t lcm = LCM(mBlocks, nBlocks);
    uint32_t diag = blockIdx % lcm;
    mIdx = diag / nBlocks;
    nIdx = (diag % nBlocks + mIdx) % nBlocks;  // 对角线偏移
}
```

**vs. baseline (lingxi-code):**
```cpp
// 基线：线性分核，相邻核可能访问不同 L2 Cache 行
```

Benefit: 提升 L2 Cache 命中率，减少 GM 访问次数
Trade-off: 错位分核增加 Tiling 计算复杂度；需要精确计算 L2 Cache 容量阈值