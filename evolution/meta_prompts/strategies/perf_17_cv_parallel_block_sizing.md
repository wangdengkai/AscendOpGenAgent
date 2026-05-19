# P17: CV Parallel Block Sizing (CV并行基本块大小设计)
## Overview
在 Flash Attention 等需要 Cube 和 Vector 并行执行的融合算子中，合理设计 Cube 基本块（mBaseSize）和 Vector 基本块（s2BaseSize）的大小，使 Cube 的矩阵乘法耗时与 Vector 的 Softmax/Update 耗时尽可能匹配，从而最大化 CV 流水掩盖效果，减少流水线气泡。核心思路是：mBaseSize 控制 M 方向（Query 序列）每次处理的 token 数，s2BaseSize 控制 S2 方向（KV 序列）每次处理的 token 数，两者的乘积决定中间 workspace 大小，两者的比例决定 Cube 与 Vector 的工作量平衡。

## When to Use
- Cube+Vector fused operators requiring CV pipeline overlap (FA, MLA, etc.)
- 算子包含多阶段 CV 交替执行（如 MM1→Softmax→MM2→Update 四阶段流水）
- 需要在 workspace 大小约束下最大化 CV 并行度的场景
- Decode（小 s1）和 Prefill（大 s1）需要不同的基本块配置

## Trade-off
- mBaseSize 过大：Vector 侧 Softmax 工作量增大，Cube 等待 Vector 完成，流水气泡增加
- s2BaseSize 过大：Cube 侧 MatMul 工作量增大，Vector 等待 Cube 完成；同时 workspace 增大
- 基本块过小：每次迭代的启动开销占比增大，跨核同步频率增高，整体吞吐下降

**Source operators**: ai_infra_fused_infer_attention_sink

---

## Variant A: 基于 sInnerSize 反向推导 mBaseSize（非 MLA 场景）
Source: ai_infra_fused_infer_attention_sink (non-MLA)

在非 MLA 的标准 FA 场景中，先根据 KV 序列长度和 GQA group size 确定 s2BaseSize（sInnerSize），再根据 sInnerSize 的对齐值反向推导 mBaseSize。核心约束是 `mBaseSize × s2BaseSize` 不能超过 workspace 上限（mm1ResSize），因此 s2BaseSize 越大，mBaseSize 越小。同时 mBaseSize 需要是 Cube 矩阵乘法的合法 M 维度（通常为 16 的倍数）。

**Expert implementation:**
```cpp
// Host 侧 Tiling: 根据 sInnerSizeAlign 反向推导 mBaseSize
void FiaTilingNonQuant::CalcMBaseSize()
{
    if (fiaInfo_->inputLayout == TilingKeyLayout::TND ||
        fiaInfo_->inputLayout == TilingKeyLayout::NTD) {
        mBaseSize_ = M_BASE_SIZE_512;  // TND/NTD 默认最大
    } else {
        if (fiaInfo_->s1Size <= S1_SIZE_16) {
            // Decode 场景: s1 很小, 根据 sInnerSizeAlign 分档
            if (sInnerSizeAlign_ <= 512)       mBaseSize_ = 512;
            else if (sInnerSizeAlign_ <= 1024) mBaseSize_ = 256;
            else if (sInnerSizeAlign_ <= 2048) mBaseSize_ = 128;
            else if (sInnerSizeAlign_ <= 4096) mBaseSize_ = 64;
            else                               mBaseSize_ = 32;
            // 关键: mBaseSize × sInnerSizeAlign ≈ 常数 (workspace 上限)
        } else {
            mBaseSize_ = M_BASE_SIZE_512;  // Prefill: M 方向给足
        }
    }
}

// s2BaseSize 根据 GQA group size 分档
uint32_t sInnerSize[3U] = {8192U, 4096U, 2048U};
uint32_t idx = std::min(fiaInfo_->gSize / 5U, 2U);
sInnerSize_ = sInnerSize[idx];
// gSize=1~4 → 8192, gSize=5~9 → 4096, gSize≥10 → 2048
```

**vs. baseline (lingxi-code):**
```cpp
// 固定基本块大小, 不考虑 CV 平衡
constexpr uint32_t mBaseSize = 256;
constexpr uint32_t s2BaseSize = 512;
// 问题: Decode 场景 s2 很长时, Cube 工作量远大于 Vector
// 导致 Vector 空等, CV 流水掩盖差
```

Benefit: 通过 mBaseSize 与 s2BaseSize 的反比关系，在不同 shape 下自动平衡 Cube 和 Vector 的工作量，workspace 大小保持在合理范围内，CV 流水掩盖率显著提升。

Trade-off: 分档策略是离散的（32/64/128/256/512），无法做到连续最优；极端 shape 下可能仍有不平衡。

---

## Variant B: 基于负载均衡的 mBaseSize 动态调整（MLA 场景）
Source: ai_infra_fused_infer_attention_sink (MLA)

MLA 场景下 mBaseSize 默认为 256，但当 B×N2 不能整除核数时，会导致分核不均衡，部分核空闲。此时通过计算 B×N2 与核数的最小公倍数，反推出能让 S1G 被均匀切分的 mBaseSize，避免开启 FlashDecode 规约（FD 有额外开销）。

**Expert implementation:**
```cpp
void FiaTilingNonQuantMla::CalcMBaseSize()
{
    uint32_t bN2 = fiaInfo_->bSize * fiaInfo_->n2Size;
    uint32_t s1G = fiaInfo_->s1Size * fiaInfo_->gSize;

    // 若 BN2 能整除核数, 不需要调整
    if (bN2 % aicNum_ == 0U) return;

    // 欧几里得算法求 GCD
    uint32_t a = bN2, b = aicNum_;
    while (b != 0U) { a %= b; std::swap(a, b); }

    // LCM(bN2, aicNum) / bN2 = S1G 需要被切的份数
    uint32_t lcm = (bN2 / a) * aicNum_;
    uint32_t splitOfS1G = lcm / bN2;

    // S1G 必须能被均匀切分
    if (splitOfS1G == 0U || s1G % splitOfS1G != 0U) return;

    uint32_t mBaseSizeTmp = s1G / splitOfS1G;
    // 约束: 128 ≤ mBaseSize ≤ 原始值(256)
    if (mBaseSizeTmp > mBaseSize_ || mBaseSizeTmp < 128U) return;

    mBaseSize_ = mBaseSizeTmp;  // 更新为均衡值
}
```

**vs. baseline (lingxi-code):**
```cpp
// 固定 mBaseSize = 256, 不考虑分核均衡
uint32_t mBaseSize_ = 256;
// 问题: 当 B*N2=5, 核数=24 时, 5 个 head 无法均分到 24 核
// 必须开启 FlashDecode 规约, 引入额外 workspace 和同步开销
```

Benefit: 避免不必要的 FlashDecode 规约开销，所有核负载均衡，减少尾核等待时间。

Trade-off: mBaseSize 可能被缩小到 128，导致单次 Cube 计算量减少，Cube 利用率略降。

---

## Variant C: 4 阶段 CV 流水与 3 深度任务缓存
> **注意**: 本 Variant 的 4 阶段 CV 流水编排机制（PRELOAD_TASK_CACHE_SIZE=3）与 P14 Variant A 相同，详见 [perf_14_cv_pipeline_preload.md](perf_14_cv_pipeline_preload.md)。

Source: ai_infra_fused_infer_attention_sink

FA 的 CV 并行采用 4 阶段流水：MM1(Cube) → Vec1/Softmax(Vector) → MM2(Cube) → Vec2/Update(Vector)。通过 3 深度的任务缓存（FIA_PRELOAD_TASK_CACHE_SIZE=3）实现流水重叠：当前轮 Cube 做 MM1 时，上一轮 Vector 做 Vec1 + Cube 做 MM2，上上轮 Vector 做 Vec2。基本块大小直接决定每个阶段的耗时，从而影响流水掩盖效果。

**Expert implementation:**
```cpp
// Kernel 侧: 3 深度任务缓存实现 CV 4 阶段流水
static constexpr uint32_t FIA_PRELOAD_TASK_CACHE_SIZE = 3;

// 跨核同步事件 ID 分配
static constexpr uint32_t SYNC_V0_C1_FLAG = 6;  // Vec0完成 → Cube可做MM1
static constexpr uint32_t SYNC_C1_V1_FLAG = 7;  // MM1完成 → Vec可做Softmax
static constexpr uint32_t SYNC_V1_C2_FLAG = 8;  // Softmax完成 → Cube可做MM2
static constexpr uint32_t SYNC_C2_V2_FLAG = 9;  // MM2完成 → Vec可做Update

void ExecuteTask(uint64_t loop, RunInfo extraInfo[3])
{
    RunInfo &cur   = extraInfo[loop % 3];       // 本轮
    RunInfo &prev1 = extraInfo[(loop+2) % 3];   // 上一轮
    RunInfo &prev2 = extraInfo[(loop+1) % 3];   // 上上轮

    // 本轮: Cube 做 MM1 (Q × K^T)
    if (cur.isValid) {
        if ASCEND_IS_AIC { ComputeMm1(cur); }
    }
    // 上一轮: Vector 做 Softmax + Cube 做 MM2 (Attn × V)
    if (prev1.isValid) {
        if ASCEND_IS_AIV { ComputeVec1(prev1); }  // Softmax
        if ASCEND_IS_AIC { ComputeMm2(prev1); }   // Attn × V
    }
    // 上上轮: Vector 做 Update/CopyOut
    if (prev2.isValid) {
        if ASCEND_IS_AIV { ComputeVec2(prev2); }  // Rescale + CopyOut
        prev2.isValid = false;
    }
}

// Workspace 通过 GM 传递 CV 中间结果, 大小由基本块决定:
// mm1ResSize = mBaseSize × s2BaseSize  (attention scores)
// mm2ResSize = mBaseSize × headDimAlign (output fragment)
// 每核双缓冲: workspace × PRELOAD_NUM(2) × coreNum
```

**vs. baseline (lingxi-code):**
```cpp
// 串行执行, 无 CV 流水
ComputeMm1(info);       // Cube: Q × K^T
PipeBarrier();
ComputeSoftmax(info);   // Vector: Softmax
PipeBarrier();
ComputeMm2(info);       // Cube: Attn × V
PipeBarrier();
ComputeUpdate(info);    // Vector: Rescale + CopyOut
// 问题: 每个阶段串行等待, Cube 和 Vector 交替空闲
```

Benefit: 3 深度流水使得 Cube 和 Vector 几乎全程并行，理论上可将 CV 交替执行的延迟掩盖 2/3 以上。基本块大小的合理设计确保每个阶段耗时接近，最大化掩盖效果。

Trade-off: 需要 3 倍的 RunInfo 缓存和 2 倍的 workspace（双缓冲），GM workspace 总量 = 2 × coreNum × (mm1ResSize + vec1ResSize + mm2ResSize + vec2ResSize)。基本块越大，workspace 越大。
