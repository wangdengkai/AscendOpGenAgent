# P60 KV NZ 格式优化 (KV NZ Format Optimization)
## Overview
将 KV 矩阵从 ND 格式改为 NZ（Channel-first）格式，使数据布局与 Cube 计算的 L0 输入格式一致，消除 MM1/MM2 阶段的格式转换开销，提升整体性能。

## When to Use
- Cube 密集型算子中 KV 矩阵需要频繁 ND→NZ 转换
- PageAttention 或 MLA 场景，KV 数据复用率高
- 追求极致性能，可接受额外的内存布局适配

## Trade-off
- 需要 KV 数据以 NZ 格式存储，增加上游数据准备复杂度
- 非对齐场景处理复杂
- 与现有 ND 格式算子不兼容

**Source operators**: IFA MLA 场景

---

## Variant A: KV NZ 格式消除格式转换
Source: 【案例总结】DeepSeek V3网络IFA性能优化.md

将 KV 数据存储为 NZ 格式，使 MM1/MM2 的 LoadB（KV→L1）无需格式转换，直接加载到 L0。

**Expert implementation:**
```cpp
// KV NZ 格式：[N, D/16, S/16, 16, 16]
// 与 Cube L0B 格式一致，无需 nd2nz 转换

// 原始 ND 格式：需要 DataCopy nd2nz
// LoadB 阶段：DataCopy(nd2nz) → L1 → LoadB → L0B

// NZ 格式优化：直接加载
// LoadB 阶段：DataCopy(连续) → L1 → LoadB → L0B（无转换）

// 性能收益
// 未叠加双页表：250us → 213us
// 叠加双页表：210us → 178us
```

**vs. baseline (lingxi-code):**
```cpp
// 基线：KV 为 ND 格式，需要 nd2nz 转换
LoadDataToL1(kvL1Tensor, kvGm, nd2nzParams);  // 格式转换开销
```

Benefit: 消除 KV 的 ND→NZ 格式转换开销，显著提升 MM1/MM2 性能
Trade-off: 需要 KV 数据以 NZ 格式存储；上游算子需要适配

---

## Variant B: MM1 输出 NZ 格式
Source: 【案例总结】DeepSeek V3网络IFA性能优化.md

MM1 输出保持 NZ 格式，便于 Vec2 阶段的搬运和计算逻辑复用。

**Expert implementation:**
```cpp
// MM1 输出 NZ 格式：[G/16, S2/16, 16, 16]
// 与 Softmax 输入格式匹配，减少格式转换

// MM1 NZ 模板配置
// baseM=128, baseN=128, baseK=128
// stepM=1, stepN=1, stepK=2
// S_L1=128, D_L1=256, D_L0=128, S_L0=128

for (dLoop_L1 = 0; dLoop_L1 < HeadDim/D_L1; dLoop_L1++) {
    CopyQueryToL1(G, StartD, D_L1);
    if (dLoop_L1 != 0) {
        SetAtomicAdd<float>();  // 首轮不累加
    }
    for (sLoop = 0; sLoop < S2/S_L1; sLoop++) {
        CopyKeyToL1(StartS, S_L1, StartD, D_L1);
        for (dLoop_L0 = 0; dLoop_L0 < D_L1/D_L0; dLoop_L0++) {
            LoadA(G, StartD, D_L0);
            LoadB(S_L1, StartD, D_L0);
            Mmad();  // 累加条件：dLoop == 0
            pipe_biarrier(PIPE_M);
        }
        Fixpipe();  // GM 累加条件：dLoop_L1 != 0
    }
}
```

Benefit: MM1 输出 NZ 格式便于后续处理；减少格式转换次数
Trade-off: 需要 NZ 模板支持；Vec2 阶段需要适配 NZ 输入

---

## Variant C: NZ 模板支持 AMLA 算法
Source: 【案例总结】DeepSeek V3网络IFA性能优化.md

在 NZ 模板基础上支持 AMLA（Attention Multi-Head Latent Attention）算法，通过编译宏传递 isAMla 参数。

**Expert implementation:**
```cpp
// Tiling 侧判断是否启用 AMLA
bool IFATiling::IsEnableAMla() {
    if ((qSeqSize == 1) && (kvSplit == 0) && 
        (blockSize == 128) && (nNumOfQInOneGroup % 16 == 0)) {
        return true;
    }
    return false;
}

// Kernel 侧模板参数
template <typename Q_T, typename KV_T, typename OUT_T, typename ORIGIN_T,
          const bool PAGE_ATTENTION = false,
          const bool FLASH_DECODE = false,
          LAYOUT LAYOUT_T = LAYOUT::BSH,
          const uint8_t ANTIQUANT_MODE = 0,
          const bool SHARED_PREFIX = false,
          LAYOUT KV_LAYOUT_T = LAYOUT::BSH,
          const bool AMLA = false,  // AMLA 开关
          typename... Args>
struct IFAType {
    static constexpr bool isAMla = AMLA;
};

// 性能数据
// 普通 NZ：302.75us
// NZ + AMLA + L0AB 共用 EventId：333.21us（额外优化后降至 270us）
```

Benefit: NZ 模板复用 AMLA 算法优化；通过编译宏实现零开销分支
Trade-off: AMLA 有特定条件限制（qSeqSize=1, blockSize=128 等）