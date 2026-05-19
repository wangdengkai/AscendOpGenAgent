# P53 L1 常驻复用策略 (L1 Resident Reuse)
## Overview
在 FlashAttention/IFA 等 Cube 密集型算子中，将 Q 矩阵（左矩阵）在首次 N 方向迭代时加载到 L1 后常驻，后续 N 迭代直接复用 L1 中已有数据，消除冗余搬运。分为"大常驻"（跨多个 S2 循环复用）和"小常驻"（单基本块内部复用）两种模式。该策略与 P17 (L1 7-buffer resident partition) 互补，P17 侧重多 buffer 分区管理，本策略侧重单 buffer 的跨迭代复用逻辑。

## When to Use
- Cube 密集型算子中 Q 矩阵在多轮 KV 迭代中保持不变，复用率高
- L1 总容量足够容纳常驻 Q 数据（如 144KB for M=128, D=576）
- headDim 较大（≥288），标准双缓冲无法充分利用 L1 容量
- 当 L1 余量 > Q 的 L1 内存需求时支持大常驻，否则使用小常驻

## Trade-off
- 大常驻期间 L1 空间被 Q 占用，BMM2 阶段无法释放供 P 使用
- M 方向 > 128 时不支持大常驻（128×576×2 = 144KB）
- C1 和 C2 的首次循环需要加载的数据量大，可能造成断流

**Source operators**: IFA, PFA, MLA 场景

---

## Variant A: L1 大常驻（跨 S2 循环复用）
Source: 【基础知识】L1内存复用：L1常驻减少内存重复搬运耗时.md

在 NBS1S2 循环的 S2 循环中，当 reuseLeft 标志为 1 时（非首轮），直接使用首轮暂存在 L1 的 Q 数据，避免重复搬运。

**Expert implementation:**
```cpp
__aicore__ inline LocalTensor<TRANS_T> LoadData(int curRow, int curCol,
    int tileHeight, int tileWidth, int batchNum = -1) {
    LocalTensor<TRANS_T> l1;
    UserDefDataType flag = MATMUL_PARAM_VAR.dataPtr_;
    LocalTensor<TRANS_T> dst;
    if (flag.reuseLeft) {
        // 非首轮：直接复用 L1 中已有的 Q 数据
        l1 = MATMUL_MODULE(CubeInBuffer)->GetBuffer(flag.leftBufIdx);
        dst = l1[static_cast<int64_t>(callTimes_) * baseWidth_ * 64];
        ++callTimes_;
        return dst;
    } else {
        // 首轮：从 GM 加载 Q 数据到 L1
        l1 = MATMUL_MODULE(CubeInBuffer)->AllocTensor(flag.leftBufIdx);
        dst = l1[static_cast<int64_t>(callTimes_) * baseWidth_ * 64];
        // ... 执行 DataCopy nd2nz ...
    }
    ++callTimes_;
    return dst;
}
```

**vs. baseline (lingxi-code):**
```cpp
// 基线：每次 S2 迭代都重新加载 Q
for (uint32_t n = 0; n < nloops; n++) {
    CopyQToL1(info, mL1Size);  // 每次 N 迭代都搬运
    // ... 计算 ...
}
```

Benefit: 消除 N 方向内循环中 Q 数据的冗余 GM→L1 搬运，搬运次数从 O(N_loops) 降为 O(1)；对于 s2BaseSize=512、N_L1_SPLIT=128 的典型配置，Q 搬运量减少 75%
Trade-off: L1 空间被 Q 常驻占用，BMM2 阶段 L1 余量减少

---

## Variant B: L1 小常驻（单基本块内部复用）
Source: 【基础知识】L1内存复用：L1常驻减少内存重复搬运耗时.md

在单个基本块的 BMM1 运算内部，K 轴切分循环开始前仅拷贝一次 Q 到 L1，后续 K 切分循环中复用该 L1 数据。

**Expert implementation:**
```cpp
template <CubeFormat OutFormat>
__aicore__ inline void PfaMatmulKvNd<INPUT_T, T>::ComputeMm1(const SplitSameABExtraInfo &info) {
    // query 的 L1 小常驻：在 N 循环外拷贝一次
    auto qTensor = CopyQToL1(info, mL1Size);

    for (uint32_t n = 0; n < nloops; n++) {
        // K 的预取和搬运
        if (!l1BufLoaded[kvL1BufIter % 2]) {
            CopyKToL1(info, n * nSplitSize, subNSizeAct);
        }
        // ... 后续计算使用 qTensor，不再重新拷贝 Q ...
    }
}
```

**vs. baseline (lingxi-code):**
```cpp
// 基线：每个 K 切分块都重新加载 Q
for (uint32_t k = 0; k < kloops; k++) {
    CopyQToL1(info, mL1Size);  // 每次 K 迭代都搬运
    CopyKToL1(info, ...);
    MMAD();
}
```

Benefit: 消除单基本块内部 K 轴循环的 Q 重复搬运，适用于 L1 余量较小无法支持大常驻的场景
Trade-off: 仅在单基本块内有效，跨 S2 循环仍需重新加载

---

## Variant C: Load3Dv2 实现 Q 常驻搬运
Source: 【案例总结】DeepSeek V3网络IFA性能优化.md

使用 Load3Dv2 指令替代普通 DataCopy 实现 Q 矩阵从 GM 到 L1 的高效搬运，配合常驻策略进一步提升性能。

**Expert implementation:**
```cpp
// Load3Dv2 实现 Q 的 L1 常驻搬运
// 收益：约 15us 优化
Load3Dv2Params loadParams;
loadParams.nValue = mSize;
loadParams.dValue = kSize;
loadParams.srcDValue = orgWidth;
loadParams.dstNzC0Stride = Align16Func(mSize);
LoadDataToL1<enableZZ>(qL1Tensor, queryGm, loadParams);
```

Benefit: Load3Dv2 比 DataCopy nd2nz 更高效，配合常驻策略进一步提升 GM→L1 搬运效率
Trade-off: 需要硬件支持 Load3Dv2 指令