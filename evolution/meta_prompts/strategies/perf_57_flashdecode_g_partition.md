# P57 FlashDecode G 分核归约 (FlashDecode G-Partition Reduction)
## Overview
在 FlashDecode 场景下，当 B 和 N2 都很小而 G 很大时，传统归约方案只使用很少的核数。通过对 G 轴进行分核，实现多核并行归约，充分利用硬件资源。

## When to Use
- FlashDecode 场景，batchSize * kvHeadNum 小于总核数的一半
- G（query_N / kv_N）较大，可以进一步切分
- 需要充分利用多核并行能力

## Trade-off
- 增加归约复杂度，需要多核协作
- G 切分粒度过小（≤4）可能导致性能劣化
- 需要额外的 workspace 存储中间结果

**Source operators**: IFA FlashDecode 场景

---

## Variant A: G 分核归约策略
Source: 【案例总结】FlashDecode归约分核优化.md

当 batchSize * kvHeadNum ≤ 阈值时，对 G 轴进行分核，每个核处理一部分 G，最后进行归约合并。

**Expert implementation:**
```cpp
// G 分核逻辑
if (batchSize * kvHeadNum <= 24) {
    uint32_t coreNumPerGD = 48 / (batchSize * kvHeadNum);
    uint32_t gSizeMin = 16;  // G 最小切分大小，避免过小导致劣化
    uint32_t maxCoreNumPerGD = (gSize + gSizeMin - 1) / gSizeMin;
    coreNumPerGD = (coreNumPerGD > maxCoreNumPerGD) ? maxCoreNumPerGD : coreNumPerGD;
    gSizeSub = (gSize + coreNumPerGD - 1) / coreNumPerGD;

    // 修正核数
    uint32_t coreNumPerGDFix = (gSize + gSizeSub - 1) / gSizeSub;
    coreNumPerGD = (coreNumPerGD > coreNumPerGDFix) ? coreNumPerGDFix : coreNumPerGD;

    if (tmpBlockIdx >= batchSize * kvHeadNum * coreNumPerGD) {
        return;  // 超出分配核数，直接返回
    }

    gSizeTail = gSize - (coreNumPerGD - 1) * gSizeSub;
    gOuter = coreNumPerGD;
    gSizeStart = 0;
    FlashDecodeComputeSplitG();
    return;
}
```

**vs. baseline (lingxi-code):**
```cpp
// 基线：仅按 BN2 分核，G 大时核利用率低
uint32_t usedCores = batchSize * kvHeadNum;  // 可能远小于总核数
```

Benefit: 充分利用多核资源，提升并行度；G 切分最小 16 为经验最优值
Trade-off: 增加归约复杂度；G 切分 ≤4 时可能性能劣化

---

## Variant B: G 分核归约计算流程
Source: 【案例总结】FlashDecode归约分核优化.md

每个核计算自己的 GD 块后，根据 softmaxSum 和 softmaxMax 计算 scale 系数，修正后累加得到最终输出。

**Expert implementation:**
```cpp
// G 分核归约计算
void FlashDecodeComputeSplitG() {
    // 1. 计算 scale 系数
    scale[0:splits] = exp(lse[i]) / Sum(exp(lse[i]));  // i [0, splits)

    // 2. 加载各核的局部结果
    res = {0};
    split_res = load_split_res();

    // 3. 加权累加
    for (j = 0; j < splits; j++) {
        res += split_res[j] * scale[j];
    }

    // 4. 输出最终结果
    output(res);
}
```

Benefit: 实现多核并行归约，充分利用硬件资源
Trade-off: 需要存储中间 lse 和 split_res，增加 workspace 占用