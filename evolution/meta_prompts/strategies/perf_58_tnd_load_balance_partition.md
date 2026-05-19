# P58 TND 负载均衡分核 (TND Load-Balanced Partition)
## Overview
在 TND（T=B*S 合轴）场景下，通过线段抽象和贪心算法实现跨核负载均衡分配。将变长序列抽象为线段，按线段总长度均分到各核，避免传统 BS 切分方式导致的负载不均衡问题。支持 FlashDecode 归约机制处理跨核切分。

## When to Use
- TND 排布格式，各 batch 的序列长度不一致
- 传统 BS 切分导致某些核负载过重成为瓶颈
- 需要充分利用多核并行能力

## Trade-off
- 需要额外的 workspace 存储规约信息（lseMax/lseSum/accumOut）
- 线段被跨核切分时需要 FlashDecode 归约
- tiling 逻辑复杂度增加

**Source operators**: IFA TND 场景

---

## Variant A: 线段抽象与均分算法
Source: 【案例总结】DSV3 TND分核方案.md

将各 batch 的 KVS 负载抽象为线段，S1 方向按 128 切，S2 方向按 512 切，线段长度表示 KVS/512。按线段总长度均分到各核。

**Expert implementation:**
```cpp
// 线段抽象与均分
// S1方向切分：QS * G <= 128
// S2方向切分：KVS = 512 为基本单位

// 计算线段总长度
uint32_t totalS2Length = 0;
for (uint32_t b = 0; b < batchSize; b++) {
    totalS2Length += (actualKVS[b] + 511) / 512;  // 向上取整
}

// 平均每个核需要计算的线段数目
avgS2Length = (totalS2Length + coreNum_ - 1) / coreNum_;  // 向上取整

// 分核时，记录跨核切分信息用于 FlashDecode 归约
for (each segment) {
    if (segment crosses core boundary) {
        recordSplitInfo(b, s1Outer, kvSplitNum);
    }
}
```

**vs. baseline (lingxi-code):**
```cpp
// 基线：按 BS 切分，不考虑实际序列长度
for (uint32_t b = 0; b < batchSize; b++) {
    assignToCore(b, coreIdx);  // 短序列和长序列可能分到同一核
}
```

Benefit: 实现真正的负载均衡，避免长序列导致的核间负载差异；支持变长序列的高效并行
Trade-off: 需要维护 actual sequence 信息；分核逻辑复杂度增加

---

## Variant B: FlashDecode 归约支持
Source: 【案例总结】DSV3 TND分核方案.md

当线段被跨核切分时，需要保存局部 sum/max 结果并进行核间归约。每个核最多存储两份规约信息（头规约和尾规约）。

**Expert implementation:**
```cpp
// Workspace 空间排布
// lseMaxFdGm/lseSumFdGm: coreNum * 2 * kvHeadNum * 128
// accumOutGm: coreNum * 2 * kvHeadNum * 128 * headDim

// 规约信息记录
struct FDSplitInfo {
    uint32_t balanceFDCoreBArr[taskId];      // 当前 task 的 B
    uint32_t balanceFDCoreS1Arr[taskId];     // 当前 task 的 S1
    uint32_t balanceFDCoreKVSplitArr[taskId]; // 被切分的份数
    uint32_t balanceFDCoreStartKVSplitNum[currCoreIdx]; // 当前核是第几份
};

// 判断是否需要规约
if (currCore包含线段的非最后部分) {
    // 尾规约：需要和后面的核一起计算
    saveTailReductionInfo();
}
if (currCore包含线段的非最开始部分) {
    // 头规约：需要和前面的核一起计算
    saveHeadReductionInfo();
}
```

Benefit: 支持跨核切分场景的正确归约；每个核最多存储两份规约信息，workspace 占用可控
Trade-off: 最大 workspace 需求约 12MB（24核 * 2 * 128 * 512 * 4B）

---

## Variant C: 交错执行提升 L2 复用
Source: 【案例总结】DSV3 TND分核方案.md

按核数分核 + S2 分块交错执行，提高 L2 Cache 复用率。不同核处理相邻 S2 块，共享 KV 数据。

**Expert implementation:**
```cpp
// 交错执行：相邻核处理相邻 S2 块
for (uint32_t core = 0; core < coreNum; core++) {
    uint32_t s2Start = core * avgS2Length * 512;
    uint32_t s2End = min(s2Start + avgS2Length * 512, totalKVS);
    // 相邻核的 S2 范围相邻，L2 中 KV 数据可复用
}
```

Benefit: 提升核间 L2 Cache 复用率，减少 GM 访问
Trade-off: 需要调整分核策略，可能增加规约复杂度