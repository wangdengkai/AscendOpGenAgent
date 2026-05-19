# P16: Cost-Driven Core Partition (基于代价模型的多粒度分核优化)
## Overview
与简单的按元素/行数均分不同，本策略通过构建计算代价模型（考虑对齐开销、轴权重等硬件特性），对每个计算块评估实际代价，然后采用多粒度层次化分配（Batch→Row→Block）将负载按代价均衡地分配到各核。同时支持Mask/稀疏感知跳过无效块、最优核数枚举搜索、容忍度机制避免碎片化、以及跨核归约(FD)的独立负载均衡。适用于计算量在不同块之间差异大、存在稀疏/Mask模式的复杂算子（如Flash Attention）。

## When to Use
- Complex operators with irregular block costs (如 FA、Attention 类多维算子)
- 存在 mask/稀疏模式导致不同核的有效计算量差异大的场景
- 多维数据（Batch × N × S1 × S2）需要跨维度联合分核的算子
- 简单均分导致核间负载严重不均衡，需要精细化代价评估的场景

## Trade-off
- Host 端 Tiling 计算复杂度显著增加，需要枚举核数 + 多级分配
- 需要维护代价模型参数（对齐系数、轴权重），模型不准会导致分核效果退化
- 代码量和调试难度大幅上升，适合高频调用的核心算子，不适合简单算子

**Source operators**: flash_attention

---

## Variant A: 对齐感知的代价模型
Source: flash_attention

构建基于硬件对齐特性的计算代价模型。对M轴按16对齐、S2轴按64对齐后，分别乘以权重系数（M轴系数6，S2轴系数10）得到每个block的代价。同时预计算4种block类型（Normal×Normal、Tail×Normal、Normal×Tail、Tail×Tail）的代价表，避免运行时重复计算。

**Expert implementation:**
```cpp
int64_t CalcCost(uint32_t basicM, uint32_t basicS2)
{
    uint32_t alignCoefM = 16U;
    uint32_t alignCoefS2 = 64U;
    uint32_t alignBasicM = (basicM + alignCoefM - 1U) >> 4U;
    uint32_t alignBasicS2 = (basicS2 + alignCoefS2 - 1U) >> 6U;
    return static_cast<int64_t>(6U * alignBasicM + 10U * alignBasicS2);
}na

BlockCost<int64_t> CalcCostTable(uint32_t s1NormalSize, uint32_t s2NormalSize,
    uint32_t s1GTailSize, uint32_t s2TailSize)
{
    BlockCost<int64_t> typeCost {};
    typeCost[NORMAL_BLOCK][NORMAL_BLOCK] = CalcCost(s1NormalSize, s2NormalSize);
    typeCost[TAIL_BLOCK][NORMAL_BLOCK] = (s1GTailSize == 0U) ? 0U : CalcCost(s1GTailSize, s2NormalSize);
    typeCost[NORMAL_BLOCK][TAIL_BLOCK] = (s2TailSize == 0U) ? 0U : CalcCost(s1NormalSize, s2TailSize);
    typeCost[TAIL_BLOCK][TAIL_BLOCK] = (s1GTailSize == 0U || s2TailSize == 0U) ?
        0U : CalcCost(s1GTailSize, s2TailSize);
    return typeCost;
}
```

**vs. baseline (lingxi-code):**
```cpp
// baseline: 按元素数量均分，不考虑对齐和计算代价差异
uint32_t elementsPerCore = totalElements / coreNum;
```

Benefit: 精确反映硬件实际计算开销，尾块代价自动降低，避免按数量均分导致的隐性不均衡
Trade-off: 需要根据硬件特性调整对齐系数和权重参数

---

## Variant B: 三级层次化分配（Batch→Row→Block）
Source: flash_attention

采用由粗到细的三级分配策略：第一级按整Batch分配（代价最低的调度粒度），第二级按行(S1G)分配（中等粒度），第三级按Block分配（最细粒度）。每级分配都基于容忍度判断是否继续装入当前核。若三级分配后当前核仍为空，则强制分配一个Block，保证每个活跃核至少有工作。

**Expert implementation:**
```cpp
void CalcSplitPlan(uint32_t coreNum, int64_t costLimit,
    const SplitContext &splitContext, SplitResult &result)
{
    for (uint32_t i = 0; i < coreNum; ++i) {
        assignContext.coreCache = {};
        assignContext.coreCache.costLimit =
            assignContext.unassignedCost / (coreNum - assignContext.curCoreIdx);

        // 1、按整batch分配
        AssignByBatch(splitContext, assignContext);
        // 2、按行分配
        AssignByRow(splitContext, assignContext);
        if (baseInfo.supportFD) {
            // 3、按块分配
            AssignByBlock(splitContext, assignContext);
            // 4、强制分配（保证至少1块）
            if (assignContext.coreCache.block == 0) {
                ForceAssign(splitContext, assignContext);
            }
        }
        result.maxCost = std::max(result.maxCost, assignContext.coreCache.cost);
        assignContext.unassignedCost -= assignContext.coreCache.cost;
    }
}
```

**vs. baseline (lingxi-code):**
```cpp
// baseline: 单粒度均分
uint32_t rowsPerCore = totalRows / coreNum;
uint32_t tailRows = totalRows - rowsPerCore * (coreNum - 1);
```

Benefit: 粗粒度优先减少调度开销，细粒度兜底保证均衡度；动态 costLimit 随剩余负载自适应调整
Trade-off: 多级分配逻辑复杂，需要维护游标状态（curBN2Idx/curS1GIdx/curS2Idx）

---

## Variant C: Mask/稀疏感知的有效范围计算
Source: flash_attention

根据 attention mask 的稀疏模式（causal、band 等），计算每行(S1G)在 S2 方向的有效 block 范围，跳过被 mask 掉的无效块。通过 preToken/nextToken 参数确定有效 token 范围，再转换为 block 索引。无效行（s2Start >= s2End）的代价直接置零，不参与分核。

**Expert implementation:**
```cpp
Range<uint32_t> CalcS2Range(uint32_t s1GIdx, const BaseInfo &baseInfo,
    const SplitParam &splitParam, const BatchCache &batchCache)
{
    // no mask: 全范围有效
    if (!baseInfo.attenMaskFlag) {
        s2Start = 0U;
        s2End = (batchCache.s2Size + splitParam.s2BaseSize - 1U) / splitParam.s2BaseSize;
        return std::make_pair(s2Start, s2End);
    }

    // 根据 S1G 行索引计算对应的 S2 有效 token 范围
    int64_t s2FirstToken = s1FirstToken - batchCache.preTokenLeftUp;
    int64_t s2LastToken = s1LastToken + batchCache.nextTokenLeftUp;

    // 无有效 token
    if (s2FirstToken >= batchCache.s2Size || s2LastToken < 0) {
        return std::make_pair(0U, 0U);
    }

    // 裁剪到有效范围并转换为 block 索引
    s2FirstToken = Clip(s2FirstToken, 0LL, (int64_t)(batchCache.s2Size - 1U));
    s2LastToken = Clip(s2LastToken, 0LL, (int64_t)(batchCache.s2Size - 1U));
    s2Start = (uint32_t)s2FirstToken / splitParam.s2BaseSize;
    s2End = (uint32_t)s2LastToken / splitParam.s2BaseSize + 1U;
    return std::make_pair(s2Start, s2End);
}
```

**vs. baseline (lingxi-code):**
```cpp
// baseline: 不感知 mask，所有核处理全量 S2 范围
uint32_t s2Blocks = (s2Size + blockSize - 1) / blockSize;
uint32_t blocksPerCore = s2Blocks / coreNum;
```

Benefit: 避免核在无效块上浪费计算，causal mask 场景下可减少近50%的无效计算
Trade-off: 需要在 Host 端预计算每行的有效范围，增加 Tiling 耗时

---

## Variant D: 最优核数枚举搜索
Source: flash_attention

不固定使用全部核，而是从 sqrt(totalBlocks) 到 min(coreNum, totalBlocks) 枚举不同核数，对每种核数执行完整的分配方案计算，选择 maxCost（最慢核的代价）最小的方案。搜索过程中利用当前最优 maxCost 作为剪枝上界，提前终止劣质方案。

**Expert implementation:**
```cpp
void SplitCore(uint32_t coreNum, const BaseInfo &baseInfo,
    const SplitParam &param, SplitResult &result)
{
    uint32_t maxCore = std::min(coreNum, splitContext.costInfo.totalBlockNum);
    uint32_t minCore = static_cast<uint32_t>(
        std::sqrt(static_cast<float>(splitContext.costInfo.totalBlockNum) + 0.25f) + 0.5f);
    minCore = std::min(minCore, maxCore);

    result.maxCost = INT64_MAX;
    SplitResult tmpResult {coreNum, result.vecCubeRatio};
    for (uint32_t i = minCore; i <= maxCore; ++i) {
        CalcSplitPlan(i, result.maxCost, splitContext, tmpResult);  // maxCost 作为剪枝上界
        if (tmpResult.maxCost < result.maxCost) {
            CopyTmpResult(tmpResult, result);
        }
        ClearTmpResult(tmpResult);
    }
}
```

**vs. baseline (lingxi-code):**
```cpp
// baseline: 固定使用全部核或硬编码核数
const uint32_t BLOCK_DIM = 32;
context->SetBlockDim(BLOCK_DIM);
```

Benefit: 自动找到最优核数，避免核数过多导致的调度开销或核数过少导致的并行不足
Trade-off: 枚举搜索增加 Host 端耗时，搜索范围为 O(sqrt(N)) 到 O(N)

---

## Variant E: 容忍度机制与动态代价上限
Source: flash_attention

每个核的代价上限 costLimit 动态计算为"剩余总代价 / 剩余核数"，随分配过程自适应调整。分配时使用容忍度比率 FA_TOLERANCE_RATIO 判断是否继续装入：只要当前核的剩余容量大于下一个块代价的 1/FA_TOLERANCE_RATIO，就继续装入。这避免了严格均分导致的碎片化问题。

**Expert implementation:**
```cpp
// 动态代价上限
assignContext.coreCache.costLimit =
    assignContext.unassignedCost / (coreNum - assignContext.curCoreIdx);

// 容忍度判断：允许轻微超载以减少碎片
while (IsWithinTolerance(assignContext.coreCache.costLimit,
    curCost / FA_TOLERANCE_RATIO,
    assignContext.coreCache.cost + curCost)) {
    assignContext.coreCache.cost += curCost;
    assignContext.coreCache.block++;
    assignContext.curS2Idx++;
}
```

**vs. baseline (lingxi-code):**
```cpp
// baseline: 严格均分，不允许超载
uint32_t blocksPerCore = totalBlocks / coreNum;
if (coreIdx < totalBlocks % coreNum) blocksPerCore++;
```

Benefit: 减少因严格均分产生的碎片核（只分到极少量工作的核），提高整体利用率
Trade-off: 容忍度参数需要调优，过大会导致负载不均，过小退化为严格均分

---

## Variant F: 跨核归约(Flash Decoding)负载均衡
Source: flash_attention

当一行(S1G)的 S2 方向被切分到多个核时，需要跨核归约。SplitFD 函数对所有需要归约的行进行二次负载均衡：计算每个归约任务的数据量（s2SplitNum × gS1SplitNum），按 Vector 核数均衡分配。分配时同样使用容忍度机制（FD_TOLERANCE_RATIO），判断当前 Vector 的剩余空间是否能容纳一半当前归约块。

**Expert implementation:**
```cpp
void SplitFD(SplitResult &result)
{
    uint32_t totalFDLoad = 0;
    for (uint32_t i = 0; i < result.numOfFdHead; i++) {
        totalFDLoad += result.fdRes.s2SplitNumOfFdHead[i] *
                       result.fdRes.gS1SplitNumOfFdHead[i];
    }

    uint32_t maxVectorNum = std::min(totalFDHeadSplit,
        result.usedCoreNum * result.vecCubeRatio);
    double loadThrOfVector = (double)totalFDLoad / (double)maxVectorNum;

    for (uint32_t i = 0; i < result.numOfFdHead; i++) {
        for (uint32_t gS1SplitIdx = 0; ...) {
            double remainSpace = loadThrOfVector - (double)loadOfCurVector;
            if (fDKVSplitNum > remainSpace * FD_TOLERANCE_RATIO) {
                // 当前 Vector 装满，切换到下一个
                curCoreIndex++;
                // 动态更新负载上限
                loadThrOfVector = (double)(totalFDLoad - loadOfCurVector) /
                    (double)(maxVectorNum - curCoreIndex);
                loadOfCurVector = 0;
            }
            loadOfCurVector += fDKVSplitNum;
        }
    }
    result.usedVecNumOfFd = curCoreIndex + 1;
}
```

**vs. baseline (lingxi-code):**
```cpp
// baseline: 不支持跨核归约，每行只在单核内处理
// 当 S2 很长时，单核成为瓶颈
```

Benefit: 支持长序列场景下的 S2 方向并行，突破单核处理 S2 的性能瓶颈
Trade-off: 引入归约同步开销，需要额外的 workspace 存储中间结果
