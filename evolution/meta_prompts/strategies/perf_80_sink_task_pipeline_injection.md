# P80 Sink Token 流水线注入与 S2 块跳过 (Sink Task Pipeline Injection and S2 Block Skip)
## Overview
在 Streaming LLM 的 Attention Sink 场景中，将 sink token（前 N 个 token 的 KV cache）作为独立 task 注入到 Flash Attention 的 3-stage pipeline 中，与普通 sparse task 共享同一套流水线调度。同时，在 sink 区域与有效 token 区域之间的 gap 中，通过 IsSkipCal 判断跳过无效 S2 block 的全部计算。

## When to Use
- Streaming LLM / Attention Sink 场景，需要保留前 N 个 token 的注意力
- Sparse Attention + Sink，sink 区域与 band 窗口之间可能有 gap
- sinkNum > 0 且 sparseMode 为 band（sparseMode=4）

## Trade-off
- 每个 S1G block 多一个 sink task，增加总 task 数
- sink tensor 需要额外的 GM 空间
- sInnerSize 减小（256）可能降低有效 block 的计算效率

**Source operators**: ai_infra_fused_infer_attention_sink

---
## Variant A: Sink Task 注入 3-Stage Pipeline
Source: ai_infra_fused_infer_attention_sink op_kernel

每个 S1G block 的第一个 S2 task 被标记为 `isSinkTensor=true`，从独立的 keySink/valueSink tensor 读取数据。

**Expert implementation:**
```cpp
// Sink task 注入流水线
void FlashAttention() {
    for (int s1g = 0; s1g < s1gLoops; s1g++) {
        // 第一个 task: sink token
        if (keySinkNumber > 0) {
            TaskInfo sinkTask;
            sinkTask.isSinkTensor = true;
            sinkTask.s2Start = 0;
            sinkTask.s2End = sinkNumber;
            ExecuteTask(sinkTask);  // 从 keySink/valueSink 读取
        }
        // 后续 tasks: sparse token
        for (int s2 = sinkNumber; s2 < s2End; s2 += sInnerSize) {
            TaskInfo sparseTask;
            sparseTask.isSinkTensor = false;
            ExecuteTask(sparseTask);  // 从 KV cache 读取
        }
    }
}
```

**vs. baseline (lingxi-code):**
```cpp
// 基线：sink token 和 sparse token 分开处理
// 先处理 sink（独立循环），再处理 sparse（独立循环）
for (int i = 0; i < sinkNum; i++) { ProcessSink(i); }
for (int s2 = 0; s2 < s2End; s2++) { ProcessSparse(s2); }
// 两个循环无法共享流水线
```

Benefit: sink 和 sparse 统一在同一流水线中处理；无需额外计算阶段
Trade-off: 每个 S1G block 多一个 sink task

---
## Variant B: Sparse S2 Block Skip with Sink Gap
Source: ai_infra_fused_infer_attention_sink op_kernel

在 band sparse + sink 场景中，sink 区域和有效 token 区域之间的 gap 通过 IsSkipCal 跳过。

**Expert implementation:**
```cpp
// S2 block 跳过判断
TaskDealMode GetTaskDealMode(int s2Start, int s2End) {
    if (isSinkTensor) return NORMAL;  // sink block 永不跳过

    // 判断是否在 sink-band gap 中
    if (IsSkipCal(s2Start, s2End, sinkNumber, preTokens, nextTokens)) {
        return SKIP;  // 跳过 MM1+Vec1+MM2+Vec2 全部计算
    }
    return NORMAL;
}

// tiling: sink 场景减小 S2 基本块
if (sinkNumber > 0 && sparseMode == BAND) {
    sInnerSize_ = 256;  // 减小粒度，使 gap 跳过更精确
}
```

Benefit: 完全跳过无效 S2 block 的所有计算和搬运
Trade-off: 每个 S2 block 需要额外的 IsSkipCal 判断开销
