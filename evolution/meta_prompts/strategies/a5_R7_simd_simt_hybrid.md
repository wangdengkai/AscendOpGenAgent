# R7: SIMD/SIMT 混合编程

## Overview
对同一算子中的不同操作，根据访问模式选择 SIMD 或 SIMT 模型。

## When to Use
- 算子同时包含规则访问（逐元素）和不规则访问（gather/scatter）
- 存在动态索引或条件分支密集的操作
- Attention 类算子（mask 索引 + 逐元素计算）

## Trade-off
- **收益**: 不规则访问性能提升显著（SIMT 天然支持随机访问）
- **风险**: SIMD/SIMT 切换有开销，不宜频繁切换
- **复杂度**: 高（需要同时掌握两种编程模型）

## Variant A: Gather(SIMT) + Compute(SIMD)

```cpp
// SIMT: 不规则 gather
__simt_vf__ __launch_bounds__(1024)
inline void GatherVF(__ubuf__ float* output, __gm__ float* input,
                     __ubuf__ int32_t* indices, int count) {
    for (int idx = threadIdx.x; idx < count; idx += blockDim.x) {
        output[idx] = input[indices[idx]];
    }
}

// SIMD: 规则逐元素计算
__simd_vf__ inline void ComputeVF(__ubuf__ float* dst, __ubuf__ float* src,
    uint32_t count, uint32_t oneRepSize, uint16_t repTimes) {
    RegTensor<float> srcReg, dstReg;
    MaskReg mask;
    for (uint16_t i = 0; i < repTimes; ++i) {
        mask = UpdateMask<float>(count);
        LoadAlign(srcReg, src + i * oneRepSize);
        Adds(dstReg, srcReg, 1.0f, mask);
        StoreAlign(dst + i * oneRepSize, dstReg, mask);
    }
}

// 混合调用
__aicore__ inline void Compute() {
    asc_vf_call<GatherVF>(dim3{256,1,1}, ubOut, gmIn, ubIdx, count);
    VF_CALL<ComputeVF>(ubOut, ubOut, count, oneRepSize, repTimes);
}
```
