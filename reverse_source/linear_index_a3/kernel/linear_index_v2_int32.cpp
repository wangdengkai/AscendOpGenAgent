#include "linear_index_v2_kernel.h"

extern "C" __global__ __aicore__ void linear_index_v2_int32_custom(
    GM_ADDR indexList,
    GM_ADDR stride,
    GM_ADDR valueSize,
    GM_ADDR output,
    GM_ADDR tiling)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    LinearIndexV2TilingData tilingData;
    CopyTiling(&tilingData, tiling);
    AscendC::TPipe pipe;
    LinearIndexKernelV2<int32_t> op(indexList, stride, valueSize, output, tilingData, pipe);
    op.Process();
#endif
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void linear_index_v2_int32_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *indexList,
    uint8_t *stride,
    uint8_t *valueSize,
    uint8_t *output,
    uint8_t *tiling)
{
    linear_index_v2_int32_custom<<<blockDim, nullptr, stream>>>(indexList, stride, valueSize, output, tiling);
}
#endif
