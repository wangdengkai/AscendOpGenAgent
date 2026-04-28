#include "gelu_kernel.h"

extern "C" __global__ __aicore__ void gelu_fp32(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    GeluKernelFp32 kernel;
    kernel.Init(x, y, tiling, &pipe);
    kernel.Process();
}
