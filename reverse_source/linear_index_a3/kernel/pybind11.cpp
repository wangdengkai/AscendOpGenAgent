#include <algorithm>
#include <cstdint>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "linear_index_v2_tiling.h"

extern "C" void linear_index_v2_int64_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *indexList,
    uint8_t *stride,
    uint8_t *valueSize,
    uint8_t *output,
    uint8_t *tiling);

extern "C" void linear_index_v2_int32_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *indexList,
    uint8_t *stride,
    uint8_t *valueSize,
    uint8_t *output,
    uint8_t *tiling);

namespace linear_index_v2_ext {

using LaunchFn = void (*)(
    uint32_t,
    void *,
    uint8_t *,
    uint8_t *,
    uint8_t *,
    uint8_t *,
    uint8_t *);

constexpr int64_t SCALE_SPACE = 20480;
constexpr int64_t UB_PART = 4;
constexpr int64_t MIN_UB_SIZE = 1024;
constexpr int64_t DEFAULT_UB_SIZE = 196608;

struct ComputeParam {
    uint64_t formerTime = 0;
    uint64_t tailTime = 0;
    uint64_t formerDataNum = 0;
    uint64_t tailDataNum = 0;
};

struct LaunchConfig {
    uint32_t blockDim = 0;
    LinearIndexV2TilingData tiling{};
};

LaunchFn ResolveLaunchFn(at::ScalarType dtype)
{
    if (dtype == at::kLong) {
        return linear_index_v2_int64_do;
    }
    if (dtype == at::kInt) {
        return linear_index_v2_int32_do;
    }
    TORCH_CHECK(false, "indices must be int32 or int64");
    return nullptr;
}

void ComputeCopyParam(ComputeParam &param, uint64_t dataNum, uint64_t dataMaxInUb)
{
    const uint64_t copyTime = (dataNum + dataMaxInUb - 1) / dataMaxInUb;
    param.formerTime = dataNum % copyTime;
    param.tailTime = copyTime - param.formerTime;
    param.formerDataNum = (dataNum + copyTime - 1) / copyTime;
    param.tailDataNum = dataNum / copyTime;
}

LaunchConfig BuildLaunchConfig(uint64_t idxNum, const std::vector<uint64_t> &indicesMask, size_t tensorId)
{
    TORCH_CHECK(idxNum > 0, "idxNum must be positive");
    TORCH_CHECK(tensorId > 0, "tensorId must be positive");
    TORCH_CHECK(tensorId <= MAX_DIM_NUM, "tensorId must be <= ", MAX_DIM_NUM);

    const uint64_t usedCoreNum = std::min<uint64_t>(DEFAULT_NUM_PHYSICAL_CORES, idxNum);
    const uint64_t ubSize = DEFAULT_UB_SIZE - SCALE_SPACE;
    TORCH_CHECK(ubSize >= MIN_UB_SIZE, "UB size is too small");
    const uint64_t dataMaxInUb = (ubSize / UB_PART) / sizeof(int32_t);

    const uint64_t formerCoreNum = idxNum % usedCoreNum;
    const uint64_t tailCoreNum = usedCoreNum - formerCoreNum;
    const uint64_t formerCoreDataNum = (idxNum + usedCoreNum - 1) / usedCoreNum;
    const uint64_t tailCoreDataNum = idxNum / usedCoreNum;

    ComputeParam formerCoreParam;
    ComputeParam tailCoreParam;
    ComputeCopyParam(formerCoreParam, formerCoreDataNum, dataMaxInUb);
    ComputeCopyParam(tailCoreParam, tailCoreDataNum, dataMaxInUb);

    LaunchConfig config;
    config.blockDim = static_cast<uint32_t>(usedCoreNum);
    config.tiling.params.usedCoreNum = usedCoreNum;
    config.tiling.params.tensorId = tensorId;
    config.tiling.params.formerCoreNum = formerCoreNum;
    config.tiling.params.formerCoreDataNum = formerCoreDataNum;
    config.tiling.params.formerCoreFormerDataNum = formerCoreParam.formerDataNum;
    config.tiling.params.formerCoreTailDataNum = formerCoreParam.tailDataNum;
    config.tiling.params.formerCoreFormerTime = formerCoreParam.formerTime;
    config.tiling.params.formerCoreTailTime = formerCoreParam.tailTime;
    config.tiling.params.tailCoreNum = tailCoreNum;
    config.tiling.params.tailCoreDataNum = tailCoreDataNum;
    config.tiling.params.tailCoreFormerDataNum = tailCoreParam.formerDataNum;
    config.tiling.params.tailCoreTailDataNum = tailCoreParam.tailDataNum;
    config.tiling.params.tailCoreFormerTime = tailCoreParam.formerTime;
    config.tiling.params.tailCoreTailTime = tailCoreParam.tailTime;
    for (size_t i = 0; i < tensorId; ++i) {
        config.tiling.params.indicesMask[i] = indicesMask[i];
    }
    return config;
}

void ValidateInputs(const std::vector<at::Tensor> &indices, const at::Tensor &stride, const at::Tensor &valueSize)
{
    TORCH_CHECK(!indices.empty(), "indices must not be empty");
    TORCH_CHECK(indices.size() <= MAX_DIM_NUM, "indices count must be <= ", MAX_DIM_NUM);
    TORCH_CHECK(stride.device().is_privateuseone(), "stride must be on NPU");
    TORCH_CHECK(valueSize.device().is_privateuseone(), "valueSize must be on NPU");
    TORCH_CHECK(stride.is_contiguous(), "stride must be contiguous");
    TORCH_CHECK(valueSize.is_contiguous(), "valueSize must be contiguous");
    TORCH_CHECK(stride.scalar_type() == at::kInt, "stride must be int32");
    TORCH_CHECK(valueSize.scalar_type() == at::kInt, "valueSize must be int32");
    TORCH_CHECK(stride.dim() == 1, "stride must be 1D");
    TORCH_CHECK(valueSize.dim() == 1, "valueSize must be 1D");
    TORCH_CHECK(static_cast<size_t>(stride.size(0)) == indices.size(), "stride length must match indices size");
    TORCH_CHECK(static_cast<size_t>(valueSize.size(0)) == indices.size(), "valueSize length must match indices size");
}

at::Tensor run_linear_index_v2(
    const std::vector<at::Tensor> &indices,
    const at::Tensor &stride,
    const at::Tensor &valueSize)
{
    ValidateInputs(indices, stride, valueSize);

    at::ScalarType dtype = at::kInt;
    bool foundValidTensor = false;
    std::vector<int64_t> outputShape;
    uint64_t idxNum = 0;
    std::vector<uint64_t> indicesMask(indices.size(), 0);

    for (size_t i = 0; i < indices.size(); ++i) {
        const at::Tensor &index = indices[i];
        TORCH_CHECK(index.device().is_privateuseone(), "indices[", i, "] must be on NPU");
        TORCH_CHECK(index.is_contiguous(), "indices[", i, "] must be contiguous");
        TORCH_CHECK(
            index.scalar_type() == at::kInt || index.scalar_type() == at::kLong,
            "indices[", i, "] must be int32 or int64");

        if (index.numel() == 0) {
            continue;
        }

        if (!foundValidTensor) {
            foundValidTensor = true;
            dtype = index.scalar_type();
            outputShape.assign(index.sizes().begin(), index.sizes().end());
            idxNum = static_cast<uint64_t>(index.numel());
        } else {
            std::vector<int64_t> currentShape(index.sizes().begin(), index.sizes().end());
            TORCH_CHECK(index.scalar_type() == dtype, "all non-empty indices tensors must share the same dtype");
            TORCH_CHECK(currentShape == outputShape, "all non-empty indices tensors must share shape");
        }
        indicesMask[i] = 1;
    }

    TORCH_CHECK(foundValidTensor, "at least one indices tensor must be non-empty");
    at::Tensor output = at::zeros(outputShape, indices[0].options().dtype(at::kInt));

    at::Tensor ptrTableCpu = at::empty(
        {static_cast<long>(indices.size())},
        at::device(at::kCPU).dtype(at::kLong));
    auto *ptrTable = reinterpret_cast<uint64_t *>(ptrTableCpu.data_ptr<int64_t>());
    for (size_t i = 0; i < indices.size(); ++i) {
        ptrTable[i] = reinterpret_cast<uint64_t>(const_cast<void *>(indices[i].storage().data()));
    }
    at::Tensor ptrTableNpu = ptrTableCpu.to(at::kPrivateUse1);

    LaunchConfig config = BuildLaunchConfig(idxNum, indicesMask, indices.size());
    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(LinearIndexV2TilingData))},
        at::device(at::kCPU).dtype(at::kByte));
    auto *tiling = reinterpret_cast<LinearIndexV2TilingData *>(tilingCpu.data_ptr());
    *tiling = config.tiling;
    at::Tensor tilingNpu = tilingCpu.to(at::kPrivateUse1);
    LaunchFn launch = ResolveLaunchFn(dtype);
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    launch(
        config.blockDim,
        aclStream,
        static_cast<uint8_t *>(ptrTableNpu.data_ptr()),
        static_cast<uint8_t *>(const_cast<void *>(stride.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(valueSize.storage().data())),
        static_cast<uint8_t *>(output.data_ptr()),
        static_cast<uint8_t *>(tilingNpu.data_ptr()));
    return output;
}

} // namespace linear_index_v2_ext

PYBIND11_MODULE(_linear_index_v2_ext, m)
{
    m.doc() = "linear_index_v2 AscendC extension";
    m.def("run_linear_index_v2", &linear_index_v2_ext::run_linear_index_v2, "");
}
