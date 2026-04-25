#ifndef LINEAR_INDEX_V2_KERNEL_H
#define LINEAR_INDEX_V2_KERNEL_H

#include "kernel_common.h"
#include "linear_index_v2_tiling.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t BLOCK_SIZE = 32;

template <typename T>
class LinearIndexKernelV2
{
public:
    __aicore__ inline LinearIndexKernelV2() = delete;

    __aicore__ inline LinearIndexKernelV2(
        GM_ADDR indexList,
        GM_ADDR stride,
        GM_ADDR valueSize,
        GM_ADDR output,
        const LinearIndexV2TilingData &tiling,
        TPipe &pipe)
    {
        InitParams(tiling, indexList);
        SetGmAddr(stride, valueSize, output);
        InitBuffers(pipe);
    }

    __aicore__ inline void Process()
    {
        for (int i = 0; i < tensorId_; ++i) {
            if (indicesMask_[i] == 0) {
                continue;
            }
            indexGm_.SetGlobalBuffer(GetTensorAddr(indexList_, i) + idxAddrOffset_);
            for (int j = 0; j < formerTime_; ++j) {
                CopyIn(i, j, true);
                Compute(true);
                CopyOut(j, true);
            }
            for (int j = 0; j < tailTime_; ++j) {
                CopyIn(i, j, false);
                Compute(false);
                CopyOut(j, false);
            }
        }
    }

private:
    __aicore__ inline void InitParams(const LinearIndexV2TilingData &tiling, GM_ADDR indexList)
    {
        indexList_ = indexList;
        blockIdx_ = GetBlockIdx();
        formerCoreNum_ = tiling.params.formerCoreNum;
        tensorId_ = tiling.params.tensorId;
        indicesMask_ = tiling.params.indicesMask;
        if (blockIdx_ < formerCoreNum_) {
            dataNum_ = tiling.params.formerCoreDataNum;
            formerDataNum_ = tiling.params.formerCoreFormerDataNum;
            tailDataNum_ = tiling.params.formerCoreTailDataNum;
            formerTime_ = tiling.params.formerCoreFormerTime;
            tailTime_ = tiling.params.formerCoreTailTime;
            idxAddrOffset_ = tiling.params.formerCoreDataNum * blockIdx_;
        } else {
            dataNum_ = tiling.params.tailCoreDataNum;
            formerDataNum_ = tiling.params.tailCoreFormerDataNum;
            tailDataNum_ = tiling.params.tailCoreTailDataNum;
            formerTime_ = tiling.params.tailCoreFormerTime;
            tailTime_ = tiling.params.tailCoreTailTime;
            idxAddrOffset_ = tiling.params.formerCoreDataNum * formerCoreNum_ +
                             tiling.params.tailCoreDataNum * (blockIdx_ - formerCoreNum_);
        }
    }

    __aicore__ inline void InitBuffers(TPipe &pipe)
    {
        uint32_t strideAlign = BLOCK_SIZE / sizeof(int32_t);
        uint32_t idxInBlock = BLOCK_SIZE / sizeof(T);
        uint32_t idxAlignNum = ((formerDataNum_ + idxInBlock - 1) / idxInBlock) * idxInBlock;
        pipe.InitBuffer(strideQue_, BUFFER_NUM, strideAlign * sizeof(int32_t));
        pipe.InitBuffer(valueSizeQue_, BUFFER_NUM, strideAlign * sizeof(int32_t));
        pipe.InitBuffer(indexInQue_, BUFFER_NUM, idxAlignNum * sizeof(T));
        pipe.InitBuffer(indexOutQue_, BUFFER_NUM, idxAlignNum * sizeof(int32_t));
        pipe.InitBuffer(remainQue_, BUFFER_NUM, idxAlignNum * sizeof(float));
    }

    __aicore__ inline void SetGmAddr(GM_ADDR stride, GM_ADDR valueSize, GM_ADDR output)
    {
        valueSizeGm_.SetGlobalBuffer((__gm__ int32_t *)valueSize);
        strideGm_.SetGlobalBuffer((__gm__ int32_t *)stride);
        outputGm_.SetGlobalBuffer((__gm__ int32_t *)output + idxAddrOffset_, dataNum_);
        InitGlobalMemory(outputGm_, dataNum_, static_cast<int32_t>(0));
        SyncAll();
    }

    __aicore__ inline void CopyIn(int64_t progress, int curTimes, bool formerFlag)
    {
        LocalTensor<T> indexLocal = indexInQue_.AllocTensor<T>();
        LocalTensor<int32_t> strideLocal = strideQue_.AllocTensor<int32_t>();
        LocalTensor<int32_t> valueSizeLocal = valueSizeQue_.AllocTensor<int32_t>();
        uint32_t copyNum = formerFlag ? formerDataNum_ : tailDataNum_;
        int64_t addrOffset =
            formerFlag ? formerDataNum_ * curTimes : formerDataNum_ * formerTime_ + tailDataNum_ * curTimes;
        DataCopyExtParams idxCopyParams{1, static_cast<uint32_t>(copyNum * sizeof(T)), 0, 0, 0};
        DataCopyExtParams scalarCopyParams{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
        if constexpr (IS_CAST_INT) {
            DataCopyPadExtParams<uint32_t> uPadParams{true, 0, 0, 0};
            DataCopyPadGm2UBImpl(
                (__ubuf__ uint32_t *)indexLocal.GetPhyAddr(),
                (__gm__ uint32_t *)indexGm_[addrOffset].GetPhyAddr(),
                idxCopyParams,
                uPadParams);
        } else {
            DataCopyPadExtParams<T> uPadParams{true, 0, 0, 0};
            DataCopyPad(indexLocal, indexGm_[addrOffset], idxCopyParams, uPadParams);
        }
        DataCopyPad(valueSizeLocal, valueSizeGm_[progress], scalarCopyParams, padParams);
        DataCopyPad(strideLocal, strideGm_[progress], scalarCopyParams, padParams);
        valueSizeQue_.EnQue<int32_t>(valueSizeLocal);
        indexInQue_.EnQue<T>(indexLocal);
        strideQue_.EnQue<int32_t>(strideLocal);
    }

    __aicore__ inline void Compute(bool formerFlag)
    {
        LocalTensor<int32_t> indexOutLocal = indexOutQue_.AllocTensor<int32_t>();
        LocalTensor<T> indexLocal = indexInQue_.DeQue<T>();
        LocalTensor<int32_t> strideLocal = strideQue_.DeQue<int32_t>();
        LocalTensor<int32_t> valueSizeLocal = valueSizeQue_.DeQue<int32_t>();
        uint32_t dataNum = formerFlag ? formerDataNum_ : tailDataNum_;
        int32_t stride = strideLocal.GetValue(0);
        int32_t size = valueSizeLocal.GetValue(0);
        float sizeDiv = size == 0 ? 1.0f : 1.0f / size;
        if constexpr (IS_CAST_INT) {
            LocalTensor<float> remainLocal = remainQue_.AllocTensor<float>();
            Cast<float, T>(remainLocal, indexLocal, RoundMode::CAST_RINT, dataNum);
            Muls(remainLocal, remainLocal, sizeDiv, dataNum);
            Cast<int32_t, float>(indexOutLocal, remainLocal, RoundMode::CAST_FLOOR, dataNum);
            Muls(indexOutLocal, indexOutLocal, size, dataNum);
            remainQue_.FreeTensor<float>(remainLocal);

            LocalTensor<int32_t> castLocal = remainQue_.AllocTensor<int32_t>();
            Cast<int32_t, T>(castLocal, indexLocal, RoundMode::CAST_NONE, dataNum);
            Sub(indexOutLocal, castLocal, indexOutLocal, dataNum);
            Muls(indexOutLocal, indexOutLocal, stride, dataNum);
            remainQue_.FreeTensor<int32_t>(castLocal);
        } else {
            LocalTensor<float> remainLocal = remainQue_.AllocTensor<float>();
            Cast<float, T>(remainLocal, indexLocal, RoundMode::CAST_NONE, dataNum);
            Muls(remainLocal, remainLocal, sizeDiv, dataNum);
            Cast<int32_t, float>(indexOutLocal, remainLocal, RoundMode::CAST_FLOOR, dataNum);
            Muls(indexOutLocal, indexOutLocal, size, dataNum);
            Sub(indexOutLocal, indexLocal, indexOutLocal, dataNum);
            Muls(indexOutLocal, indexOutLocal, stride, dataNum);
            remainQue_.FreeTensor<float>(remainLocal);
        }
        indexInQue_.FreeTensor<T>(indexLocal);
        strideQue_.FreeTensor<int32_t>(strideLocal);
        valueSizeQue_.FreeTensor<int32_t>(valueSizeLocal);
        indexOutQue_.EnQue<int32_t>(indexOutLocal);
    }

    __aicore__ inline void CopyOut(int curTimes, bool formerFlag)
    {
        LocalTensor<int32_t> indexOutLocal = indexOutQue_.DeQue<int32_t>();
        uint32_t copyNum = formerFlag ? formerDataNum_ : tailDataNum_;
        int64_t addrOffset =
            formerFlag ? formerDataNum_ * curTimes : formerDataNum_ * formerTime_ + tailDataNum_ * curTimes;
        DataCopyExtParams idxCopyParams{1, static_cast<uint32_t>(copyNum * sizeof(int32_t)), 0, 0, 0};
        SetAtomicAdd<int32_t>();
        DataCopyPad(outputGm_[addrOffset], indexOutLocal, idxCopyParams);
        SetAtomicNone();
        indexOutQue_.FreeTensor<int32_t>(indexOutLocal);
    }

    __aicore__ inline __gm__ T *GetTensorAddr(GM_ADDR indexListPtr, int offset)
    {
        __gm__ uint64_t *tensorPtr = reinterpret_cast<__gm__ uint64_t *>(indexListPtr);
        return reinterpret_cast<__gm__ T *>(*(tensorPtr + offset));
    }

private:
    GM_ADDR indexList_;
    GlobalTensor<T> indexGm_;
    GlobalTensor<int32_t> valueSizeGm_;
    GlobalTensor<int32_t> strideGm_;
    GlobalTensor<int32_t> outputGm_;

    TQue<TPosition::VECIN, BUFFER_NUM> valueSizeQue_;
    TQue<TPosition::VECIN, BUFFER_NUM> strideQue_;
    TQue<TPosition::VECIN, BUFFER_NUM> remainQue_;
    TQue<TPosition::VECIN, BUFFER_NUM> indexInQue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> indexOutQue_;

    uint64_t idxAddrOffset_ = 0;
    uint64_t blockIdx_ = 0;
    uint64_t tensorId_ = 0;
    uint64_t dataNum_ = 0;
    uint64_t formerCoreNum_ = 0;
    uint64_t formerDataNum_ = 0;
    uint64_t tailDataNum_ = 0;
    uint64_t formerTime_ = 0;
    uint64_t tailTime_ = 0;
    const uint64_t *indicesMask_ = nullptr;
};

#endif
