#pragma once

#include "kernel_operator.h"

// GELU tanh approximation constants
// gelu(x) = x / (1 + exp(-1.5957691 * (x + 0.044715 * x^3)))
// This is equivalent to PyTorch GELU(approximate="tanh")
constexpr float GELU_APPROX_FACTOR = 1.5957691f;
constexpr float GELU_CUBIC_COEF = 0.044715f;
constexpr float GELU_NEG_APPROX_FACTOR = -1.5957691f;
constexpr float GELU_ONE_F = 1.0f;

struct GeluTilingData {
    uint32_t totalElements;
    uint32_t blockSize;
    uint32_t usedCoreNum;
};

// FP32 GELU kernel - direct computation
class GeluKernelFp32 {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR tilingGM, AscendC::TPipe *pipe)
    {
        pipe_ = pipe;

        auto tilingPtr = reinterpret_cast<__gm__ GeluTilingData *>(tilingGM);
        totalElements_ = tilingPtr->totalElements;
        blockSize_ = tilingPtr->blockSize;
        usedCoreNum_ = tilingPtr->usedCoreNum;

        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x), totalElements_);
        yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y), totalElements_);

        uint32_t bufferSize = blockSize_ * sizeof(float);
        pipe_->InitBuffer(xInQueue_, 1, bufferSize);
        pipe_->InitBuffer(yOutQueue_, 1, bufferSize);
        pipe_->InitBuffer(tmpBuf1_, bufferSize);
        pipe_->InitBuffer(tmpBuf2_, bufferSize);
    }

    __aicore__ inline void Process()
    {
        const int32_t blockIdx = AscendC::GetBlockIdx();
        const int32_t tileStep = static_cast<int32_t>(usedCoreNum_);

        for (int32_t offset = blockIdx * blockSize_;
             offset < static_cast<int32_t>(totalElements_);
             offset += tileStep * blockSize_) {
            const int32_t remain = totalElements_ - offset;
            const int32_t size = (remain < static_cast<int32_t>(blockSize_)) ? remain : blockSize_;
            if (size > 0) {
                ComputeBlock(offset, size);
            }
        }
    }

private:
    __aicore__ inline void ComputeBlock(int32_t offset, int32_t size)
    {
        xLocal_ = xInQueue_.AllocTensor<float>();
        yLocal_ = yOutQueue_.AllocTensor<float>();

        // Copy input from GM
        AscendC::DataCopy(xLocal_, xGM_[offset], size);

        // Test: y = x * 0.5
        AscendC::Muls(yLocal_, xLocal_, 0.5f, size);

        // Copy output to GM
        AscendC::DataCopy(yGM_[offset], yLocal_, size);

        yOutQueue_.FreeTensor(yLocal_);
        xInQueue_.FreeTensor(xLocal_);
    }

    AscendC::TPipe *pipe_{nullptr};
    uint32_t totalElements_;
    uint32_t blockSize_;
    uint32_t usedCoreNum_;

    AscendC::GlobalTensor<float> xGM_;
    AscendC::GlobalTensor<float> yGM_;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> xInQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> yOutQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf2_;

    AscendC::LocalTensor<float> xLocal_;
    AscendC::LocalTensor<float> yLocal_;
    AscendC::LocalTensor<float> t1_;
    AscendC::LocalTensor<float> t2_;
};

// FP16 GELU kernel - cast to FP32, compute, cast back
class GeluKernelFp16 {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR tilingGM, AscendC::TPipe *pipe)
    {
        pipe_ = pipe;

        auto tilingPtr = reinterpret_cast<__gm__ GeluTilingData *>(tilingGM);
        totalElements_ = tilingPtr->totalElements;
        blockSize_ = tilingPtr->blockSize;
        usedCoreNum_ = tilingPtr->usedCoreNum;

        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x), totalElements_);
        yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(y), totalElements_);

        uint32_t fp16BufferSize = blockSize_ * sizeof(half);
        uint32_t fp32BufferSize = blockSize_ * sizeof(float);
        pipe_->InitBuffer(xInQueue_, 1, fp16BufferSize);
        pipe_->InitBuffer(yOutQueue_, 1, fp16BufferSize);
        pipe_->InitBuffer(fp32Buf_, fp32BufferSize);
        pipe_->InitBuffer(tmpBuf1_, fp32BufferSize);
        pipe_->InitBuffer(tmpBuf2_, fp32BufferSize);
    }

    __aicore__ inline void Process()
    {
        const int32_t blockIdx = AscendC::GetBlockIdx();
        const int32_t tileStep = static_cast<int32_t>(usedCoreNum_);

        for (int32_t offset = blockIdx * blockSize_;
             offset < static_cast<int32_t>(totalElements_);
             offset += tileStep * blockSize_) {
            const int32_t remain = totalElements_ - offset;
            const int32_t size = (remain < static_cast<int32_t>(blockSize_)) ? remain : blockSize_;
            if (size > 0) {
                ComputeBlock(offset, size);
            }
        }
    }

private:
    __aicore__ inline void ComputeBlock(int32_t offset, int32_t size)
    {
        xLocal_ = xInQueue_.AllocTensor<half>();
        yLocal_ = yOutQueue_.AllocTensor<half>();

        // Copy input from GM using plain DataCopy
        AscendC::DataCopy(xLocal_, xGM_[offset], size);

        // Simple test: y = 0.5 * x in half precision
        AscendC::Muls(yLocal_, xLocal_, half(0.5f), size);

        // Copy output to GM
        AscendC::DataCopy(yGM_[offset], yLocal_, size);

        yOutQueue_.FreeTensor(yLocal_);
        xInQueue_.FreeTensor(xLocal_);
    }

    AscendC::TPipe *pipe_{nullptr};
    uint32_t totalElements_;
    uint32_t blockSize_;
    uint32_t usedCoreNum_;

    AscendC::GlobalTensor<half> xGM_;
    AscendC::GlobalTensor<half> yGM_;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> xInQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> yOutQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf2_;

    AscendC::LocalTensor<half> xLocal_;
    AscendC::LocalTensor<half> yLocal_;
    AscendC::LocalTensor<float> fp32X_;
    AscendC::LocalTensor<float> t1_;
    AscendC::LocalTensor<float> t2_;
};

// BF16 GELU kernel - cast to FP32, compute, cast back
class GeluKernelBf16 {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR tilingGM, AscendC::TPipe *pipe)
    {
        pipe_ = pipe;

        auto tilingPtr = reinterpret_cast<__gm__ GeluTilingData *>(tilingGM);
        totalElements_ = tilingPtr->totalElements;
        blockSize_ = tilingPtr->blockSize;
        usedCoreNum_ = tilingPtr->usedCoreNum;

        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(x), totalElements_);
        yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(y), totalElements_);

        uint32_t bf16BufferSize = blockSize_ * sizeof(bfloat16_t);
        uint32_t fp32BufferSize = blockSize_ * sizeof(float);
        pipe_->InitBuffer(xInQueue_, 1, bf16BufferSize);
        pipe_->InitBuffer(yOutQueue_, 1, bf16BufferSize);
        pipe_->InitBuffer(fp32Buf_, fp32BufferSize);
        pipe_->InitBuffer(tmpBuf1_, fp32BufferSize);
        pipe_->InitBuffer(tmpBuf2_, fp32BufferSize);
    }

    __aicore__ inline void Process()
    {
        const int32_t blockIdx = AscendC::GetBlockIdx();
        const int32_t tileStep = static_cast<int32_t>(usedCoreNum_);

        for (int32_t offset = blockIdx * blockSize_;
             offset < static_cast<int32_t>(totalElements_);
             offset += tileStep * blockSize_) {
            const int32_t remain = totalElements_ - offset;
            const int32_t size = (remain < static_cast<int32_t>(blockSize_)) ? remain : blockSize_;
            if (size > 0) {
                ComputeBlock(offset, size);
            }
        }
    }

private:
    __aicore__ inline void ComputeBlock(int32_t offset, int32_t size)
    {
        xLocal_ = xInQueue_.AllocTensor<bfloat16_t>();
        yLocal_ = yOutQueue_.AllocTensor<bfloat16_t>();

        // Copy input from GM
        AscendC::DataCopyExtParams extParams{1, static_cast<uint32_t>(size * sizeof(bfloat16_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(xLocal_, xGM_[offset], extParams, padParams);

        // Cast BF16 -> FP32 for computation
        fp32X_ = fp32Buf_.Get<float>();
        AscendC::Cast(fp32X_, xLocal_, AscendC::RoundMode::CAST_NONE, size);

        // GELU tanh approximation in FP32
        t1_ = tmpBuf1_.Get<float>();
        t2_ = tmpBuf2_.Get<float>();

        // t1 = x^2
        AscendC::Mul(t1_, fp32X_, fp32X_, size);
        // t2 = x^3
        AscendC::Mul(t2_, t1_, fp32X_, size);
        // t1 = 0.044715 * x^3
        AscendC::Muls(t1_, t2_, GELU_CUBIC_COEF, size);
        // t2 = x + 0.044715 * x^3
        AscendC::Add(t2_, fp32X_, t1_, size);
        // t1 = -1.5957691 * (x + 0.044715 * x^3)
        AscendC::Muls(t1_, t2_, GELU_NEG_APPROX_FACTOR, size);
        // t2 = exp(-1.5957691 * (...))
        AscendC::Exp(t2_, t1_, size);
        // t1 = 1 + exp(...)
        AscendC::Adds(t1_, t2_, GELU_ONE_F, size);
        // fp32_result = x / (1 + exp(...))
        AscendC::Div(t2_, fp32X_, t1_, size);

        // Cast FP32 -> BF16
        AscendC::Cast(yLocal_, t2_, AscendC::RoundMode::CAST_ROUND, size);

        // Copy output to GM
        AscendC::DataCopyPad(yGM_[offset], yLocal_, extParams);

        yOutQueue_.FreeTensor(yLocal_);
        xInQueue_.FreeTensor(xLocal_);
    }

    AscendC::TPipe *pipe_{nullptr};
    uint32_t totalElements_;
    uint32_t blockSize_;
    uint32_t usedCoreNum_;

    AscendC::GlobalTensor<bfloat16_t> xGM_;
    AscendC::GlobalTensor<bfloat16_t> yGM_;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> xInQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> yOutQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf2_;

    AscendC::LocalTensor<bfloat16_t> xLocal_;
    AscendC::LocalTensor<bfloat16_t> yLocal_;
    AscendC::LocalTensor<float> fp32X_;
    AscendC::LocalTensor<float> t1_;
    AscendC::LocalTensor<float> t2_;
};
