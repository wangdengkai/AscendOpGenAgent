#ifndef LINEAR_INDEX_V2_KERNEL_COMMON_H
#define LINEAR_INDEX_V2_KERNEL_COMMON_H

#include <cstddef>
#include <cstdint>

#include "kernel_operator.h"

template <typename Tp, Tp v>
struct integralConstant {
    static constexpr Tp value = v;
};

using true_type = integralConstant<bool, true>;
using false_type = integralConstant<bool, false>;

template <typename, typename>
struct isSame : public false_type {
};

template <typename Tp>
struct isSame<Tp, Tp> : public true_type {
};

#define IS_CAST_INT (isSame<T, int64_t>::value)

template <typename T>
__aicore__ inline void CopyTiling(T *tiling, GM_ADDR tilingGM)
{
    int32_t *dst = reinterpret_cast<int32_t *>(tiling);
    auto *src = reinterpret_cast<__gm__ int32_t *>(tilingGM);
    for (size_t i = 0; i < sizeof(T) / sizeof(int32_t); ++i) {
        dst[i] = src[i];
    }
}

#endif
