#ifndef LINEAR_INDEX_V2_TILING_H
#define LINEAR_INDEX_V2_TILING_H

#include <cstddef>
#include <cstdint>

constexpr int32_t DEFAULT_NUM_PHYSICAL_CORES = 20;
constexpr size_t MAX_DIM_NUM = 8;

struct LinearIndexV2TilingParam {
    uint64_t usedCoreNum = 0;
    uint64_t tensorId = 0;
    uint64_t formerCoreNum = 0;
    uint64_t formerCoreDataNum = 0;
    uint64_t formerCoreFormerDataNum = 0;
    uint64_t formerCoreTailDataNum = 0;
    uint64_t formerCoreFormerTime = 0;
    uint64_t formerCoreTailTime = 0;
    uint64_t tailCoreNum = 0;
    uint64_t tailCoreDataNum = 0;
    uint64_t tailCoreFormerDataNum = 0;
    uint64_t tailCoreTailDataNum = 0;
    uint64_t tailCoreFormerTime = 0;
    uint64_t tailCoreTailTime = 0;
    uint64_t indicesMask[MAX_DIM_NUM] = {0};
};

struct LinearIndexV2TilingData {
    LinearIndexV2TilingParam params;
};

#endif
