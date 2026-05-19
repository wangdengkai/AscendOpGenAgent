# P54 头开销优化策略 (Launch Overhead Optimization)
## Overview
头开销定义为算子启动到开始搬运 Tensor 数据（MTE2）的耗时，这段代码主要由 Scalar 流水执行，完成计算初始化工作。优化重点在于减少 Cache miss 导致的 LD/ST 延迟，通过减少栈变量访问、合并写操作、利用 STB 加速等手段降低头开销。

## When to Use
- 算子初始化阶段存在大量 LD/ST 操作
- ESL 日志分析显示大量 "[ISSUE INSTR] general hazard failed" 与 Load 依赖相关
- 头开销占算子总耗时比例较高（>5%）
- Cache miss 导致 LD/ST 延迟达数百 cycle

## Trade-off
- 需要深入理解 Scalar 流水线和 Cache 结构
- 过度优化可能降低代码可读性
- 部分优化手段需要硬件特性支持

**Source operators**: IFA, PFA 等复杂 Transformer 算子

---

## Variant A: 减少栈变量访问
Source: 【案例总结】IFA头开销优化.md

将频繁访问的栈变量改为寄存器变量或成员变量，减少对 GM 的 LD/ST 操作。

**Expert implementation:**
```cpp
// 优化前：频繁访问栈变量
struct LocalVars {
    uint32_t var1;
    uint32_t var2;
    // ... 大量栈变量
};

void Process() {
    LocalVars vars;
    for (int i = 0; i < nloops; i++) {
        vars.var1 = ...;  // 每次访问都是 LD/ST
        vars.var2 = ...;
    }
}

// 优化后：使用成员变量减少栈访问
class Kernel {
    uint32_t var1_;  // 成员变量，编译器优化为寄存器访问
    uint32_t var2_;
};
```

**vs. baseline (lingxi-code):**
```cpp
// 基线：大量栈变量，频繁 LD/ST
// Cache miss 时 LD/ST 延迟可达数百 cycle
```

Benefit: 减少 Cache miss 导致的 LD/ST 延迟，降低头开销
Trade-off: 需要重构代码结构，增加类成员变量

---

## Variant B: 合并写操作利用 STB 加速
Source: 【案例总结】IFA头开销优化.md

ST（Store）操作可以通过 STB（Store Buffer）加速，先写入 STB 后续合并写入 Cache。通过合并连续写操作，减少 STB FULL 情况。

**Expert implementation:**
```cpp
// 优化前：分散的写操作
void Init() {
    for (int i = 0; i < n; i++) {
        buffer[i] = value1;  // 每次写操作都可能触发 STB
    }
}

// 优化后：批量写操作
void Init() {
    // 使用 Duplicate 批量初始化
    Duplicate(buffer, initValue, count);
    // 或合并写操作到同一代码块
    buffer[0] = value1;
    buffer[1] = value2;
    // ... 连续写，STB 合并效率高
}
```

Benefit: 利用 STB 合并写入，减少写 Cache miss 延迟
Trade-off: 需要调整代码顺序，可能影响代码可读性

---

## Variant C: 减少 DCache 访问
Source: 【案例总结】IFA头开销优化.md

将初始化阶段的复杂计算尽量在 Tiling 阶段完成，减少 Kernel 初始化过程中的 LD/ST 操作。

**Expert implementation:**
```cpp
// Tiling 阶段预计算
struct TilingData {
    uint32_t precomputedOffset;  // 预计算的偏移量
    uint32_t precomputedSize;    // 预计算的尺寸
};

// Kernel 阶段直接使用预计算值
void Init() {
    // 直接使用 TilingData 中的预计算值，无需运行时计算
    offset_ = tilingData->precomputedOffset;
    size_ = tilingData->precomputedSize;
}
```

Benefit: 将计算前移到 Host 侧，减少 Kernel 初始化的计算量和访存量
Trade-off: 增加 TilingData 大小，可能触发 16K 限制

---

## Variant D: TPipe 外置优化 Scalar 常量折叠
Source: SIMD算子性能优化/头尾开销优化/避免TPipe在对象内创建和初始化.md

将 TPipe 对象创建于 Kernel 类外部，使 TPipe 内存空间独立于 Kernel 类对象。TPipe 内部初始化设置全局变量指针，若在类内部则编译器对该类 Scalar 变量采取保守策略，无法进行常量折叠和常量传播优化。

**Expert implementation:**
```cpp
// 优化前：TPipe 在类内创建
class KernelOp {
    TPipe pipe;  // 编译器保守策略，无法常量折叠
    void Init() { pipe.InitBuffer(...); }
};

// 优化后：TPipe 在 kernel 入口创建，通过指针传入
extern "C" __global__ __aicore__ void kernel_func(...) {
    TPipe pipe;
    KernelOp op;
    op.Init(&pipe);  // TPipe 独立于类，触发编译器 Scalar 优化
}
```

**vs. baseline (lingxi-code):**
```cpp
// 基线：TPipe 作为类成员，编译器无法优化 Scalar 变量
```

Benefit: Scalar 时间平均下降 17%（281us→236us），Scalar_time 占比从 21% 降至 17%
Trade-off: 需修改 Kernel 类接口，传入 TPipe 指针

---

## Variant E: TilingData 结构精简与对齐排布
Source: SIMD算子性能优化/头尾开销优化/限制TilingData结构大小.md

限制 TilingData 结构大小以减少 GM→栈空间的拷贝开销。三个手段：(1) 去除冗余字段（blockDim 可通过 GetBlockNum 获取）；(2) 根据数据范围选择最小类型（uint8_t 替代 uint64_t）；(3) 合理排布字段顺序减少 8 字节对齐 padding。

**Expert implementation:**
```cpp
// 优化前：字段类型过大，排布不合理
BEGIN_TILING_DATA_DEF(TilingData)
    TILING_DATA_FIELD_DEF(uint64_t, blockDim);   // 冗余，可用 GetBlockNum()
    TILING_DATA_FIELD_DEF(uint64_t, totalLength); // uint32_t 足够
    TILING_DATA_FIELD_DEF(uint8_t, flag1);        // padding 10 字节
    TILING_DATA_FIELD_DEF(uint64_t, tileNum);
END_TILING_DATA_DEF;

// 优化后：精简类型 + 对齐排布
BEGIN_TILING_DATA_DEF(TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF(uint32_t, tileNum);
    TILING_DATA_FIELD_DEF(uint8_t, flag1);        // padding 仅 2 字节
    TILING_DATA_FIELD_DEF(uint8_t, flag2);
END_TILING_DATA_DEF;
```

Benefit: 减少 GM 拷贝开销（us 级），减少栈空间占用，padding 从 10 字节降至 2 字节
Trade-off: 需要仔细分析每个字段的数据范围

---

## Variant F: 核数与 Kernel 类型精确配置
Source: SIMD算子性能优化/头尾开销优化/设置合适的核数和算子Kernel类型.md

头开销随核数线性增长（MIX kernel: 1 核 2.72us → 24 核 12.66us）。对微秒级小算子，减少启动核数可获得性能提升。纯 Vector 算子若以 MIX 方式启动会白白启动 Cube 核产生额外头开销。

**Expert implementation:**
```cpp
// Host 侧：根据数据量动态设置核数
uint32_t optimalCores = std::min(totalBlocks, maxCores);
context->SetBlockDim(optimalCores);

// 纯 Vector 算子：手动设置 Kernel 类型避免 MIX 启动
// MIX 空 kernel: 1核=2.72us, 24核=12.66us
// 纯 Cube: 1核=2.52us, 24核=5.7us
// 纯 Vector: 头开销最低
```

Benefit: 减少核启动开销，避免无用 Cube 核启动
Trade-off: 减少核数会增加单核计算量，需实测找最优平衡点

---

## Variant G: K_MAX_SHAPE_DIM 缩减栈空间
Source: SIMD算子性能优化/内存访问/通过缩减Tensor_ShapeInfo维度，优化栈空间.md

GlobalTensor/LocalTensor 内部 ShapeInfo 默认支持 8 维（K_MAX_SHAPE_DIM=8），每个 Tensor 占用 64 字节栈空间。不使用 ShapeInfo 时，定义 `#define K_MAX_SHAPE_DIM 0` 可完全消除此开销。

**Expert implementation:**
```cpp
// 在包含头文件前定义
#define K_MAX_SHAPE_DIM 0
#include "aclnn/kernel_operator.h"
// 每个 Tensor 节省 64 字节栈空间，减少 scalar 指令和 cache miss
```

Benefit: 缩减栈空间（每 Tensor 节省 64 字节），减少 scalar 指令，降低 cache miss 几率
Trade-off: 设置为 0 后无法使用 ShapeInfo/SetShapeInfo/GetShapeInfo 功能

---

## Variant H: 纯 Cube 模式消除 AIV 开销
Source: 优秀实践/Matmul性能调优案例/Matmul高阶API使能纯Cube模式.md

通过定义 ASCENDC_CUBE_ONLY 宏，使 AIV 核进入 kernel 后立即返回，消除 MIX 模式下 AIV 核的 Scalar 初始化开销和同步等待。

**Expert implementation:**
```cpp
#define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"
// AIV 核进入 kernel 后立即返回，不参与计算
// 总耗时 17.85us→11.21us（37.2%提升）
// Scalar 耗时 15.02us→5.17us（65.6%提升）
```

Benefit: 总耗时降低 37.2%，Scalar 耗时降低 65.6%
Trade-off: AIV 核完全不可用，仅适用于纯 Cube 场景

---

## Variant I: Matmul Tiling 全量常量化
Source: 优秀实践/Matmul性能调优案例/Matmul高阶API使能Tiling全量常量化.md

将 Matmul Tiling 参数在编译期通过 GetMatmulApiTiling 预计算，使用 MatmulApiStaticTiling 替代运行时 TCubeTiling，消除运行时 LD/ST 操作。

**Expert implementation:**
```cpp
// Host 侧编译期生成 Tiling
MatmulApiStaticTiling staticTiling;
GetMatmulApiTiling(staticTiling, M, N, K, ...);

// Kernel 侧直接使用常量 Tiling，无需运行时加载
// 总耗时 10.62us→7.87us（25.9%提升）
// Scalar 耗时 6.32us→3.38us（46.5%提升）
```

Benefit: 总耗时降低 25.9%，Scalar 耗时降低 46.5%，Init 阶段耗时降低 38.2%
Trade-off: 仅适用于编译期 shape 已知的场景；shape 变化时需重新编译