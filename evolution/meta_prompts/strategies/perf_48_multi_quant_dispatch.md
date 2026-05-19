# P48: 多量化模式编译期分发 (Multi-Quantization Mode Dispatch)

## Overview
通过 TilingKey + 编译期宏分发支持 A8W8/A4W4/A16W8/A8W4 MSD 等多种量化路径。编译期分支消除运行时开销，每种量化模式生成独立的内核模板。

## When to Use
- 需要支持多种量化精度的 matmul 算子
- A8W8 单一模式无法满足精度/性能需求
- 需要 A4W4（2x 吞吐）或 A8W4 MSD（1.5-2x 吞吐）加速
- 需要 per-channel/per-group 量化支持

## Trade-off
- 编译期分支增加二进制体积（每种模式一个内核）
- TilingKey 组合数可能爆炸（20+ 种），需要裁剪
- MSD 分解需要额外的中间 buffer

**Source operators**: grouped_matmul, grouped_matmul_swiglu_quant_v2, grouped_matmul_finalize_routing, ffn

---

## Variant A: 6 种量化内核模板
Source: grouped_matmul

通过编译期宏定义 6 种量化路径：A8W8O8/O16/O32、A4W4、A16W8、A16W4。

**Expert implementation:**
```cpp
#if defined(GMM_A8W8)
    // int8 × int8 → int32, 支持 O8/O16/O32 输出
    using aT = int8_t; using bT = int8_t; using cT = int32_t;
#elif defined(GMM_A4W4)
    // int4 × int4 直接计算
    using aT = int4b_t; using bT = int4b_t; using cT = int32_t;
#elif defined(GMM_A16W8)
    // Antiquant: int8 权重反量化到 fp16 后与 fp16 输入相乘
    using aT = half; using bT = int8_t; using cT = float;
#endif

// TilingKey 运行时选择
if (TILING_KEY_IS(2)) { /* A8W8 path */ }
if (TILING_KEY_IS(3)) { /* A4W4 path */ }
if (TILING_KEY_IS(4)) { /* A16W8 path */ }
```

**vs. baseline:**
```cpp
// 硬编码 int8×int8→int32 单一路径
// 手动 Cast(int32→float32) + Mul(scale) + Muls(per-token scale)
```

Benefit: 编译期分发零运行时开销，A4W4 吞吐 2x
Trade-off: 6 种内核增加二进制体积

## Variant B: 20+ 种 TilingKey 组合
Source: grouped_matmul_finalize_routing

TilingKey 编码量化模式 + 输出类型 + 后处理选项的组合。

```cpp
// TilingKey 编码: quantMode | outputType | postProcess
// 例: key=0 → A8W8+FP16+None
//     key=1 → A8W8+FP16+TopK
//     key=2 → A8W4+FP16+None
//     ...
if (TILING_KEY_IS(0)) { A8W8_FP16_Kernel(); }
if (TILING_KEY_IS(1)) { A8W8_FP16_TopK_Kernel(); }
```

Benefit: 精细控制每种组合的优化路径
Trade-off: 组合爆炸需要裁剪不常用路径

## Variant C: 三条路径编译期分发
Source: grouped_matmul_swiglu_quant_v2

A8W4 MSD / A4W4 / A8W8 三条主路径，MSD 使用高低位分解 + 两次 int4 matmul。

```cpp
#ifdef GMM_A8W4_MSD
    // MSD: int8 输入分解为高4位和低4位
    // 两次 int4 matmul + 结果合并
    GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_MSD_API_DEQUANT(...)
#elif defined(GMM_A4W4)
    // int4 × int4 直接计算
#elif defined(GMM_A8W8)
    // 标准 int8 × int8
#endif
```

Benefit: A8W4 MSD 吞吐 1.5-2x vs A8W8
Trade-off: MSD 需要额外中间 buffer 存储分解结果

## Variant D: AscendQuant/AscendDequant + MSD 管线
Source: ffn

使用硬件 AscendQuant/AscendDequant API 实现量化/反量化，配合 MSD 分解。

```cpp
AscendQuant(quantResult, fpInput, scale, offset);   // 硬件量化
AscendDequant(dequantResult, intOutput, scale, tmp); // 硬件反量化
// MSD 管线: Quant → Matmul(int4) → Dequant → PostProcess
```

Benefit: 硬件加速量化/反量化，端到端融合
Trade-off: 需要 AscendQuant/AscendDequant 硬件支持
