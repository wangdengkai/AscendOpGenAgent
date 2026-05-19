# P64 避免 GM 同地址访问冲突 (GM Address Conflict Avoidance)
## Overview
MTE2/MTE3/Scalar 访问 GM 时按 512 字节粒度对齐处理，多核同时访问连续 512 字节范围内的地址会被串行处理。通过错位访问顺序或修改切分策略，使各核同一轮访问不同 512B 区域，消除多核 GM 地址冲突。

## When to Use
- 多核并行访问 GM 场景
- 数据行宽 ≤ 512B 时尤其严重
- Profiling 显示 aiv_mte2_time 或 aiv_mte3_time 异常高

## Trade-off
- 可能需要 SyncAll 全核同步配合
- 切分策略变更可能影响其他优化
- 错位访问增加地址计算复杂度

**Source operators**: SIMD算子性能优化/内存访问

---
## Variant A: 错位访问顺序
Source: SIMD算子性能优化/内存访问/避免同地址访问.md

各核按 blockIdx 偏移访问数据块，使同一轮各核访问不同 512B 区域。

**Expert implementation:**
```cpp
// 反例：所有核同一轮访问相同地址范围
for (int i = 0; i < loopOneCore; i++) {
    DataCopy(dst, src[i * blockSize], blockSize);
}

// 正例：错位访问，各核偏移不同
for (int i = 0; i < loopOneCore; i++) {
    int newProgress = (i + GetBlockIdx()) % loopOneCore;
    DataCopy(dst, src[newProgress * blockSize], blockSize);
}
```

Benefit: 消除多核 GM 地址冲突导致的串行等待，显著降低 MTE2/MTE3 时间
Trade-off: 可能需要 SyncAll 全核同步配合

---
## Variant B: 行切分替代列切分
Source: SIMD算子性能优化/内存访问/避免同地址访问.md

从列切分改为行切分，使各核数据在不同地址范围，天然避免 512B 范围内的地址冲突。

**Expert implementation:**
```cpp
// 反例：列切分，各核访问同一行的不同列（地址相邻）
// 核0: row[0:H][0:W/N], 核1: row[0:H][W/N:2W/N]

// 正例：行切分，各核访问不同行（地址分离）
// 核0: row[0:H/N][0:W], 核1: row[H/N:2H/N][0:W]
```

Benefit: 天然避免地址冲突，无需额外同步
Trade-off: 行切分可能导致尾行负载不均
