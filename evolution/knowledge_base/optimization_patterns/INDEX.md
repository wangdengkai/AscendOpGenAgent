# Optimization Patterns Index

Proven AscendC optimization patterns with code examples. Each pattern includes:
- When to apply (operator characteristics)
- Code template
- Measured impact range
- Common mistakes

## Patterns

| File | Pattern | Impact | Applicability | 算子类型标签 |
|------|---------|--------|---------------|-------------|
| `double_buffering.md` | Double buffering for pipeline overlap | 20-80% speedup | All memory-bound kernels | elementwise, reduction, softmax |
| `tiling_strategies.md` | Adaptive tiling (Split-N, Split-D, Split-ND) | 10-50% speedup | Variable-shape operators | reduction, norm, attention |
| `causal_block_skip.md` | Skip invalid blocks in causal/masked attention | 20-50% reduction | Attention operators with causal mask | attention |
| `pipeline_overlap.md` | Fine-grained pipeline stage overlap | 5-30% speedup | Already double-buffered kernels | elementwise, reduction |
| `memory_coalescing.md` | Coalesced memory access patterns | 10-40% speedup | Operators with strided access | transpose, gather, attention |

## Quick Selection Guide

```
Is the kernel memory-bound?
  ├─ Yes: Start with double_buffering.md + tiling_strategies.md
  └─ No (compute-bound or balanced):
      ├─ Has causal/mask logic? → causal_block_skip.md
      ├─ Already has double buffering? → pipeline_overlap.md
      └─ Has strided access? → memory_coalescing.md
```
