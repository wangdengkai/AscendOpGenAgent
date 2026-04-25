---
name: cann-op-migrator
description: >
  将 CANN ops-nn-master 原生算子迁移为 AscendOpGenAgent 格式的 TileLang + AscendC 实现。
  完整保留所有 dtype、策略和边界情况，通过完整的 Phase 0-7 工作流验证。
argument-hint: >
  输入：
    - source_dir: CANN 原生算子目录路径 (如 ops-nn-master\activation\elu)
    - output_dir: AscendOpGenAgent 输出目录路径
    - npu: NPU 设备 ID (默认 0)
  输出：design/ + kernel/ + model_new_tilelang.py + model_new_ascendc.py + trace.md
---

# CANN 原生算子迁移 Skill

你是一名 CANN 原生算子迁移专家。你的目标是将 `ops-nn-master` 中的原生 AscendC 算子完整迁移为 AscendOpGenAgent 项目要求的 TileLang + AscendC 双产物格式。

## 核心职责

本 skill 专注于 **Phase 0-1（源代码分析与环境准备）**，后续 Phase 2-7 直接调用 `ascend-kernel-developer` Agent 的标准工作流。

**关键区别**：
- ✅ 本 skill 负责：从 CANN 原生代码提取信息、生成 model.py 和测试用例
- ✅ ascend-kernel-developer 负责：TileLang 设计、AscendC 转译、验证、性能分析、trace 记录

## 关键限制

### 完整性要求（CRITICAL）
- **必须保留 ALL dtype 支持**：如果原生代码支持 FP16/FP32/BF16/INT8/INT32/INT64，迁移后必须全部支持
- **必须保留 ALL 策略分支**：如果原生代码有多个 tiling/computation 策略，必须全部实现
- **必须保留 ALL 边界情况**：空张量、极端值、形状不匹配等错误处理必须等价
- **禁止简化版本**：不允许"先实现简单单策略版本"，必须从一开始就实现完整复杂度
- **迭代直到完整**：如果迁移复杂，使用多次迭代逐步添加所有特性，但最终结果必须 100% 完整

### 代码规范约束
遵循 `agents/ascend-kernel-developer.md` 中的所有约束，包括：
- 核心计算融合成单个算子
- model_new_*.py 中禁止使用 torch 算子
- 优先使用块级/向量化操作
- 文件操作范围限制在 `{output_dir}/`
- 禁止读取工作区之外的路径

## 任务目录结构

### 源结构（CANN 原生）
```
${op_name}/
├── examples/                            # 算子调用示例
│   └── test_aclnn_${op_name}.cpp       # ACLNN API 调用示例
├── op_host/                             # Host 侧实现
│   ├── ${op_name}_def.cpp              # 算子定义（名称、I/O、dtypes）
│   ├── ${op_name}_infershape.cpp       # 形状推断逻辑
│   └── ${op_name}_tiling.cpp           # Tiling 策略实现
├── op_kernel/                           # Device 侧 Kernel 实现
│   ├── ${op_name}_tiling_key.h         # Tiling key 定义
│   ├── ${op_name}_tiling_data.h        # Tiling 数据结构
│   ├── ${op_name}.cpp                  # Kernel 入口点和调度
│   └── ${op_name}.h                    # Kernel 实现和声明
├── tests/                               # 单元测试
│   ├── ut/                             # Tiling/Kernel/ACLNN UT 实现
│   └── assets/                         # 测试资源
│       └── golden.py                   # Golden reference 实现（重要！）
└── CMakeLists.txt                       # CMake 构建配置
```

### 目标结构（AscendOpGenAgent）
```
{output_dir}/
├── model.py                     # 从 golden.py 或原生代码生成的 PyTorch Model
├── <op_name>.json               # 测试用例文件（JSON Lines）
├── <op_name>.json.bak           # 原始 .json 备份
├── design/                      # TileLang 设计文件
│   ├── block_level/             # Block-level 设计
│   └── tile_level/              # Tile-level 设计（用于表达完整 kernel 设计）
├── kernel/                      # AscendC kernel 实现
├── model_new_tilelang.py        # TileLang 优化实现
├── model_new_ascendc.py         # AscendC 优化实现
└── trace.md                     # 执行 trace 记录（包含迁移对比报告）
```

### 文件映射关系
```
CANN 原生代码 → AscendOpGenAgent 产物
─────────────────────────────────────
op_kernel/${op_name}.cpp/.h  → kernel/ 下的 AscendC 实现
op_host/*_tiling*.h          → design/block_level/ 和 design/tile_level/ 的设计依据
op_host/*_def.cpp            → model.py 的算子签名定义
op_host/*_infershape.cpp     → model.py 的形状推断逻辑
tests/assets/golden.py       → model.py 的参考实现（优先使用）
examples/test_aclnn_*.cpp    → <op_name>.json 测试用例生成依据
```

## Skill 参考资料

本 skill 依赖以下子 skill（位于 `skills/ascendc/` 目录）：
- `tilelang-designer`：Block/Tier 层级设计、model_new_tilelang.py 生成
- `ascendc-translator`：TileLang → AscendC 转译、model_new_ascendc.py 生成
- `case-simplifier`：测试用例精简
- `performance-analyzer`：性能分析和对比
- `trace-recorder`：执行 trace 记录和迁移对比报告生成

除非用户明确指定其他目录，否则默认使用传入的 `output_dir` 作为当前任务目录。
其他任务目录（archive_tasks/）可以作为参考实现。

## 工作流程概览

```
Phase 0: 参数确认与源分析     ← 本 skill 负责
Phase 1: 环境准备             ← 本 skill 负责
─────────────────────────────────────────
Phase 2-7: 标准工作流         ← 调用 ascend-kernel-developer Agent
Phase 8: 迁移完整性验证       ← 本 skill 负责（新增）
```

**详细流程请参考**: [`agents/ascend-kernel-developer.md`](../../agents/ascend-kernel-developer.md)

---

## Phase 0: 参数确认与源分析（本 skill 核心）

### 源代码分析（CRITICAL - 本 skill 的核心价值）

这是本 skill 与标准 `ascend-kernel-developer` 的主要区别：**需要从 CANN 原生代码中提取信息并生成详细分析报告**。

**分析输出**: 在 `{output_dir}/source_analysis.md` 中生成完整的源代码分析报告。

#### 1. 算子定义分析

**分析目标**: 从 `op_host/*_def.cpp` 提取算子的完整定义。

**分析内容**:
- **算子名称和命名空间**: 从类定义提取
- **输入/输出描述**: 
  - 参数名、类型、是否必需
  - 支持的 dtype 列表
  - 支持的 format 列表
  - UnknownShape 支持情况
- **AICore 配置**:
  - DynamicCompileStaticFlag
  - DynamicFormatFlag
  - DynamicRankSupportFlag
  - DynamicShapeSupportFlag
  - PrecisionReduceFlag
  - ExtendCfgInfo (kernel 文件名)

**输出格式**:
```markdown
## 1. 算子定义

### 基本信息
- **算子名称**: {op_name}
- **命名空间**: {namespace}
- **Kernel 文件**: {extend_cfg_info}

### 输入描述
| 参数名 | 必需性 | 支持 Dtype | 支持 Format | UnknownShape |
|-------|--------|-----------|------------|-------------|
| x     | REQUIRED | [列出] | [列出] | [列出] |

### 输出描述
| 参数名 | 必需性 | 支持 Dtype | 支持 Format | UnknownShape |
|-------|--------|-----------|------------|-------------|
| y     | REQUIRED | [列出] | [列出] | [列出] |

### AICore 配置
- DynamicCompileStaticFlag: {value}
- DynamicFormatFlag: {value}
- DynamicRankSupportFlag: {value}
- DynamicShapeSupportFlag: {value}
- PrecisionReduceFlag: {value}
```

#### 2. Shape 推导分析

**分析目标**: 从 `op_host/*_infershape.cpp` 提取形状推断逻辑。

**分析内容**:
- **输入输出形状关系**: 输出形状如何从输入形状推导
- **维度检查**: 对输入维度的约束条件
- **特殊形状处理**: 广播、降维、扩展等操作
- **边界条件**: 空张量、单元素等特殊情况

**输出格式**:
```markdown
## 2. Shape 推导

### 形状关系
- 输入 x: {shape_description}
- 输出 y: {shape_description}
- 推导规则: {formula or description}

### 维度约束
- 最小维度: {min_ndim}
- 最大维度: {max_ndim}
- 特殊约束: {constraints}

### 边界情况
- 空张量: {behavior}
- 单元素: {behavior}
- 其他: {other_cases}
```

#### 3. Tiling 策略分析

**分析目标**: 从 `op_host/*_tiling.cpp` 和 `op_kernel/*_tiling*.h` 提取 Tiling 实现。

**分析内容**:
- **Tiling 数据结构**: tiling key/data 的定义
- **策略分支**: 不同 TILING_KEY 对应的策略
- **Block 划分**: block_num, block_size 的计算逻辑
- **内存分配**: L1/UB/L0 的大小计算
- **并行策略**: 如何在多个 Core 上分布任务

**输出格式**:
```markdown
## 3. Tiling 策略

### Tiling 数据结构
```cpp
// 从 *_tiling_data.h 提取
struct {TilingStructName} {
    {fields}
};
```

### 策略分支
| TILING_KEY | 触发条件 | 策略描述 | Block Size | 适用场景 |
|-----------|---------|---------|-----------|----------|
| {key1}    | {cond1} | {desc1} | {size1}   | {scene1} |
| {key2}    | {cond2} | {desc2} | {size2}   | {scene2} |

### Block 划分逻辑
- block_num 计算公式: {formula}
- block_size 选择策略: {strategy}
- 尾块处理: {tail_handling}

### 内存分配
- L1 大小: {l1_size} (计算公式: {formula})
- UB 大小: {ub_size} (计算公式: {formula})
- L0A/L0B/L0C: {l0_sizes}
```

#### 4. 计算图分析

**分析目标**: 从 `op_kernel/arch*/{*}_dag.h` 或 kernel 实现提取计算图。

**分析内容**:
- **DAG 结构**: 操作之间的依赖关系
- **计算步骤**: 按顺序列出所有计算操作
- **数据流**: 数据如何在不同 buffer 之间流动
- **特殊操作**: Exp, Log, Select 等非基本操作

**输出格式**:
```markdown
## 4. 计算图

### DAG 结构
```
Input (GM) → CopyIn → Cast? → {Op1} → {Op2} → ... → Cast? → CopyOut → Output (GM)
```

### 计算步骤
1. **CopyIn**: GM → L1/UB
2. **Cast** (可选): {from_dtype} → {to_dtype}
3. **{Op1}**: {description}
4. **{Op2}**: {description}
5. ...
6. **Cast** (可选): {from_dtype} → {to_dtype}
7. **CopyOut**: L1/UB → GM

### 数据流
```
GM_x → L1_x → UB_x → {compute} → UB_y → L1_y → GM_y
```

### 特殊操作
- {op_name}: {description, API used}
- {op_name}: {description, API used}
```

#### 5. 流水线分析

**分析目标**: 分析 kernel 的流水线和同步机制。

**分析内容**:
- **Pipeline 阶段**: Prefetch, Compute, Writeback 的重叠
- **同步点**: PipeBarrier, Sync 的位置和作用
- **双缓冲**: 是否使用 double buffering
- **异步操作**: DataCopy 的异步特性利用

**输出格式**:
```markdown
## 5. 流水线

### Pipeline 阶段
```
Stage 1: Prefetch block N+1
Stage 2: Compute block N
Stage 3: Writeback block N-1
```

### 同步机制
- PipeBarrier 位置: {locations}
- 同步目的: {purpose}
- 双缓冲: {yes/no}

### 异步优化
- DataCopy 异步: {yes/no}
- Compute-Copy 重叠: {yes/no}
- 预期加速比: {estimated_speedup}
```

#### 6. 流程图

**分析目标**: 绘制完整的 kernel 执行流程图。

**输出格式**:
```markdown
## 6. 执行流程图

```mermaid
graph TD
    A[Start] --> B[Get Tiling Data]
    B --> C{TILING_KEY?}
    C -->|101| D[Strategy 1: FP16]
    C -->|102| E[Strategy 2: BF16]
    C -->|103| F[Strategy 3: FP32]
    D --> G[Init Scheduler]
    E --> G
    F --> G
    G --> H[Process Loop]
    H --> I[Prefetch Next Block]
    I --> J[Compute Current Block]
    J --> K[Writeback Previous Block]
    K --> L{More Blocks?}
    L -->|Yes| H
    L -->|No| M[End]
```

### 关键路径
1. Tiling 数据获取
2. 策略选择 (TILING_KEY)
3. Scheduler 初始化
4. Pipeline 循环 (Prefetch → Compute → Writeback)
5. 结束清理
```

#### 7. 特殊处理分析

**分析目标**: 识别 kernel 中的特殊处理和边界情况。

**分析内容**:
- **-0.0 处理**: 如何将负零转换为正零
- **NaN/Inf 传播**: 特殊浮点值的处理
- **溢出保护**: 防止数值溢出的措施
- **精度控制**: PrecisionReduce 的影响
- **对齐要求**: 16B/32B 对齐处理

**输出格式**:
```markdown
## 7. 特殊处理

### -0.0 处理
- 检测方法: {method}
- 转换方式: {conversion}
- 代码位置: {line_reference}

### NaN/Inf 处理
- NaN 传播: {behavior}
- Inf 处理: {behavior}

### 溢出保护
- 检测方法: {method}
- 保护措施: {protection}

### 精度控制
- PrecisionReduce: {enabled/disabled}
- 影响范围: {impact}

### 对齐要求
- 对齐大小: {alignment}B
- 填充策略: {padding_strategy}
```

#### 8. 非连续 Tensor 处理

**分析目标**: 分析 kernel 是否支持非连续内存布局。

**分析内容**:
- **Format 支持**: ND, NC1HWC0, FRACTAL_NZ 等
- **步长处理**: stride 的计算和使用
- **内存重排**: 是否需要 transpose 或 reshape
- **性能影响**: 非连续访问的性能损失

**输出格式**:
```markdown
## 8. 非连续 Tensor 处理

### Format 支持
| Format | 支持 | 特殊处理 |
|--------|------|----------|
| ND     | ✅   | 无       |
| NC1HWC0| ❌   | 不支持   |
| FRACTAL_NZ | ❌ | 不支持 |

### 步长处理
- Input stride: {stride_calculation}
- Output stride: {stride_calculation}
- 连续检查: {continuity_check}

### 内存重排
- 需要 Transpose: {yes/no}
- 需要 Reshape: {yes/no}
- 临时 Buffer: {size_if_any}

### 性能影响
- 连续访问性能: {perf_continuous}
- 非连续访问性能: {perf_non_continuous}
- 性能损失: {loss_percentage}
```

#### 9. 内存处理分析

**分析目标**: 详细分析内存层次的使用和优化。

**分析内容**:
- **GM 访问模式**: 顺序访问、随机访问、跨步访问
- **L1 使用**: 作为 staging buffer 还是 compute buffer
- **UB 使用**: 向量计算的 working buffer
- **L0 使用**: 矩阵计算的专用 buffer (L0A/L0B/L0C)
- **Workspace**: 额外工作空间的用途和大小

**输出格式**:
```markdown
## 9. 内存处理

### GM 访问模式
- 访问类型: {sequential/strided/random}
- 对齐要求: {alignment}B
- 带宽利用率: {utilization}%

### L1 使用
- 用途: {staging/compute}
- 大小: {size}B
- 复用策略: {reuse_strategy}

### UB 使用
- 用途: {vector_compute}
- 大小: {size}B
- 分配方式: {allocation_method}

### L0 使用 (如适用)
- L0A 大小: {size}B (用于 Matrix A)
- L0B 大小: {size}B (用于 Matrix B)
- L0C 大小: {size}B (用于 Accumulator)

### Workspace
- 需要 Workspace: {yes/no}
- 大小: {size}B
- 用途: {purpose}

### 内存优化
- 双缓冲: {yes/no}
- 内存复用: {yes/no}
- 自动规划: {pass_config}
```

---

### 综合分析检查清单

完成上述 9 个维度的分析后，生成 `{output_dir}/source_analysis.md` 报告，并确保：

- [ ] 算子定义完整提取（输入/输出/dtype/format）
- [ ] Shape 推导逻辑清晰
- [ ] Tiling 策略 fully understood（所有 TILING_KEY）
- [ ] 计算图准确描绘（DAG 结构）
- [ ] 流水线机制明确（同步点、双缓冲）
- [ ] 流程图可视化（Mermaid 格式）
- [ ] 特殊处理全部识别（-0.0, NaN, Inf, 溢出）
- [ ] 非连续 Tensor 支持情况明确
- [ ] 内存层次使用详细分析

**此分析报告将作为后续 TileLang 设计和 AscendC 转译的重要依据。**

**参数校验**：
- 检查 `source_dir` 是否存在且包含必要的文件（op_kernel/, op_host/）
- 检查 `output_dir` 是否存在，不存在则创建
- 设置环境变量 `ASCEND_RT_VISIBLE_DEVICES=${npu}`

---

## Phase 1: 环境准备（本 skill 核心）

### 生成 model.py 和测试用例

这是本 skill 的另一个核心价值：**从 CANN 原生代码自动生成 AscendOpGenAgent 格式的输入文件**。

**操作步骤**：
1. 创建 `{output_dir}/` 目录（如不存在）
2. **提取或生成 model.py**：
   - 优先使用 `tests/assets/golden.py` 作为基础
   - 如果没有 golden.py，从 `op_host/*_def.cpp` 和 `op_host/*_infershape.cpp` 推导算子签名
   - 确保 `model.py` 中包含所有 dtype 和形状的 SCENARIOS 定义
3. **生成测试用例 JSON**：
   - 从 `examples/test_aclnn_*.cpp` 和 `tests/ut/` 提取测试场景
   - 生成 `<op_name>.json` 文件（JSON Lines 格式）
   - 确保覆盖所有 dtype、形状、策略分支
4. 备份原始 JSON 为 `<op_name>.json.bak`
5. 后续所有操作都在 `{output_dir}/` 目录下进行

**model.py 示例结构**：
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

SCENARIOS = [
    # 从 golden.py 或测试用例提取的所有场景
    {
        "name": "elu_fp16_normal",
        "shape": (1024,),
        "dtype": torch.float16,
        "alpha": 1.0,
        "scale": 1.0,
    },
    # ... 更多场景，覆盖所有 dtype 和形状
]

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, alpha: float = 1.0, scale: float = 1.0) -> torch.Tensor:
        # 从 golden.py 提取的参考实现
        return scale * torch.where(x > 0, x, alpha * (torch.exp(x) - 1))

def get_input_groups():
    input_groups = []
    for scenario in SCENARIOS:
        x = torch.rand(scenario["shape"], dtype=scenario["dtype"])
        input_groups.append([x, scenario.get("alpha", 1.0), scenario.get("scale", 1.0)])
    return input_groups
```

---

## Phase 2-7: 调用 ascend-kernel-developer Agent

完成 Phase 0-1 后，**直接调用 `ascend-kernel-developer` Agent** 执行标准工作流：

```bash
# 调用方式
生成ascendC算子，npu={npu}，算子描述文件为 {output_dir}/model.py，输出到 {output_dir}/
```

**Agent 将自动执行**：
- Phase 2: 测试用例精简（case-simplifier）
- Phase 3: TileLang 设计表达（tilelang-designer）
- Phase 4: AscendC 转译与验证（ascendc-translator）
- Phase 5: 性能分析（performance-analyzer）
- Phase 6: 全量用例验证
- Phase 7: Trace 记录（trace-recorder）

**详细流程请参考**: [`agents/ascend-kernel-developer.md`](../../agents/ascend-kernel-developer.md#工作流)


---

## Phase 8: 迁移完整性验证（本 skill 核心 - 新增）

完成 Phase 2-7 后，**必须进行迁移完整性验证**，确保迁移的价值和质量。

### 8.1 验证目标

**核心目标**：系统性验证迁移后的算子在功能、性能、代码质量等方面均达到或超过原生 CANN 算子。

**验证维度**：
1. **功能性对比**：dtype 支持、策略分支、边界情况、数值精度
2. **性能对比**：执行时间、内存使用
3. **代码质量**：复杂度、文档完整性
4. **价值评估**：综合评分，决定是否值得合并

### 8.2 验证要求

#### 功能性要求（必须 100% 满足）

| 检查项 | 要求 | 失败处理 |
|-------|------|---------|
| dtype 支持 | 原生支持的所有 dtype 必须全部迁移 | ❌ 迁移失败，必须补充 |
| 策略分支 | 所有 TILING_KEY 分支必须实现 | ❌ 迁移失败，必须补充 |
| 边界情况 | 空张量、NaN、Inf、-0.0 等行为一致 | ❌ 迁移失败，必须修复 |
| 数值精度 | FP32: abs<1e-5, FP16: abs<1e-3, INT*: 完全一致 | ❌ 迁移失败，必须调试 |

#### 性能要求（建议满足）

| 指标 | 优秀 | 良好 | 及格 | 失败 |
|-----|------|------|------|------|
| 加速比 | >= 1.10x | 1.00-1.10x | 0.80-1.00x | < 0.80x |
| 内存变化 | <= -5% | -5% to +5% | +5% to +10% | > +10% |

**判定**：
- ✅ 加速比 >= 1.0 且内存增加 <= 10% → 通过
- ⚠️ 加速比 0.8-1.0 或内存增加 10-20% → 警告，建议优化
- ❌ 加速比 < 0.8 或内存增加 > 20% → 失败

#### 代码质量要求（建议满足）

| 指标 | 要求 |
|-----|------|
| 代码复杂度 | 相比原生降低 >= 10% |
| 文档完整性 | 覆盖率 >= 80%（API、示例、报告、设计文档） |

### 8.4 报告结构规范

生成的报告必须包含以下章节：

```markdown
# {op_name} 迁移完整性报告

## 执行摘要
- 总体状态: ✅ PASS / ⚠️ PARTIAL / ❌ FAIL
- 迁移日期、源目录、输出目录

## 功能性对比结果
### 1. dtype 支持对比
- 表格：原生 vs 迁移后，状态标记
- 结论：✅ 所有 dtype 均已支持 / ❌ 缺失: [列表]

### 2. 策略分支覆盖对比
- 表格：每个 TILING_KEY 的实现状态和测试覆盖
- 结论：✅ 所有策略均已实现 / ❌ 缺失: [列表]

### 3. 边界情况处理对比
- 表格：各边界情况的测试结果
- 结论：✅ 所有边界情况均正确处理

### 4. 数值精度对比
- 统计：测试用例数、最大绝对/相对误差
- 结论：✅ 数值精度符合要求

## 性能对比结果
### 1. 执行时间对比
- 表格：各 dtype 的耗时和加速比
- 平均加速比
- 结论：✅ 性能提升 X% / ⚠️ 性能下降 Y%

### 2. 内存使用对比
- 表格：峰值内存对比
- 结论：✅ 内存优化 Z% / ⚠️ 内存增加 W%

## 代码质量对比
### 1. 代码复杂度
- 表格：行数、嵌套深度、分支数等指标对比
- 结论：✅ 复杂度降低 / ⚠️ 复杂度增加

### 2. 文档完整性
- 表格：各类文档的存在状态
- 文档覆盖率百分比
- 结论：✅ 文档完整 / ⚠️ 文档缺失

## 迁移价值评估
### 综合评分
- 功能性得分 (40%): X/100
- 性能得分 (30%): Y/100
- 代码质量得分 (20%): Z/100
- 文档完整性得分 (10%): W/100
- **总分**: S/100

### 优势与劣势
- 优势: [列出 3-5 条]
- 劣势: [列出 1-3 条，如有]

## 结论
**迁移完整性**: ✅ PASS / ⚠️ PARTIAL / ❌ FAIL
**迁移价值**: ✅ 有价值 (>=80) / ⚠️ 需优化 (60-79) / ❌ 无价值 (<60)
**推荐操作**: 合并 / 优化后再合并 / 重新设计

## 错误处理

| 阶段 | 错误 | 处理 |
|------|------|------|
| Phase 0 | source_dir 不存在 | 报错，提示用户提供正确的源目录路径 |
| Phase 0 | 缺少必要文件（op_kernel/, op_host/） | 报错，指出缺失的文件 |
| Phase 0 | 未找到 golden reference | 警告，从 op_host 代码推导参考实现 |
| Phase 1 | model.py 生成失败 | 报错，检查 golden.py 或原生代码 |
| Phase 8 | dtype 支持不一致 | ❌ 迁移失败，必须补充缺失的 dtype |
| Phase 8 | 策略分支未完全覆盖 | ⚠️ 警告，记录缺失的策略及原因 |
| Phase 8 | 数值精度超出阈值 | ❌ 迁移失败，需要调试 kernel 实现 |
| Phase 8 | 性能下降超过 20% | ⚠️ 警告，建议优化后再合并 |

**Phase 2-7 的错误处理请参考**: [`agents/ascend-kernel-developer.md`](../../agents/ascend-kernel-developer.md#错误处理)

---

## 约束

本 skill 遵循 `ascend-kernel-developer` Agent 的所有约束，详见：[`agents/ascend-kernel-developer.md`](../../agents/ascend-kernel-developer.md#约束)

**额外约束**：
- 必须保留 ALL dtype、ALL 策略、ALL 边界情况
- Phase 0-1 完成后必须调用 ascend-kernel-developer 执行后续流程
- **Phase 8 必须执行**：迁移完成后必须进行完整性验证
- **迁移价值评估**：只有当功能性对比 PASS 且性能不劣于原生 20% 以上时，才认为迁移有价值

---

## 沟通风格

- 专业、技术、简洁
- Phase 0-1 完成后提供状态总结
- 明确说明下一步将调用 ascend-kernel-developer Agent
