# AscendOpGenAgent

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

中文 | [English](README.en.md)

**AscendOpGenAgent** 是一个面向昇腾（Ascend）NPU 的自动化算子生成与评测框架。本项目内置**进化式性能优化系统**，通过世界模型驱动的搜索策略、多层级 Profiling 诊断和多 Agent 协同，自动搜索出高性能 AscendC 算子实现，实现从"能用"到"好用"的性能跃迁。

```
                    ┌─────────────────────────────────────────┐
                    │           进化系统核心理念              │
                    │                                         │
                    │   World Model  ──→  Strategy Search     │
                    │        ↑                    │           │
                    │        │              Parallel Gen      │
                    │        │                    │           │
                    │   Evidence ←── Profiling ←─ Evaluate    │
                    └─────────────────────────────────────────┘
```

## 核心亮点

- **世界模型驱动的定向进化** — 不是随机搜索，而是通过决策树持续积累证据，用效用函数选择最有价值的优化方向
- **多层级 Profiling 闭环** — CSV 级快速诊断 + 指令级深度空泡分析，精准定位 MTE2/MTE3 搬运瓶颈、D 类/C 类空泡、跨核负载不均衡
- **Profiling 驱动的自适应优化** — 不依赖固定策略库，根据实际 profiling 瓶颈数据自主设计针对性优化方案
- **59 条人工编写策略 + 动态策略发现** — 策略库覆盖双缓冲、自适应分块、混合精度等维度，open_exploration 模式可自动发现并提炼新策略
- **Supervisor Agent** — 当进化停滞或 profiling 无法定向时，引入外部视角打破局部最优
- **双管线支持** — lingxi-evo（TileLang→AscendC 全流程）和 ops-evo（直接优化已有 AscendC 代码）

---

## 进化优化系统

### 架构概览

本项目提供两条进化管线，覆盖不同的算子开发场景：

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         Evolution Pipelines                    │
├──────────────────────────────────┬───────────────────────────────────────┤
│    lingxi-evo (TileLang 管线)    │       ops-evo (直接修改管线)          │
├──────────────────────────────────┼───────────────────────────────────────┤
│ 输入: PyTorch Model 文件         │ 输入: ops仓已有算子代码               │
│ 流程: Model→TileLang→AscendC    │ 流程: 直接修改 kernel/tiling 代码     │
│ 子Agent: lingxi-partial         │ 子Agent: ops-partial                 │
│ 并行: 多子Agent并行生成+评估     │ 并行: 多子Agent并行生成, 串行构建评估 │
│ 场景: 新算子从零开发             │ 场景: 已有算子性能优化                │
└──────────────────────────────────┴───────────────────────────────────────┘
```

### 世界模型决策树

进化的核心是一棵持久化的决策树（`world_model.json`），每个节点代表一个优化方向：

```
                            root (baseline, 1.0x)
                         ┌────┼────┬────┐
                        n1   n2   n3   n4
                     (2.3x) (1.7x) [F]  ...
                    ┌──┴──┐
                  n1_1  n1_2          ← 在最优分支上深度探索
                 (2.8x)  ...
                   │
                 n1_1_1              ← profiling_driven: 基于瓶颈诊断的定向优化
```

节点属性包括：
- **strategy_combination** — 应用的策略组合（如 `["P1", "P7"]` 双缓冲+对齐）
- **mode** — 探索模式：`strategy_guided` | `open_exploration` | `profiling_driven`
- **score** — 相对基线的加速比
- **profiling_insight** — CSV 级瓶颈诊断（每轮必做）
- **profiling_evidence** — 指令级深度空泡分析（条件触发）

### 进化循环

每轮进化执行以下步骤：

```
┌─────────────────────────────────────────────────────────────────────┐
│                        单轮进化流程                                  │
│                                                                     │
│  ┌──────┐     ┌──────┐     ┌──────┐     ┌──────┐     ┌──────────┐ │
│  │Select│ ──→ │ Gen  │ ──→ │Build │ ──→ │ Eval │ ──→ │ Analyze  │ │
│  │ 选节点│     │并行生成│     │ 编译 │     │ 评估 │     │Profiling │ │
│  └──────┘     └──────┘     └──────┘     └──────┘     │+ Refine  │ │
│      ↑                                                └────┬─────┘ │
│      │                                                     │       │
│      └─────────────── 更新世界模型 ←───────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

1. **Select** — 效用函数评估所有 open 节点，选取最有价值的 N 个方向
2. **Generate** — 并行启动子 Agent，每个按指定策略组合生成/修改代码
3. **Build & Evaluate** — 编译、精度验证、性能测试（支持 build retry）
4. **Profiling & Analyze** — CSV 级 + 深度空泡分析，诊断瓶颈类型
5. **Refine** — 基于评测结果更新决策树：生成子节点（深度优化）、标记失败、累积证据

### 三种探索模式

| 模式 | 触发条件 | 行为 |
|------|---------|------|
| `strategy_guided` | 默认 | 从 59 条策略库中选取组合，应用到代码生成 |
| `open_exploration` | 连续停滞 ≥2 轮 | 禁止读取策略库，从最优代码+profiling 自主推理新方向 |
| `profiling_driven` | 深度空泡分析触发 | 基于具体 profiling 瓶颈数据（如"MTE2 stall 62%"），设计针对性 patch |

### Profiling 管线

本项目实现了两层 profiling 诊断，形成完整的性能分析闭环：

```
┌──────────────────────────────────────────────────────────────────┐
│                      Profiling Pipeline                          │
│                                                                  │
│  第一层: CSV 级快速诊断 (每轮必做)                                │
│  ┌─────────────────────────────────────────────────────┐        │
│  │ analyze_profiling.py → bottleneck + pipeline_summary │        │
│  │ 输出: memory_bound / compute_bound / scalar_bound   │        │
│  └───────────────────────────┬─────────────────────────┘        │
│                              │                                   │
│              ┌───────────────┴───────────────┐                  │
│              │ 条件触发: balanced + 性能远不达标 │                  │
│              └───────────────┬───────────────┘                  │
│                              ↓                                   │
│  第二层: 指令级深度空泡分析 (条件触发)                             │
│  ┌─────────────────────────────────────────────────────┐        │
│  │ run_deep_profiling.py → profiling_evidence           │        │
│  │ 输出: D类空泡%, C类空泡%, 跨核不均衡比,               │        │
│  │       主等待流水线, 空泡模式, DMA效率                  │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                  │
│  诊断结果 → 注入子Agent的 [Profiling Context] → 指导下轮优化      │
└──────────────────────────────────────────────────────────────────┘
```

### 策略库

`evolution/meta_prompts/strategies/` 包含 59 条人工编写策略，分为三大类：

| 类型 | 前缀 | 数量 | 示例 |
|------|------|------|------|
| 性能优化 | `perf_` | 46 条（P1-P52） | P1 双缓冲、P2 自适应分块、P4 多核负载均衡、P10 向量化搬运 |
| 精度保障 | `acc_` | 8 条（A1-A8） | A1 FP32中间计算、A2 Welford算法、A6 高精度Rsqrt |
| 数据类型 | `dtype_` | 5 条（D1-D5） | D1 混合精度、D2 模板化内核、D4 FP8/INT4转换 |

open_exploration 模式可自动发现新策略（`disc_X*.md`），扩展策略库。

### Supervisor Agent

当进化陷入停滞时，进化系统会自动引入 Supervisor Agent 提供外部视角：

| 触发条件 | 场景 |
|---------|------|
| 连续停滞 ≥ 阈值轮 | 策略搜索空间可能已饱和 |
| 深度 profiling 无法定向 | 瓶颈为 near_optimal/balanced 但性能仍远不达标 |
| profiling_driven 全失败 | 基于 profiling 的定向优化方向均未奏效 |

Supervisor 从 8 个维度（算法级优化、硬件架构利用、数据流重组等）分析后，注入新的 open 节点到决策树。

### 知识库

`evolution/knowledge_base/` 提供领域知识支撑：

```
knowledge_base/
├── INDEX.md                    # 知识库索引
├── algorithm_insights/         # 算法洞察（归约、Softmax、LayerNorm等）
├── ascendc_api/                # AscendC API 参考
├── hardware/                   # 硬件规格（UB大小、核数、带宽）
├── optimization_patterns/      # 优化模式（双缓冲、分块、向量化）
└── proven_solutions/           # 已验证的优化方案
```

---

## 单次生成（lingxi Agent）

除了进化优化，本项目也支持单次算子生成，适用于快速原型开发：

```
PyTorch Model → 用例精简 → TileLang 设计（block/tile level）
                                          ↓
评估结果 ← 全量验证 ← 性能分析 ← AscendC 转译（迭代验证）
```

### 8 Phase 管线

| Phase | Skill | 输出 |
|-------|-------|------|
| 0 | 参数确认 | npu, op_file, output_dir |
| 1 | 环境准备 | model.py, 测试用例复制 |
| 2 | case-simplifier | 精简后的测试用例（≤10） |
| 3 | tilelang-designer | design/block_level/, design/tile_level/ |
| 4 | ascendc-translator | kernel/, model_new_ascendc.py |
| 5 | performance-analyzer | 性能对比报告 |
| 6 | 全量验证 | 全量用例验证结果 |
| 7 | trace-recorder | trace.md 执行记录 |

Phase 3-4 支持迭代修复：AST 退化检测（validate_*.py）+ Conductor 分析（A/B/C 错误分类）。

---

## 快速开始

### 环境准备

```bash
# python >= 3.10, CANN >= 8.3.RC1, torch-npu >= 2.6.0.RC1, Ascend NPU

# 1. 设置 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. 安装 tilelang-ascend（源码编译）
git clone --recursive https://github.com/tile-ai/tilelang-ascend.git
cd tilelang-ascend
bash install_ascend.sh
source set_env.sh
cd ..

# 3. 安装其他依赖
pip install torch_npu numpy attrs pyyaml decorator scipy psutil protobuf
```

### 使用示例

**lingxi 示例**:
```
用户: 生成 ascendC 算子，npu=6，算子描述文件为 /path/to/31_ELU.py，输出到 /path/to/output/31_ELU/
```

**lingxi-evo 示例**:
```
用户: 进化优化算子

配置:
- NPU 设备号: 6
- 算子文件: /path/to/31_ELU.py
- 进化轮数: 3
- 并行数: 5
- 目标加速比: 2.0
```

**ops-evo 示例**:
```
用户: 优化 ops-nn 仓中的 HardShrink 算子

配置:
- 算子路径: /path/to/ops-nn/built-in/HardShrink
- 进化轮数: 3
- 并行数: 4
- 目标加速比: 1.5
```

### 输出结构

**单次生成**:
```
output/{op_name}/
├── model.py                     # 算子描述（PyTorch Model）
├── <op_name>.json               # 测试用例（精简后）
├── design/                      # TileLang 设计
│   ├── block_level/
│   └── tile_level/
├── kernel/                      # AscendC 内核代码
├── model_new_ascendc.py         # AscendC 调用 wrapper
├── evaluation_results.json      # 评估结果
└── trace.md                     # 执行 trace
```

**进化优化（lingxi-evo）**:
```
output/{op_name}_evo_{timestamp}/
├── world_model.json              # 世界模型决策树
├── solution_db.jsonl             # 所有变体的血统追踪
├── shared/                       # 共享文件（所有变体复用）
│   ├── model.py
│   ├── <op_name>.json
│   └── design/
├── round_1/
│   ├── parallel_0/               # 变体 0（含 kernel/, evaluation_results.json）
│   ├── parallel_1/               # 变体 1
│   └── ...
├── round_2/
│   └── ...
└── profiling/                    # Profiling 数据
```

---

## 单用例多 Shape 支持

本框架支持在一个算子用例中定义多个 Shape 配置进行批量验证和性能评测，适用于需要测试算子在不同规模输入下的性能表现的场景。

### 输入规格（算子描述文件）

#### 单 Shape 格式（向后兼容）

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x)

def get_inputs():
    """返回单组输入，形式为 List[Tensor/...]"""
    return [torch.randn(128, 128, dtype=torch.float16)]

def get_init_inputs():
    """返回初始化参数列表"""
    return []
```

**规格说明**：
- `get_inputs()`: 返回 `List[Tensor/...]`，代表单组输入
- 适用于单一 Shape 场景
- `get_init_inputs()`: 返回 `__init__` 的初始化参数列表

#### 多 Shape 格式

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
    def forward(self, x: torch.Tensor, approximate='none') -> torch.Tensor:
        return torch.nn.functional.gelu(x, approximate=approximate)

# 多 Shape 配置列表
INPUT_CASES = [
    {'inputs': [{'dtype': 'float32', 'name': 'x', 'shape': [128, 128], 'type': 'tensor'},
                 {'dtype': 'str', 'name': 'approximate', 'type': 'attr', 'value': 'none'}]},
    {'inputs': [{'dtype': 'float32', 'name': 'x', 'shape': [256, 256], 'type': 'tensor'},
                 {'dtype': 'str', 'name': 'approximate', 'type': 'attr', 'value': 'tanh'}]},
    {'inputs': [{'dtype': 'float16', 'name': 'x', 'shape': [1024, 1024], 'type': 'tensor'},
                 {'dtype': 'str', 'name': 'approximate', 'type': 'attr', 'value': 'none'}]},
]

# 必须实现，返回 List[List[Tensor/...]]
def get_input_groups():
    """返回多组输入列表，每组对应一个 Shape 配置"""
    input_groups = []
    for case in INPUT_CASES:
        group = []
        for spec in case['inputs']:
            if spec['type'] == 'tensor':
                dtype = {'float16': torch.float16, 'float32': torch.float32}[spec['dtype']]
                group.append(torch.randn(*spec['shape'], dtype=dtype))
            elif spec['type'] == 'attr':
                group.append(spec['value'])
        input_groups.append(group)
    return input_groups

# 可选实现，用于向后兼容
def get_inputs():
    """返回单组输入，取第一组"""
    return get_input_groups()[0]

def get_init_inputs():
    """返回初始化参数列表"""
    return []
```

**输入规格说明**：

| 函数 | 返回类型 | 用途 | 必需 |
|------|---------|------|------|
| `get_input_groups()` | `List[List[Tensor/...]]` | 多 Shape 入口，每组对应一个测试配置 | 多 Shape 场景必需 |
| `get_inputs()` | `List[Tensor/...]` | 单 Shape 入口，返回第一组或单组输入 | 建议实现（向后兼容） |
| `get_init_inputs()` | `List[Any]` | `Model.__init__` 的初始化参数 | 必需 |

**输入配置字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `dtype` | `str` | 数据类型：float16/float32/float64/bfloat16/int8/int16/int32/int64/bool |
| `shape` | `List[int]` | 张量形状，如 `[128, 256]` |
| `name` | `str` | 参数名称 |
| `type` | `str` | 类型："tensor"（张量）、"attr"（属性值）、"tensor_list"（张量列表） |
| `value` | `Any` | 当 `type="attr"` 时，属性值 |

### 输出规格（性能报告）

#### 单 Shape 性能报告

```json
{
  "op_name": "gelu",
  "warmup": 5,
  "repeats": 50,
  "total_cases": 1,
  "passed_cases": 1,
  "failed_cases": 0,
  "nan_indices": [],
  "inf_indices": [],
  "zero_indices": [],
  "negative_indices": [],
  "none_indices": [],
  "framework": {
    "avg_latency_ms": 0.2345,
    "peak_memory_mb": 2.50,
    "operators": {}
  },
  "implementation": {
    "avg_latency_ms": 0.1567,
    "peak_memory_mb": 1.25,
    "operators": {}
  },
  "speedup_vs_torch": 1.4965,
  "per_shape_results": [
    {
      "case_idx": 1,
      "input_desc": [{"type":"tensor","shape":[1024,1024],"dtype":"torch.float16"}],
      "status": "pass",
      "framework": {"avg_latency_ms": 0.2345, "peak_memory_mb": 2.50},
      "implementation": {"avg_latency_ms": 0.1567, "peak_memory_mb": 1.25},
      "speedup_vs_torch": 1.4965,
      "error_type": null,
      "error_msg": null
    }
  ]
}
```

#### 多 Shape 性能报告

```json
{
  "op_name": "gelu",
  "warmup": 5,
  "repeats": 50,
  "total_cases": 3,
  "passed_cases": 3,
  "failed_cases": 0,
  "nan_indices": [],
  "inf_indices": [],
  "zero_indices": [],
  "negative_indices": [],
  "none_indices": [],
  "framework": {
    "avg_latency_ms": 0.4567,
    "peak_memory_mb": 8.50,
    "operators": {}
  },
  "implementation": {
    "avg_latency_ms": 0.3123,
    "peak_memory_mb": 4.25,
    "operators": {}
  },
  "speedup_vs_torch": 1.4910,
  "per_shape_results": [
    {
      "case_idx": 1,
      "input_desc": [{"type":"tensor","shape":[128,128],"dtype":"torch.float16"}],
      "status": "pass",
      "framework": {"avg_latency_ms": 0.0234, "peak_memory_mb": 0.50},
      "implementation": {"avg_latency_ms": 0.0156, "peak_memory_mb": 0.25},
      "speedup_vs_torch": 1.5000,
      "error_type": null,
      "error_msg": null
    },
    {
      "case_idx": 2,
      "input_desc": [{"type":"tensor","shape":[256,256],"dtype":"torch.float16"}],
      "status": "pass",
      "framework": {"avg_latency_ms": 0.0891, "peak_memory_mb": 2.00},
      "implementation": {"avg_latency_ms": 0.0588, "peak_memory_mb": 1.00},
      "speedup_vs_torch": 1.5153,
      "error_type": null,
      "error_msg": null
    },
    {
      "case_idx": 3,
      "input_desc": [{"type":"tensor","shape":[1024,1024],"dtype":"torch.float16"}],
      "status": "pass",
      "framework": {"avg_latency_ms": 1.2577, "peak_memory_mb": 8.00},
      "implementation": {"avg_latency_ms": 0.8625, "peak_memory_mb": 12.50},
      "speedup_vs_torch": 1.4582,
      "error_type": null,
      "error_msg": null
    }
  ]
}
```

**输出字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `op_name` | `str` | 算子名称 |
| `warmup` | `int` | 预热次数 |
| `repeats` | `int` | 正式测试次数 |
| `total_cases` | `int` | 测试的 Shape 数量（单 Shape 为 1，多 Shape >=2） |
| `passed_cases` / `failed_cases` | `int` | 多 Shape 通过 / 失败用例数（异常 `s_i` 的 shape 仍计入 `passed_cases`）|
| `nan_indices` / `inf_indices` / `zero_indices` / `negative_indices` / `none_indices` | `List[int]` | 各类异常 `s_i` 的 case_idx 列表（从 1 开始，不进入几何平均）；无异常时为 `[]` |
| `framework.avg_latency_ms` | `float` | PyTorch 实现平均延迟（毫秒），各 Shape 算术平均（兼容语义）|
| `framework.peak_memory_mb` | `float` | PyTorch 峰值内存（MB）各 Shape 平均 |
| `implementation.avg_latency_ms` | `float` | 实现平均延迟（毫秒），各 Shape 算术平均（兼容语义）|
| `implementation.peak_memory_mb` | `float` | 实现峰值内存（MB）各 Shape 平均 |
| `speedup_vs_torch` | `float\|null` | **几何平均加速比** = `(prod s_i)^(1/n)`，仅对 status==pass 且 `s_i` 为有限正数的 Shape；全部异常时为 `null` |
| `perf_method` | `str` | 评测方式："profiler"（torch_npu.profiler）或 "fallback"（time.perf_counter 兜底） |
| `skill_path` | `str` | 使用的 benchmark skill 路径 |
| `per_shape_results` | `List[Dict]` | 各 Shape 明细数据（永远存在，含失败用例）|

**per_shape_results 元素说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `case_idx` | `int` | 用例序号（从 1 开始）|
| `input_desc` | `List[Dict]` | 输入结构化描述（tensor: shape+dtype；scalar: value）|
| `status` | `str` | `"pass"` 或 `"fail"` |
| `framework` / `implementation` | `Dict\|null` | pass 时含 `avg_latency_ms`、`peak_memory_mb`；fail 时为 null |
| `speedup_vs_torch` | `float\|null` | 该 Shape 的加速比；fail 或 `s_i` 异常（NaN/Inf/0/负数/None）时为 null |
| `error_type` / `error_msg` | `str\|null` | fail 时记录异常类型与堆栈（截断 2000 字符）|

### 适用场景

1. **算子泛化性测试**：验证生成的算子在多种输入规模下的正确性和稳定性
2. **性能趋势分析**：通过对比不同 Shape 的加速比，识别算子的优势和局限性
3. **AI 模型场景复现**：模拟真实模型中的典型输入 Shape 分布（如 LLM 的多种序列长度）
4. **自动 Benchmark 评测**：批量评测时自动覆盖多种 Shape，减少重复工作量

---

## 项目结构

```
AscendOpGenAgent/
├── .claude/
│   ├── agents/                   # Agent 定义
│   │   ├── lingxi.md             # 单次生成 Agent（TileLang→AscendC）
│   │   ├── lingxi-evo.md         # TileLang 进化 Agent
│   │   ├── lingxi-partial.md     # TileLang 进化子 Agent
│   │   ├── ops-evo.md            # Ops 进化 Agent
│   │   └── ops-partial.md        # Ops 进化子 Agent
│   └── skills/                   # Skill 模块
│       ├── tilelang-designer/    # TileLang 设计（block/tile level）
│       ├── ascendc-translator/   # AscendC 转译（含 API 知识库）
│       ├── case-simplifier/      # 测试用例精简
│       ├── performance-analyzer/ # 性能分析
│       ├── trace-recorder/       # Trace 记录
│       ├── ascendc-profiling/    # CSV 级 Profiling 诊断
│       ├── ascendc-profiling-analysis/ # 指令级深度空泡分析
│       ├── hardware-specs-query/ # 硬件规格查询
│       └── ...
├── evolution/
│   ├── world_model/              # 世界模型
│   │   ├── schema.md             # JSON Schema 定义
│   │   ├── operations.md         # 世界模型操作协议
│   │   ├── wm_ops.py             # 世界模型操作脚本（refine/select/diagnose）
│   │   ├── check_round_artifacts.py  # 产物完整性检查
│   │   └── solution_db.py        # 变体血统追踪
│   ├── meta_prompts/
│   │   ├── strategies/           # 28 条优化策略
│   │   ├── strategy-index.md     # 策略索引
│   │   ├── lingxi-partial-prompt.md  # 子 Agent prompt 模板
│   │   └── supervisor-prompt.md  # Supervisor prompt 模板
│   └── knowledge_base/           # 领域知识库
│       ├── algorithm_insights/
│       ├── optimization_patterns/
│       ├── proven_solutions/
│       └── hardware/
├── utils/                        # 辅助脚本
│   ├── build_ascendc.py          # AscendC 构建
│   ├── performance.py            # 性能测试
│   ├── verification_ascendc.py   # AscendC 验证
│   └── verification_tilelang.py  # TileLang 验证
└── CLAUDE.md                     # 项目配置
```

## 依赖

- Python 3.10+
- CANN >= 8.3.RC1
- torch-npu >= 2.6.0.RC1
- [tilelang-ascend](https://github.com/tile-ai/tilelang-ascend)（源码编译）
- `numpy`, `attrs`, `pyyaml`, `decorator`, `scipy`, `psutil`, `protobuf`

## 许可证

本项目采用 [Apache 2.0 License](LICENSE) 开源许可证。
