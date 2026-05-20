# Z-Search

Z-Search 是一个面向昇腾（Ascend）NPU 的**进化式算子性能优化系统**。它通过世界模型驱动的搜索策略、多层级 Profiling 诊断和多 Agent 协同，自动搜索出高性能 AscendC 算子实现，实现从"能用"到"好用"的性能跃迁。

```
                    ┌─────────────────────────────────────────┐
                    │           Z-Search 核心理念              │
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
- **长程任务防护栈** — 三类 Claude Code hooks（PreToolUse / Stop / SubagentStop）+ 文件系统证据 + transcript 审计构成 13 条规则的多层防御，防止 agent 偷懒 / 撒谎 / 跳步骤 / 路径事故，所有判定基于客观证据，agent 无法绕过
- **双管线支持** — lingxi-evo（TileLang→AscendC 全流程）和 ops-evo（直接优化已有 AscendC 代码）

---

## 进化优化系统

### 架构概览

Z-Search 提供两条进化管线，覆盖不同的算子开发场景：

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         Z-Search Evolution Pipelines                    │
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

Z-Search 实现了两层 profiling 诊断，形成完整的性能分析闭环：

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

当进化陷入停滞时，Z-Search 会自动引入 Supervisor Agent 提供外部视角：

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

## 长程任务防护栈

进化任务跑几小时甚至几天，期间 agent 可能因 context compression、LLM 漂移、偷懒、误操作让 state 损坏或事故发生。Z-Search 通过 Claude Code 原生 hook + 客观证据校验构建三层防御：

```
┌──────────────────────────────────────────────────────────────────────┐
│              三类 Hook + 13 条规则的多层防御栈                       │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ PreToolUse(Bash)  →  loop-bash-safety.sh    [路径安全]         │ │
│  │   B1: 空变量做路径参数                                          │ │
│  │   B2: 根目录 / $HOME 保护                                       │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Stop  →  loop-stop.sh        [主 agent 边界 / 状态完整性]      │ │
│  │   R2.x: 半轮 partial 未完不能停                                 │ │
│  │   R3:  drift_status=replan_required 必须先处理                  │ │
│  │   R4:  must_run_before_next_round 未跑完不能进下一轮            │ │
│  │   R5:  session 锚定不匹配                                       │ │
│  │   R6:  少跑轮数 (actual < requested)                            │ │
│  │   R7:  少跑变体 (partials < expected_parallel_num)              │ │
│  │   R8:  跳过 refine                                              │ │
│  │   R9:  跳过 msprof profiling (自动 mark must_run)               │ │
│  │   R10: ≥50% 精度失败警告                                        │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ SubagentStop  →  loop-subagent-stop.sh   [partial 子 agent]    │ │
│  │   S1: partial transcript 必须有 bash evaluate_ascendc.sh /     │ │
│  │       python build_ops.py 真实执行（cat/grep 不算）            │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

### 设计原则：state.json 是 hook 私有账本

`<evo_dir>/state.json` 完全由 hook + wm_ops 维护，**agent prompt 不写**。每次 Stop 触发时 hook 自动重新从文件系统证据推断 state：

| 证据源 | 信任度 |
|---|---|
| `wm.session.actual_rounds_completed` (由 `wm_ops.py refine` 写入) | 最高 |
| `round_N/parallel_K/evaluation_results.json` 是否存在（脚本写） | 高 |
| `parallel_K/profiling/.../op_summary_*.csv`（msprof 写） | 高 |
| `evolution-report_*.html`（evolution-report skill 写） | 高 |
| Subagent transcript（Claude Code 写）— 子 agent 改不了自己的对话日志 | 最高 |

**LLM 写的 state 字段一律不被信任，每次 hook 触发自动覆写**——这彻底避免了"hook 校验 LLM 自报"的循环论证。

### Stage 状态机

state.json 的 `stage` 字段反映当前进化进度，由 `_infer_state_from_filesystem` 从证据自动推断：

```
init → shared_prep → wm_init
                       │
                       ↓
  ┌──── round_select ←─┴──┐
  │        ↓              │
  │   round_generate      │
  │   (partial_status)    │
  │        ↓              │
  │   round_refine        │
  │        ↓              │
  │   round_react         │
  │        ↓              │
  └── round_checkpoint    │
            │             │
    (max_rounds 达到)     │
            ↓             │
        finalize ←────────┘
            │
    (evolution-report*.html 出现)
            ↓
          done
```

### 兜底机制

- **`LINGXI_LOOP_HOOK_DISABLE=1`** — 30 秒生效，全局降级为 warn-only（仅 stderr 提示不阻塞）
- **删除 `.claude/settings.json` 中具体 hook 注册** — 单点禁用
- **`git revert`** — 完全回退到无 hook 状态

### 测试覆盖

`tests/hooks/run_all.sh` 一键回归 8 个测试套件 + 50+ 断言：

```
test_bash_safety       — B1/B2 路径安全 (7 场景)
test_state_ops         — state.json CLI (7 场景)
test_state_infer       — _infer_state_from_filesystem 决策表 (11 场景)
test_stop_hook         — R2-R5 + done 终态 + HOOK_DISABLE (9 场景)
test_wm_ops_state_sync — wm_ops ↔ state.json 集成
test_drift_breaker     — R9 drift_status 自动触发 (7 场景)
test_anti_skip         — R6/R7/R8/R9/R10 防偷懒 (8 场景)
test_subagent_stop     — S1 partial transcript 审计 (9 场景)
```

详细 schema + 规则定义见 `evolution/world_model/state_schema.md`。

---

## 单次生成（lingxi Agent）

除了进化优化，Z-Search 也支持单次算子生成，适用于快速原型开发：

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

> 👉 **新手强烈建议先看 [docs/QUICKSTART.md](docs/QUICKSTART.md)** — 含完整环境检查、安装步骤、3 种模式入门示例、参数详解、FAQ。

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
├── world_model.json              # 世界模型决策树（决策证据）
├── state.json                    # 运行时状态游标（hook 私有账本）
├── solution_db.jsonl             # 所有变体的血统追踪
├── shared/                       # 共享文件（所有变体复用）
│   ├── model.py
│   ├── <op_name>.json
│   └── design/
├── round_1/
│   ├── parallel_0/               # 变体 0（含 kernel/, evaluation_results.json, profiling/）
│   ├── parallel_1/               # 变体 1
│   └── ...
├── round_2/
│   └── ...
├── evolution-report_*.html       # 步骤 6 产出的 HTML 报告（stage → done 的客观证据）
└── profiling/                    # Profiling 数据
```

---

## 项目结构

```
Z-Search/
├── .claude/
│   ├── agents/                   # Agent 定义
│   │   ├── lingxi.md             # 单次生成 Agent（TileLang→AscendC）
│   │   ├── lingxi-evo.md         # TileLang 进化 Agent
│   │   ├── lingxi-partial.md     # TileLang 进化子 Agent
│   │   ├── ops-evo.md            # Ops 进化 Agent
│   │   └── ops-partial.md        # Ops 进化子 Agent
│   ├── hooks/                    # Claude Code hook 脚本（长程任务防护栈）
│   │   ├── loop-bash-safety.sh   # PreToolUse(Bash) — B1/B2 路径安全
│   │   ├── loop-stop.sh          # Stop — R2-R10 主 agent 边界
│   │   ├── loop-subagent-stop.sh # SubagentStop — S1 partial transcript 审计
│   │   └── lib/                  # 共享 helpers (common.sh / paths.sh / state.sh)
│   ├── settings.json             # Hook 注册（clone 即生效）
│   └── skills/                   # Skill 模块
│       ├── tilelang-designer/    # TileLang 设计（block/tile level）
│       ├── ascendc-translator/   # AscendC 转译（含 API 知识库）
│       ├── case-simplifier/      # 测试用例精简
│       ├── performance-analyzer/ # 性能分析
│       ├── trace-recorder/       # Trace 记录
│       ├── ascendc-profiling/    # CSV 级 Profiling 诊断
│       ├── ascendc-profiling-analysis/ # 指令级深度空泡分析
│       ├── hardware-specs-query/ # 硬件规格查询
│       ├── evolution-report/     # 进化 HTML 报告生成
│       └── ...
├── .claude-plugin/               # Plugin 元数据占位（未来发布用）
├── evolution/
│   ├── world_model/              # 世界模型 + 运行时状态
│   │   ├── schema.md             # world_model.json Schema 定义
│   │   ├── operations.md         # 世界模型操作协议
│   │   ├── wm_ops.py             # 世界模型操作脚本（refine/select/diagnose）
│   │   ├── state_ops.py          # state.json CLI（init/read/infer/validate）
│   │   ├── state_schema.md       # state.json schema + hook 规则文档
│   │   ├── transcript_audit.py   # Subagent transcript 审计（S1）
│   │   ├── session_anchor.py     # Session 身份锚定
│   │   ├── check_round_artifacts.py  # 产物完整性检查
│   │   └── solution_db.py        # 变体血统追踪
│   ├── meta_prompts/
│   │   ├── strategies/           # 59 条优化策略
│   │   ├── strategy-index.md     # 策略索引
│   │   ├── lingxi-partial-prompt.md  # 子 Agent prompt 模板
│   │   └── supervisor-prompt.md  # Supervisor prompt 模板
│   └── knowledge_base/           # 领域知识库
│       ├── algorithm_insights/
│       ├── optimization_patterns/
│       ├── proven_solutions/
│       └── hardware/
├── tests/
│   └── hooks/                    # Hook 测试套件（8 个套件 + 50+ 断言）
│       └── run_all.sh            # 一键回归
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
