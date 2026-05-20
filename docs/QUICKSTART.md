# Z-Search 新手入门指南

面向第一次使用 Z-Search 的用户，覆盖：环境准备 → 安装 → 两种进化模式 → 参数详解 → 常见问题。

---

## 📋 两种进化模式速览

Z-Search 提供两条进化管线，按场景选择：

| 模式 | 适用场景 | 输入 | 输出 |
|---|---|---|---|
| **ops-evo** | 已有 ops 仓里的 AscendC 算子，要进一步优化 | ops 仓 + 算子路径 | 优化后的 kernel + HTML 报告 |
| **lingxi-evo** | 从 PyTorch model 出发端到端生成 + 进化 | PyTorch model.py | 最优 AscendC 算子 + HTML 报告 |

**新手建议**：从 ops-evo 开始（如果你有现成的 ops 仓和算子）。

---

## 🔧 环境依赖

### 硬件
- **Ascend NPU**（如 Ascend 910B 系列）— 必须
- 可用显存 ≥ 16 GB（取决于算子大小）

### 软件
| 依赖 | 版本 | 何时需要 |
|---|---|---|
| Python | ≥ 3.10 | 必须 |
| CANN | ≥ 8.3.RC1 | 必须 |
| torch-npu | ≥ 2.6.0.RC1 | 必须 |
| tilelang-ascend | latest | 仅 lingxi-evo 需要，ops-evo 可跳过 |
| Claude Code CLI | latest | 必须 |

### 检查环境是否就绪

```bash
# 1. NPU 可用？
npu-smi info
# 期望: 看到 910B / 310B / 等 NPU 设备表

# 2. CANN 路径？
echo $ASCEND_HOME_PATH
# 期望: 显示路径，如 /usr/local/Ascend/cann-8.5.0
# 没有 → source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 3. torch_npu？
python3 -c "import torch_npu; print(torch_npu.__version__)"
# 期望: 2.6.x 或更高

# 4. Claude Code？
claude --version
# 期望: 显示版本号
```

任何一项失败 → 见下方"安装步骤"。

---

## 📦 安装步骤

### Step 1: 配置 CANN

```bash
# 系统已装 CANN 的情况下，只需要 source 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 验证
echo $ASCEND_HOME_PATH  # 应输出路径
```

如系统没装 CANN，需从昇腾官网下载 CANN Toolkit 安装包安装。

### Step 2: 安装 torch_npu

```bash
pip install torch-npu==2.6.0.rc1   # 或更新版本
# 验证: python3 -c "import torch_npu"
```

### Step 3: 安装 tilelang-ascend（仅 lingxi-evo 需要，ops-evo 可跳过）

```bash
git clone --recursive https://github.com/tile-ai/tilelang-ascend.git
cd tilelang-ascend
bash install_ascend.sh
source set_env.sh
cd ..
```

### Step 4: 拉取 Z-Search 仓

```bash
git clone https://gitcode.com/yyzhang0715/Z-Search.git
cd Z-Search
```

### Step 5: 安装 Python 依赖

```bash
pip install numpy attrs pyyaml decorator scipy psutil protobuf
```

### Step 6: 启动 Claude Code（⚠️ 关键步骤）

**必须从 Z-Search 仓根目录启动**，否则 `.claude/settings.json`（含 hook 注册）不会加载，长程任务防护栈失效。

```bash
cd Z-Search   # 必须在仓根目录
claude        # 启动 Claude Code
```

**启动后第一件事验证 hook 已激活**：在 Claude Code 里发一句：

```
跑 cp -r $EMPTYVAR/foo.txt /tmp/x （EMPTYVAR 不设值），如果看到 [loop-bash-safety] BLOCK: B1 提示，说明 hook 激活了。
```

如果 hook 没拦截而是命令直接报错——说明 Claude Code 没正确加载 settings.json，重启并确保在仓根目录启动。

---

## 🚀 入门示例

### 示例 1: ops-evo（推荐新手）

**场景**：你的 ops-nn 仓里已经有一个 AscendC 算子（如 `norm/add_layer_norm`），想进一步优化性能。

**在 Claude Code 中发起任务**：

```
帮我用 ops-evo 优化 ops-nn 中的算子：

- REPO_ROOT: /home/user/ops-nn
- 算子路径: norm/add_layer_norm
- 算子名: add_layer_norm_custom
- 最大轮数: 2
- 并行数: 2
- 目标加速比: 1.3
```

**预期**：约 30-60 分钟，输出在 `output/add_layer_norm_custom_ops-evo_<timestamp>/`：
- `evolution-report_*.html`：可视化报告（浏览器打开）
- `round_*/parallel_*/`：每个变体的 kernel 修改 + 评测结果
- `world_model.json`：决策树
- `state.json`：运行时状态（hook 维护）

### 示例 2: lingxi-evo（从 PyTorch model 端到端）

**场景**：你有一个 PyTorch 实现（`model.py`），想自动生成高性能 AscendC 实现。

**准备 model.py**（保存到 `/tmp/myop/model.py`）：

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        return x + alpha * y

def get_inputs():
    return [torch.randn(4096, 1024, dtype=torch.float16),
            torch.randn(4096, 1024, dtype=torch.float16),
            1.0]

def get_init_inputs():
    return []
```

**在 Claude Code 中发起任务**：

```
帮我用 lingxi-evo 进化优化 add 算子：

- 算子文件: /tmp/myop/model.py
- 算子名: MyAdd
- NPU: 0
- 最大轮数: 2
- 并行数: 3
- 目标加速比: 1.5
```

**预期**：约 1-2 小时（取决于算子复杂度），输出结构同 ops-evo。

---

## ⚙️ 参数详解

### ops-evo（进化-已有 AscendC 代码）

| 参数 | 必填？ | 默认值 | 说明 |
|---|---|---|---|
| **REPO_ROOT** | ✅ 必填 | — | ops 仓根目录（如 `/home/user/ops-nn`） |
| **算子路径** | ✅ 必填 | — | 相对路径（如 `norm/add_layer_norm`） |
| **算子名** | ✅ 必填 | — | custom_op_name（如 `add_layer_norm_custom`） |
| **目标加速比** | ✅ 必填 | — | 相对于 baseline 的提升倍数 |
| **SOC** | 可选 | 自动检测 | `ascend910b` 等 |
| **最大轮数** | 可选 | `2` | 新手用 1-2，资深用 3-5 |
| **并行数量** | 可选 | `3` | 每轮跑几个变体，越多 NPU 占用越高 |
| **停滞窗口** | 可选 | 自动 | 不动 |
| **NPU 设备 ID** | 可选 | `0` | 多卡机器选一张空闲卡 |

### lingxi-evo（进化-从 PyTorch model）

| 参数 | 必填？ | 默认值 | 推荐值 | 说明 |
|---|---|---|---|---|
| **算子文件路径** | ✅ 必填 | — | — | PyTorch model.py 路径 |
| **算子名** | ✅ 必填 | — | 短英文标识 | 如 `FastGELU` / `MyAdd` |
| **NPU 设备号** | 可选 | `0` | 用空闲卡 | 多卡机器选一张 |
| **最大轮数** | 可选 | `2` | **2-3** | 新手用 1-2，资深用 3-5 |
| **并行数量** | 可选 | `3` | **2-3** | 每轮跑几个变体，越多 NPU 占用越高 |
| **目标加速比** | 可选 | `3x` | `1.5-3x` | 设太高会无法达成，设太低进化提前停 |
| **停滞窗口** | 可选 | 自动 | 不动 | `max_rounds=2 → 1` |
| **改进轮数** | 可选 | `3` | 不动 | Local refinement 内层循环次数 |

> ⚠️ **新手最小配置**：max_rounds=1, parallel_num=2 — 跑 30-90 分钟，先体验流程

---

## 📁 输出解读

进化完成后的 `output/<op_name>_<mode>_<timestamp>/` 目录：

```
output/add_layer_norm_custom_ops-evo_20260520_141500/
├── evolution-report_add_layer_norm_custom_20260520_141500.html  ← 浏览器打开看可视化报告
├── state.json                                    ← 当前进度（自动维护）
├── world_model.json                              ← 决策树 + 评测数据
├── shared/                                       ← 共享文件
├── round_1/
│   ├── parallel_0/
│   │   ├── kernel/                               ← AscendC 内核代码
│   │   ├── evaluation_results.json               ← compile/precision/speedup
│   │   └── profiling/                            ← msprof 数据
│   └── parallel_1/
│       └── ...
└── round_2/
    └── ...
```

### 关键查看点

| 想看 | 看哪里 |
|---|---|
| 进化整体效果 | `evolution-report_*.html`（浏览器打开） |
| 最优变体的 kernel 代码 | `world_model.json` 找 `best_score` 节点 → 它的 `solution_ref` 字段 → 对应的 `round_N/parallel_K/kernel/` |
| 每个变体的精度/性能 | `round_N/parallel_K/evaluation_results.json` |
| 当前进度（任务跑到哪了） | `cat state.json` |
| 为什么进化停滞 | `world_model.json` 的 `stagnation_count` + `open_questions` 字段 |

---

## 🛡️ 长程任务防护机制（新手了解即可）

Z-Search 内置三类 Claude Code hook 防止意外：

| 何时触发 | 干什么 |
|---|---|
| 每次 Bash 命令前 | 检查路径变量空 / 根目录保护 |
| 每次 agent 想退出时 | 校验产物完整性（partial 都跑完了吗？msprof 跑了吗？refine 跑了吗？） |
| 每次子 agent 完成时 | 审计 transcript（partial 真的跑了 evaluate.sh / build_ops.py 吗？） |

**对新手意味着什么**：如果你看到 `[loop-stop] BLOCK: ...` 或 `[loop-subagent-stop] BLOCK: ...`，**不要慌**——agent 正在被框架阻止偷懒，按 stderr 提示让它补齐工作即可。

**紧急情况一键关闭**：

```bash
export LINGXI_LOOP_HOOK_DISABLE=1
# 所有 hook 降级为 warn-only（仅 stderr 提示，不阻塞）
```

详细规则见 `evolution/world_model/state_schema.md`。

---

## ❓ 常见问题

### Q1: hook 没激活？

**症状**：跑 `cp -r $EMPTYVAR/* /tmp/x` 没被拦截。

**原因**：Claude Code 不是从 Z-Search 仓根目录启动的。

**解决**：
```bash
cd /path/to/Z-Search
claude
# 检查环境变量
echo $CLAUDE_PROJECT_DIR  # 应该指向 Z-Search 仓
```

### Q2: partial 子 agent 跑得很慢（> 1 小时）？

**正常**：partial 每个变体 30-90 分钟（含 LLM 重写 + 编译 + msprof）。

**异常**：> 2 小时。可能是：
- LLM 反复 retry（精度调试 / 编译失败）— 看 `parallel_K/` 目录是否在持续更新
- NPU 资源被占用（其他进程） — `npu-smi info` 看占用

如果一直卡住，主 agent 会自动 timeout（30 分钟）并跳过这个 partial。

### Q3: 进化跑完 stage=finalize 不是 done？

**原因**：步骤 6 evolution-report 生成失败 / 没生成 HTML。

**确认**：
```bash
ls output/<op_name>_*_evo_*/evolution-report_*.html
```

如果文件不在，手动重跑：
```bash
EVO_DIR=$(ls -td output/<op_name>_*_evo_* | head -1)
python3 .claude/skills/evolution-report/scripts/generate_report.py "$EVO_DIR"
```

之后任何 `state_ops.py infer` 调用都会自动把 stage 推到 `done`。

### Q4: 限流（rate limit）退出怎么办？

**症状**：Claude 主会话报 `API Error: Request rejected (429) ... rate limit`。

**解决**：
- 等几分钟（限流是按分钟计的）
- 然后**用 ops-evo / lingxi-evo 续跑同一个 EVO_DIR**：在 Claude Code 里发"继续 EVO_DIR=output/<op_name>_<mode>_<ts> 的任务"——agent 会读 `state.json` 自动续上

### Q5: 进化半途想停？

**优雅停**：等当前 round 跑完看到 `state.stage=round_checkpoint`，然后 Ctrl+C。

**紧急停**：直接 Ctrl+C 多次，下次启动 agent 时它会读 `state.json` 决定续上还是放弃。

### Q6: 多人在同一台机器跑？

每人用**不同 NPU 卡**（NPU 设备号设不同值）。Z-Search 不会共享 NPU。

### Q7: 怎么验证基础设施 OK？

任何时候在 Z-Search 仓根目录跑：

```bash
bash tests/hooks/run_all.sh
# 期望: 8 pass, 0 fail
```

---

## 📚 进一步阅读

- 完整设计文档：[README.md](../README.md)
- 长程任务防护栈详细规则：[evolution/world_model/state_schema.md](../evolution/world_model/state_schema.md)
- Agent 内部步骤：`.claude/agents/ops-evo.md` / `lingxi-evo.md`
- Skill 详情：`.claude/skills/*/SKILL.md`

---

## 🎯 推荐学习路径

1. **第一次**：跑示例 1 (ops-evo + 一个简单算子) 最小配置（1 轮 × 2 并行）— 体验完整流程
2. **第二次**：换个稍复杂的 ops 算子，用推荐配置（2 轮 × 3 并行）— 观察进化效果
3. **第三次**：尝试 lingxi-evo 从 PyTorch model 端到端生成
4. **第四次+**：在自己的算子上跑，遇到问题查 FAQ 或 `state.json`

遇到问题反馈 → 在仓内创建 issue 时附上 `state.json` + `world_model.json` 内容。
