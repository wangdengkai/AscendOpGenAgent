---
name: ttk-test
description: 对优化后的自定义算子进行TTK精度和性能测试。直接从最优变体目录复制vendors到CANN包，迁移算子文件到CANN内置目录，执行TTK测试获取baseline和优化后的精度/性能结果，并自动分析对比。适用于ops-evo优化完成后的端到端TTK验证。触发条件：用户指定"TTK测试"、"ttk test"、"跑TTK"、"TTK验证"，或在算子优化完成后需要TTK级别的精度性能验证。
---

# TTK Test Skill

对ops-evo优化后的自定义算子执行TTK（算子测试套件）精度和性能测试。

## When to use me

- 用户明确要求 "TTK测试"、"ttk test"、"跑TTK"、"TTK验证"
- ops-evo 算子优化完成，需要进行TTK级别验证
- 需要对比 baseline 与优化后算子在 TTK 框架下的精度和性能差异

## Prerequisites

- CANN 环境已配置（`ASCEND_HOME_PATH` 已设置）
- TTK 工具已部署（用户需提供 TTK 目录路径）
- 用户需提供 TTK 测试用例 CSV 文件路径
- 变体目录下存在 `evolved/vendors/{pkg_name}` 目录（ops-evo 流程的产物）

## Workflow

### Step 0: 收集参数

询问用户以下必要参数（若未提供）:

| 参数 | 说明 | 示例 |
|------|------|------|
| `TTK_DIR` | TTK 工具根目录 | `/home/user/ttk` |
| `TTK_CSV` | 测试用例 CSV 文件路径（绝对路径或相对于TTK_DIR） | `/home/user/ttk/cases.csv` |
| `OP_NAMES` | 算子名列表（逗号分隔） | `sinkhorn,sinkhorn_grad` |
| `VARIANT_DIR` | 最优变体目录（包含 evolved/vendors/） | `output/xxx_evo_xxx/round_5/parallel_3` |
| `REPO_NAME` | 目标仓名（ops_nn/ops_transformer等） | `ops_nn` |
| `SOC` | 芯片型号 | `ascend910b` 或 `ascend910_93` |
| `TTK_RUN_COUNT` | TTK 运行次数（默认5） | `5` |
| `TTK_PC` | TTK 并行核数（默认4） | `4` |
| `TTK_BUILD_MODE` | TTK 构建模式（默认release） | `release` |

**自动推导规则**:
- `PKG_NAME` 从 `{VARIANT_DIR}/evolved/vendors/` 下的子目录名自动检测（如 `custom_nn`）
- `VENDOR_DIR` = `{VARIANT_DIR}/evolved/vendors/{PKG_NAME}`
- `CUST_CONFIG_NAME` 默认为 `aic-{SOC}-ops-info-{REPO_NAME去掉ops_前缀}.json`
- `CANN_TBE_PATH` = `${ASCEND_HOME_PATH}/opp/built-in/op_impl/ai_core/tbe`
- `OP_NAMES` 可从 `VARIANT_DIR` 目录名或 world_model.json 中的 `op_name` 字段推导
- `REPO_NAME` 可通过在 CANN 内置 TBE 中搜索算子所在目录来推导（如 `find ${CANN_TBE_PATH}/impl/ -name "{op_name}" -type d`）
- `SOC` 可通过 `npu-smi info` 检测芯片型号后映射（910B3→ascend910b）

### Step 1: 环境检查

```bash
# 检查 ASCEND_HOME_PATH
echo "ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}"

# 检查 TTK 目录
ls {TTK_DIR}/run.sh

# 检查 CANN TBE 路径
ls ${ASCEND_HOME_PATH}/opp/built-in/op_impl/ai_core/tbe/

# 检查变体 vendors 目录
ls {VARIANT_DIR}/evolved/vendors/
```

如果任何检查失败，报告错误并要求用户修复。

### Step 2: 执行 baseline TTK 测试

在迁移自定义算子文件之前，先执行 baseline TTK 测试获取原始性能数据。

```bash
cd {TTK_DIR}
./run.sh {TTK_CSV} --pc={TTK_PC} -c=false -s=false -d=false -b={TTK_BUILD_MODE} --run={TTK_RUN_COUNT}
```

**TTK 结果文件命名规则**: 结果 CSV 与输入 CSV 在同一目录，文件名在原名基础上加 `_result` 后缀。
例如输入 `marndq_net_cases.csv` → 结果 `marndq_net_cases_result.csv`。

**保存 baseline 结果**: 将结果文件复制到 `{EVO_OUTPUT_DIR}/ttk_baseline/` 目录。

```bash
EVO_OUTPUT_DIR=$(dirname $(dirname {VARIANT_DIR}))  # 回到 evo 根目录
mkdir -p {EVO_OUTPUT_DIR}/ttk_baseline
# 结果文件 = 输入CSV去掉.csv + _result.csv，与输入CSV同目录
TTK_RESULT_CSV="${TTK_CSV%.csv}_result.csv"
cp "${TTK_RESULT_CSV}" {EVO_OUTPUT_DIR}/ttk_baseline/
```

### Step 3: 迁移自定义算子文件到 CANN 包

使用 `enable_ttk.py` 脚本，通过 `--vendor-dir` 参数直接从变体目录复制 vendors 到 CANN，然后迁移算子文件到 CANN 内置 TBE 目录。

**无需预先安装自定义算子包**，脚本会自动完成：
1. 将 `{VENDOR_DIR}` 复制到 `${ASCEND_HOME_PATH}/opp/vendors/{PKG_NAME}`
2. 从已复制的 vendors 中迁移各组件文件到 CANN 内置 TBE 目录

```bash
python3 .claude/skills/ttk-test/scripts/enable_ttk.py \
    --op-names {OP_NAMES} \
    --vendor-dir {VARIANT_DIR}/evolved/vendors/{PKG_NAME} \
    --repo-name {REPO_NAME} \
    --soc {SOC}
```

脚本内部执行以下操作:

**Phase 1 - 安装 vendor 包**:
- 将 `{VENDOR_DIR}` 整体复制到 `${ASCEND_HOME_PATH}/opp/vendors/{PKG_NAME}`
- 如已存在则先删除再复制

**Phase 2 - 迁移算子文件到 CANN 内置目录**（对每个算子）:
1. **kernel 实现文件**: `{PKG_NAME}_impl/ascendc/{op_name}/` → `impl/{REPO_NAME}/ascendc/`
2. **动态编译脚本**: `{PKG_NAME}_impl/dynamic/{op_name}.py` → `impl/{REPO_NAME}/dynamic/`
3. **算子注册 JSON**: `kernel/config/{SOC}/{op_name}.json` → `kernel/config/{SOC}/{REPO_NAME}/`
4. **算子二进制**: `kernel/{SOC}/{op_name}` → `kernel/{SOC}/{REPO_NAME}`
5. **binary_info_config.json**: 合并算子节点到 `kernel/config/{SOC}/{REPO_NAME}/binary_info_config.json`
6. **ops-info.json**: 合并算子节点到 `config/{SOC}/{CUST_CONFIG_NAME}`

### Step 4: 执行优化后 TTK 测试

迁移完成后，先 source 自定义算子包的环境脚本，再执行 TTK 测试获取优化后的性能数据。

**source 自定义算子包环境**: 在 TTK 运行前 source vendor 目录下的 `bin/set_env.bash`，确保 TTK 能正确加载优化后的算子。`VENDOR_DIR` 即 Step 3 中传给 `enable_ttk.py --vendor-dir` 的路径（如 `{VARIANT_DIR}/evolved/vendors/{PKG_NAME}` 或 `{VARIANT_DIR}/install/vendors/{PKG_NAME}`）。

```bash
# source 自定义算子包环境
# VENDOR_DIR = Step 3 中 --vendor-dir 的值，例如:
#   {VARIANT_DIR}/evolved/vendors/{PKG_NAME}
#   {VARIANT_DIR}/install/vendors/{PKG_NAME}
source {VENDOR_DIR}/bin/set_env.bash

cd {TTK_DIR}
./run.sh {TTK_CSV} --pc={TTK_PC} -c=false -s=false -d=false -b={TTK_BUILD_MODE} --run={TTK_RUN_COUNT}
```

**保存优化后结果**:

```bash
mkdir -p {EVO_OUTPUT_DIR}/ttk_evolved
TTK_RESULT_CSV="${TTK_CSV%.csv}_result.csv"
cp "${TTK_RESULT_CSV}" {EVO_OUTPUT_DIR}/ttk_evolved/
```

### Step 5: 分析 TTK 结果

使用 `analyze_ttk.py` 脚本对比 baseline 和优化后的 TTK 结果。

结果文件名推导: `{TTK_CSV}` 去掉 `.csv` 后加 `_result.csv`，取其 basename 用于定位保存的副本。

```bash
TTK_RESULT_BASENAME=$(basename "${TTK_CSV%.csv}_result.csv")
python3 .claude/skills/ttk-test/scripts/analyze_ttk.py \
    --baseline-csv {EVO_OUTPUT_DIR}/ttk_baseline/${TTK_RESULT_BASENAME} \
    --evolved-csv {EVO_OUTPUT_DIR}/ttk_evolved/${TTK_RESULT_BASENAME} \
    --output {EVO_OUTPUT_DIR}/ttk_comparison.json
```

分析内容:
1. **精度验证**: 检查 `precision_status` 列是否全部为 `pass`，若有 `fail` 则精度不通过
2. **性能对比**: 统计 baseline 和优化后 `bin_perf_us` 列的平均值，计算加速比

### Step 6: 输出报告

展示 TTK 测试结果摘要:

```
TTK 测试结果:
  算子:       {OP_NAMES}
  精度:       全部通过 / {N}个用例未通过
  Baseline:   {baseline_avg_us}us (平均)
  优化后:     {evolved_avg_us}us (平均)
  加速比:     {speedup}x
  结果目录:   {EVO_OUTPUT_DIR}/ttk_comparison.json
```

## Output Format

`ttk_comparison.json`:
```json
{
  "precision": {
    "all_passed": true,
    "total_cases": 100,
    "passed_cases": 100,
    "failed_cases": []
  },
  "performance": {
    "baseline_avg_us": 456.5,
    "evolved_avg_us": 233.8,
    "speedup": 1.95,
    "baseline_details": {"min": 440.0, "max": 470.0, "std": 8.2},
    "evolved_details": {"min": 220.0, "max": 250.0, "std": 7.5}
  }
}
```

## Error Guide

- `TTK run.sh not found`: 检查 TTK_DIR 路径是否正确
- `vendor-dir not found`: 检查变体目录路径是否正确，确认 evolved/vendors/ 目录存在
- `Custom TBE path not found`: vendor 安装失败，检查 ASCEND_HOME_PATH 权限
- `binary_info_config.json merge failed`: 检查源 JSON 文件是否存在，算子名是否正确
- `precision_status has failures`: 优化后算子精度不通过，需要回退检查算子实现
- `TTK CSV not found`: 检查 TTK_CSV 路径，确认测试用例文件存在
