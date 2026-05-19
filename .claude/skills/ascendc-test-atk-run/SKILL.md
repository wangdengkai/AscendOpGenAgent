---
name: ascendc-test-atk-run
description: 使用 ATK（API Toolkit）工具对 CustomOP 算子进行精度测试。覆盖：安装 ATK 工具、查找算子对应的 ATK 测试命令、切换到正确的算子目录、执行 atk task 命令并解析结果。触发关键词：跑ATK、ATK测试、atk task、atk算子精度测试、atk算子性能测试、运行ATK用例。
---

## 我能做什么

对 `CustomOP-master` 中的算子执行 ATK（API Toolkit）精度测试：

1. **检查 ATK 工具是否已安装**，未安装则自动安装
2. **从 Readme.md 中查找指定算子的测试命令**
3. **切换到对应算子的 ATK 子目录**，执行 `atk task` 命令
4. **解析并汇报测试结果**（通过 / 失败 / 超时）

## 基础路径说明

> [注意] **重要提示**：`CustomOP-master` 必须放在与 `Z-Search` **同级**的目录下，即 `Z-Search` 的父目录中。如果目录位置不对，所有路径将无法找到。

`CustomOP-master` 与 `Z-Search`（Claude Code 工作目录）**同级**，因此在 skill 中统一使用相对路径 `../CustomOP-master/`（相对于 Claude Code 工作目录 `Z-Search/`）：

| 相对路径（基于工作目录 Z-Search/） | skill 文件相对路径（基于 .claude/skills/ascendc-test-atk-run/） | 说明 |
|------|------|------|
| `../CustomOP-master/atk/` | `../../../../CustomOP-master/atk/` | ATK 测试根目录，每个算子一个子目录 |
| `../CustomOP-master/atk/Readme.md` | `../../../../CustomOP-master/atk/Readme.md` | 所有算子的 `atk task` 命令汇总 |
| `../CustomOP-master/atk/ATK-0.12.17-py3-none-any.whl` | `../../../../CustomOP-master/atk/ATK-0.12.17-py3-none-any.whl` | ATK 安装包 |
| `$(which atk)` | `$(which atk)` | ATK 可执行文件（安装后自动在 PATH 中） |

## 工作流程

### 第一步：检查 ATK 是否已安装

先验证 `CustomOP-master` 目录是否在正确位置：

```bash
# 检查 CustomOP-master 是否存在（与 Z-Search 同级）
[ -d "../CustomOP-master/atk" ] && echo "[OK] CustomOP-master/atk 目录存在" || echo "[FAIL] 缺失：请将 CustomOP-master 放到 Z-Search 的同级目录"
[ -f "../CustomOP-master/atk/ATK-0.12.17-py3-none-any.whl" ] && echo "[OK] ATK whl 存在" || echo "[FAIL] 缺失：ATK-0.12.17-py3-none-any.whl"
[ -f "../CustomOP-master/atk/Readme.md" ] && echo "[OK] Readme.md 存在" || echo "[FAIL] 缺失：Readme.md"
```

**如果检查失败**，提示用户确认目录结构：
```
期望结构：
  Z-Search/          ← Claude Code 工作目录
  CustomOP-master/   ← 与 Z-Search 同级
    └── atk/
        ├── ATK-0.12.17-py3-none-any.whl
        ├── Readme.md
        ├── AiInfraCausalConv1dFnAdd/
        └── AiInfraCausalConv1dUpdateAdd/
```

**检查通过后**，验证 ATK 是否已安装：

```bash
# 方式1：检查 atk 命令
which atk 2>/dev/null && atk --version 2>/dev/null || echo "未找到 atk"

# 方式2：通过 pip 检查
pip show atk 2>/dev/null | grep Version || echo "ATK 未安装"
```

**如果未安装**，执行以下命令安装：

```bash
pip install ../CustomOP-master/atk/ATK-0.12.17-py3-none-any.whl
```

安装完成后，验证：

```bash
pip show atk
```

---

### 第二步：查找算子对应的测试命令

首先提示用户是否将算子ATK脚本（node.yaml、可执行文件 executor.py、json 用例文件）放在正确位置，也就是 `../CustomOP-master/atk/{算子子目录}/`

检查目标算子目录是否存在：

```bash
# 以 AiInfraCausalConv1dFnAdd 为例
[ -d "../CustomOP-master/atk/AiInfraCausalConv1dFnAdd" ] && echo "[OK] 目录存在" || echo "[FAIL] 缺失：AiInfraCausalConv1dFnAdd/"
```

json 用例文件名带有 `network` 关键字的是网络用例，json用例文件名字不带 `network` 关键字是泛化用例

从 Readme.md 中定位指定算子的所有 `atk task` 命令。

```bash
# 读取完整 Readme.md，找到目标算子的命令
cat ../CustomOP-master/atk/Readme.md
```

**各算子与 ATK 子目录的对应关系：**

| 算子名称 | ATK 子目录 | Readme.md 中的关键词 |
|----------|-----------|---------------------|
| AiInfraCausalConv1dFnAdd | `AiInfraCausalConv1dFnAdd/` | `ai_infra_causal_conv1d_fn_add` |
| AiInfraCausalConv1dUpdateAdd | `AiInfraCausalConv1dUpdateAdd/` | `ai_infra_causal_conv1d_update_add` |
---

### 第三步：切换到算子 ATK 目录并执行测试

**[注意] 关键要求**：`atk task` 必须在对应算子的 ATK 子目录下执行，因为命令中的 `-n`（yaml 文件）、`-c`（json 文件）、`-p`（executor py 文件）均为相对路径。

```bash
# 切换到目标算子目录
cd ../CustomOP-master/atk/{算子子目录}/

# 执行 Readme.md 中对应的 atk task 命令，例如 AiInfraCausalConv1dFnAdd 泛化用例：
atk task -n node_accuracy_npu.yaml -c ai_infra_causal_conv1d_fn_add.json -p executor.py -to 3000

# 执行 Readme.md 中对应的 atk task 命令，例如 AiInfraCausalConv1dFnAdd 网络用例:
atk task -n node_accuracy_npu.yaml -c ai_infra_causal_conv1d_fn_add_network.json -p executor.py -to 3000
```

**`atk task` 常用参数说明：**

| 参数 | 说明 | 示例 |
|------|------|------|
| `-n` | 节点配置 yaml 文件（指定后端和设备） | `node_accuracy_npu.yaml` |
| `-c` | 测试用例 json 文件 | `ai_infra_causal_conv1d_fn_add.json` |
| `-p` | executor python 文件（算子调用逻辑） | `executor.py` |
| `-mt` | 最大并发线程数 | `100`（默认）|
| `-to` | 超时时间（秒） | `650`（默认）|
| `-s` | 起始用例 ID（可选） | `0` |
| `-e` | 结束用例 ID（可选， 不包含此id） | `52` |
| `-wl` | 白名单用例范围（可选） | `[1,2,[4,6]]`，`[1]`代表`[1]`，`[[1,5]]`代表`[1,2,3,4]`（左闭右开）,如`[1,2,[4,6]]`代表`[1,2,4,5]` |
| `--input_data` | 输入数据路径（部分算子需要） | `/path/to/input/` |
| `--bm_output_path` | benchmark 输出路径（部分算子需要） | `/path/to/output/` |

**node yaml 文件说明：**

两个算子的 `node_accuracy_npu.yaml` 格式略有不同，以实际文件为准：

```yaml
# AiInfraCausalConv1dFnAdd/node_accuracy_npu.yaml
nodes:
  - backend: npu
    devices: [0,1,2,3]       # vector 核使用的 NPU 设备编号
    task: ['accuracy']       # accuracy：精度测试，如果需要测试性能，将`task`字段设置为['performance_device']
    name: vector
  - backend: npu
    devices: [4,5,6]       # cube 核使用的 NPU 设备编号
    task: ['accuracy']
    name: cube
```
> [提示] 如需修改使用的 NPU 设备编号，直接编辑对应算子目录下的 `node_accuracy_npu.yaml` 中的 `devices` 字段；如果需要测试性能，将`task`字段设置为['performance_device'] 。

---

### 第四步：解析测试结果

ATK 执行完毕后，观察控制台输出，重点关注：

**成功标志：**
```
PASSED  [case_id=xxx] ...
All cases PASSED
```

**失败标志：**
```
FAILED  [case_id=xxx] max_re_ratio exceeded ...
精度超标：max_re_ratio / avg_re_ratio / root_mean_squared_ratio 超出阈值
```

**结果汇报格式：**

```
算子：{算子名称}
测试命令：atk task -n ... -c ... -p ...
总用例数：N
通过：X
失败：Y
失败用例：[case_id_1, case_id_2, ...]（如有）
结论：精度测试通过 / 存在精度问题，需要排查
```

---

## 完整示例

### 示例：测试 AiInfraCausalConv1dFnAdd 算子

```bash
# 1. 检查 ATK 安装
which atk || pip install ../CustomOP-master/atk/ATK-0.12.17-py3-none-any.whl

# 2. 切换到算子 ATK 目录
cd ../CustomOP-master/atk/AiInfraCausalConv1dFnAdd/

# 3a. 限制最大并发任务数场景，最大并发任务数为1，超时时间3000秒
atk task -n node_accuracy_npu.yaml -c ai_infra_causal_conv1d_fn_add.json -p executor.py -mt 1 -to 3000

# 3b. 带白名单, 超时时间默认
atk task -n node_accuracy_npu.yaml -c ai_infra_causal_conv1d_fn_add.json -p executor.py -to 3000 -wl [[0,21],[57,70]]

# 3c. 通过 -s -e 执行指定的case, 执行id=0,1,2,3,4的case
atk task -n node_accuracy_npu.yaml -c ai_infra_causal_conv1d_fn_add.json -p executor.py -s 0 -e 5
```


## 常见问题排查

| 问题 | 原因 | 解决方法 |
|------|------|---------|
| `atk: command not found` | ATK 未安装或不在 PATH | 安装 ATK whl 或用 `find / -name atk -type f 2>/dev/null` 找到完整路径 |
| `FileNotFoundError: node_accuracy_npu.yaml` | 未切换到算子 ATK 子目录 | `cd ../CustomOP-master/atk/{算子目录}/` 后再执行 |
| `FileNotFoundError: executor_xxx.py` | json/py 文件名与目录不匹配 | `ls` 查看当前目录文件名，与命令中的 `-p` 参数核对 |
| 超时（`TimeoutError`） | `-to` 参数设置太小或用例数量多 | 增大 `-to` 值，或用 `-s`/`-e` 缩小测试范围 |
| 精度超标（`max_re_ratio exceeded`） | 算子实现存在精度问题 | 调用 `ascendc-dumptensor` skill 进行 DumpTensor 定位 |
| `input_data 路径不存在` | 外部数据未准备 | 确认 `--input_data` 路径是否正确，联系数据提供方 |
| NPU 设备占用 | 其他进程占用了 yaml 中指定的设备 | 修改 `node_accuracy_npu.yaml` 中 `devices` 字段，换用空闲设备 |

---

## 目录文件速查

```
Z-Search/                        ← Claude Code 工作目录（pwd）
CustomOP-master/                 ← 与 Z-Search 同级，相对路径 ../CustomOP-master/
└── atk/
    ├── ATK-0.12.17-py3-none-any.whl                    # ATK 安装包
    ├── Readme.md                                        # 所有算子的 atk task 命令汇总
    ├── AiInfraCausalConv1dFnAdd/                        # FnAdd 算子目录
    │   ├── node_accuracy_npu.yaml                           # 节点配置（指定设备）
    │   ├── ai_infra_causal_conv1d_fn_add.json               # 泛化用例
    │   ├── ai_infra_causal_conv1d_fn_add_network.json       # 网络用例
    │   └── executor.py                                      # 执行文件
    └── AiInfraCausalConv1dUpdateAdd/                    # UpdateAdd 算子目录
        ├── node_accuracy_npu.yaml                           # 节点配置（指定设备）
        ├── ai_infra_causal_conv1d_update_add.json           # 泛化用例
        ├── ai_infra_causal_conv1d_update_add_network.json   # 网络用例
        └── executor.py                                      # 执行文件
```