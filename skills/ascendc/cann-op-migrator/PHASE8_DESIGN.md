# Phase 8: 迁移完整性验证 - 设计说明

## 📋 概述

Phase 8 是 `cann-op-migrator` skill 的**新增核心阶段**，用于在迁移完成后验证迁移的完整性和价值。

**目标**：确保迁移后的算子在功能、性能、代码质量等方面均达到或超过原生 CANN 算子。

---

## 🎯 为什么需要 Phase 8？

### 问题背景

在没有 Phase 8 的情况下，可能存在以下风险：

1. **功能缺失未被发现**
   - 某些 dtype 可能未被迁移
   - 某些策略分支可能被遗漏
   - 边界情况处理可能不一致

2. **性能退化未被察觉**
   - 迁移后性能可能下降 30-50%
   - 内存使用可能大幅增加
   - 但没有量化对比

3. **迁移价值不明确**
   - 无法回答"为什么要迁移？"
   - 无法证明迁移带来的收益
   - 难以说服团队采用迁移方案

### Phase 8 的价值

✅ **质量保证**：系统性验证迁移完整性  
✅ **价值量化**：明确展示迁移带来的收益  
✅ **风险控制**：及时发现并修复问题  
✅ **决策支持**：为是否合并提供数据支撑  

---

## 📊 Phase 8 的四个维度

### 1. 功能性对比（Functional Comparison）

**目标**：验证迁移后的算子在功能上与原生完全等价。

**对比维度**：

#### 1.1 dtype 支持对比

**检查内容**：
- 原生代码支持的所有 dtype
- 迁移后代码支持的所有 dtype
- 两者必须完全一致

**验证方法**：
```python
# 从原生代码提取
native_dtypes = parse_tiling_keys("op_kernel/relu_apt.cpp")
# 输出: {101: "half", 102: "bfloat16", ...}

# 从迁移后代码提取
migrated_dtypes = parse_scenarios("model.py")
# 输出: {"float16", "float32", "bfloat16", ...}

# 对比
assert set(native_dtypes.values()) == set(migrated_dtypes)
```

**通过标准**：
- ✅ 所有原生 dtype 均有对应
- ✅ 无额外未定义的 dtype
- ❌ 有任何缺失 → 迁移失败

#### 1.2 策略分支覆盖对比

**检查内容**：
- 原生代码的策略分支数量
- 迁移后代码的实现状态
- 每个策略是否有测试用例

**验证方法**：
```bash
# 统计原生策略数
grep -c "TILING_KEY_IS" op_kernel/*.cpp
# 输出: 6

# 检查迁移后实现
grep -c "TILING_KEY_IS" kernel/*.cpp
# 输出: 6

# 检查测试覆盖
python -c "import json; cases = [json.loads(l) for l in open('relu.json')]; print(len(cases))"
# 输出: 6 (每个策略至少一个用例)
```

**通过标准**：
- ✅ 策略数量一致
- ✅ 每个策略有测试覆盖
- ⚠️ 有策略未实现但有合理原因 → 警告
- ❌ 策略缺失且无解释 → 失败

#### 1.3 边界情况处理对比

**检查内容**：
- 空张量行为
- NaN/Inf 处理
- -0.0 转 +0.0
- 极大/极小值

**验证方法**：
```python
test_cases = [
    {"input": torch.tensor([]), "name": "empty"},
    {"input": torch.tensor([float('nan')]), "name": "nan"},
    {"input": torch.tensor([-0.0]), "name": "neg_zero"},
]

for case in test_cases:
    native = run_native(case["input"])
    migrated = run_migrated(case["input"])
    assert torch.allclose(native, migrated, equal_nan=True)
```

**通过标准**：
- ✅ 所有边界情况行为一致
- ❌ 有任何不一致 → 失败

#### 1.4 数值精度对比

**检查内容**：
- 在所有测试用例上的数值一致性
- 最大绝对误差和相对误差
- 误差是否在可接受范围内

**验证方法**：
```python
max_abs_err = 0
max_rel_err = 0

for test_case in all_cases:
    native = run_native(test_case)
    migrated = run_migrated(test_case)
    
    abs_err = torch.abs(native - migrated)
    rel_err = abs_err / (torch.abs(native) + 1e-10)
    
    max_abs_err = max(max_abs_err, abs_err.max())
    max_rel_err = max(max_rel_err, rel_err.max())

# 判断
if dtype == torch.float32:
    assert max_abs_err < 1e-5 and max_rel_err < 1e-4
elif dtype == torch.float16:
    assert max_abs_err < 1e-3 and max_rel_err < 1e-2
```

**通过标准**：
- FP32/BF16: abs < 1e-5, rel < 1e-4
- FP16: abs < 1e-3, rel < 1e-2
- INT*: 完全一致 (error = 0)
- ❌ 超出阈值 → 失败

---

### 2. 性能对比（Performance Comparison）

**目标**：验证迁移后的算子在性能上不劣于原生。

**对比维度**：

#### 2.1 执行时间对比

**测试方法**：
```python
import time

# 预热
for _ in range(100):
    run_native(test_input)
    run_migrated(test_input)

# 测试原生
native_times = []
for _ in range(1000):
    start = time.perf_counter()
    run_native(test_input)
    native_times.append(time.perf_counter() - start)
native_avg = np.mean(native_times)

# 测试迁移后
migrated_times = []
for _ in range(1000):
    start = time.perf_counter()
    run_migrated(test_input)
    migrated_times.append(time.perf_counter() - start)
migrated_avg = np.mean(migrated_times)

speedup = native_avg / migrated_avg
```

**通过标准**：
- ✅ speedup >= 1.0 (性能不下降)
- ⚠️ 0.8 <= speedup < 1.0 (性能下降 < 20%，警告)
- ❌ speedup < 0.8 (性能下降 > 20%，失败)

#### 2.2 内存使用对比

**测试方法**：使用 NPU 性能分析工具

```bash
# 监控原生算子
msprof --device=0 --time=10s --output=native_prof python run_native.py

# 监控迁移后算子
msprof --device=0 --time=10s --output=migrated_prof python run_migrated.py

# 提取峰值内存
native_peak = extract_peak_memory("native_prof")
migrated_peak = extract_peak_memory("migrated_prof")

memory_change = (migrated_peak - native_peak) / native_peak * 100
```

**通过标准**：
- ✅ memory_change <= 0 (内存不增加)
- ⚠️ 0 < memory_change <= 10% (增加 < 10%，警告)
- ❌ memory_change > 10% (增加 > 10%，失败)

---

### 3. 代码质量对比（Code Quality Comparison）

**目标**：评估迁移后代码的可维护性和可扩展性。

**对比维度**：

#### 3.1 代码复杂度

**指标**：
- 总行数
- 最大嵌套深度
- 条件分支数量
- 函数/类数量

**分析方法**：
```python
import radon  # Python 代码复杂度分析工具

# 分析原生代码
native_metrics = radon.run("op_host/", "op_kernel/")

# 分析迁移后代码
migrated_metrics = radon.run("kernel/", "design/")

# 对比
complexity_reduction = (native_metrics.cyclomatic - migrated_metrics.cyclomatic) / native_metrics.cyclomatic
```

**通过标准**：
- ✅ 复杂度降低 >= 10%
- ⚠️ 复杂度变化在 ±10% 以内
- ❌ 复杂度增加 > 10%

#### 3.2 文档完整性

**检查清单**：
- [ ] API 文档
- [ ] 使用示例
- [ ] 性能报告
- [ ] 迁移对比报告
- [ ] 设计文档
- [ ] Trace 记录

**评分方法**：
```python
docs = ["api", "examples", "performance", "comparison", "design", "trace"]
present = sum(1 for doc in docs if exists(f"{output_dir}/{doc}.md"))
coverage = present / len(docs) * 100
```

**通过标准**：
- ✅ coverage >= 80%
- ⚠️ 60% <= coverage < 80%
- ❌ coverage < 60%

---

### 4. 迁移价值评估（Value Assessment）

**目标**：综合评估迁移是否值得。

**评估公式**：

```
迁移价值得分 = 
  功能性得分 (40%) +
  性能得分 (30%) +
  代码质量得分 (20%) +
  文档完整性得分 (10%)
```

**评分标准**：

| 维度 | 权重 | 优秀 (100) | 良好 (80) | 及格 (60) | 失败 (0) |
|-----|------|-----------|----------|----------|---------|
| 功能性 | 40% | 100% 覆盖 | 90-99% | 80-89% | < 80% |
| 性能 | 30% | 加速 >= 10% | 0-10% | -20-0% | < -20% |
| 代码质量 | 20% | 复杂度降 >= 20% | 10-20% | 0-10% | 增加 |
| 文档 | 10% | 100% | 80-99% | 60-79% | < 60% |

**最终判定**：
- ✅ **有价值** (总分 >= 80): 推荐合并
- ⚠️ **需优化** (60 <= 总分 < 80): 优化后再合并
- ❌ **无价值** (总分 < 60): 不建议合并

---

## 📝 输出产物

### 1. 迁移完整性报告

**文件**: `{output_dir}/migration_completeness_report.md`

**结构**：
```markdown
# {op_name} 迁移完整性报告

## 执行摘要
- 总体状态: ✅ PASS / ⚠️ PARTIAL / ❌ FAIL

## 功能性对比结果
1. dtype 支持对比
2. 策略分支覆盖对比
3. 边界情况处理对比
4. 数值精度对比

## 性能对比结果
1. 执行时间对比
2. 内存使用对比

## 代码质量对比
1. 代码复杂度
2. 文档完整性

## 迁移价值评估
- 优势
- 劣势
- 建议

## 结论
- 迁移完整性: ✅/⚠️/❌
- 迁移价值: ✅/⚠️/❌
- 推荐操作: 合并/优化/重做
```

### 2. 自动化验证脚本

**文件**: `scripts/validate_migration_completeness.py`

**功能**：
- 自动执行所有对比检查
- 生成结构化报告
- 返回退出码（0=PASS, 1=FAIL）

**使用方法**：
```bash
python scripts/validate_migration_completeness.py \
    <source_dir> \
    <output_dir>
```

---

## 🔧 实施建议

### 短期（立即执行）

1. **手动执行 Phase 8**
   - 对 ReLU 算子进行完整验证
   - 生成第一个迁移完整性报告
   - 验证流程的可行性

2. **创建基础验证脚本**
   - 实现 dtype 覆盖检查
   - 实现数值精度对比
   - 实现简单的性能测试

### 中期（1-2 周）

3. **完善验证脚本**
   - 添加策略分支覆盖检查
   - 添加边界情况测试
   - 添加内存使用监控
   - 添加代码复杂度分析

4. **批量验证**
   - 对 3-5 个算子执行 Phase 8
   - 收集统计数据
   - 优化验证流程

### 长期（1 个月）

5. **自动化集成**
   - 将 Phase 8 集成到 CI/CD 流程
   - 每次迁移自动执行验证
   - 自动生成报告

6. **建立基准库**
   - 收集所有算子的原生性能数据
   - 建立性能回归测试框架
   - 设置性能告警阈值

---

## 📊 预期收益

### 量化收益

| 收益类型 | 预期值 | 说明 |
|---------|--------|------|
| **问题发现率** | 95%+ | 能在合并前发现 95% 的功能缺陷 |
| **性能保障** | 100% | 确保迁移后性能不下降 |
| **决策效率** | 提升 50% | 数据驱动的合并决策 |
| **维护成本** | 降低 30% | 减少后续 bug 修复工作量 |

### 定性收益

1. **信心提升**：团队对迁移质量有信心
2. **标准化**：建立了统一的验证标准
3. **知识沉淀**：验证流程和报告成为组织资产
4. **持续改进**：基于数据不断优化迁移流程

---

## 🎯 成功标准

Phase 8 成功的标志：

1. ✅ 所有迁移的算子都通过了完整性验证
2. ✅ 没有发生因迁移导致的功能回归
3. ✅ 迁移后性能平均提升 >= 5%
4. ✅ 团队成员能够独立执行 Phase 8
5. ✅ 验证脚本覆盖率达到 90%+

---

**文档版本**: v1.0  
**创建时间**: 2026-04-24  
**作者**: AI Assistant  
**审核状态**: 待审核
