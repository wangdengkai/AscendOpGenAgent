---
name: ascendc-dev-knowledge
description: "AscendC operator development knowledge base (~3742 docs). Use when looking up AscendC API signatures, programming concepts, operator examples, error codes, or troubleshooting. Covers Kernel-side APIs, Host-side data structures, programming guides, performance optimization patterns, and fault diagnosis. Triggers on AscendC API, operator development, Tiling, InferShape, DataCopy, error codes, troubleshooting, programming guide."
---

# AscendC Knowledge Base

本知识库为 L2 层级（完整官方文档），提供 AscendC 开发的全量 API 参考和编程指南。

与 L1 层级（`evolution/knowledge_base/`，精选进化知识）的关系：
- **L1**: 19 篇精选文档，面向进化优化决策（策略选择、硬件约束、API 陷阱）
- **L2（本知识库）**: 3742 篇官方文档，面向代码编写时的精确 API 查阅
- **检索优先级**: L1 guide.md → L2 按需 grep → WebSearch

## 前置检查

使用前确认知识库数据就绪：

```bash
ls ./references/api_reference_docs/INDEX.md ./references/basic_knowledge_docs/INDEX.md \
   ./references/basic_data_api_docs/INDEX.md ./references/troubleshooting_docs/INDEX.md \
   ./references/log_reference_docs/INDEX.md 2>/dev/null | wc -l
```

结果为 **5** 即就绪。

## 知识库结构

```
./references/
├── api_reference_docs/         — Ascend C 算子开发接口
│   ├── INDEX.md
│   ├── 基础API/                # 矢量计算/矩阵计算/数据搬运/同步控制/Kernel Tiling
│   ├── 高阶API/                # Matmul/Conv3D/数学计算/归约/排序/量化
│   ├── Utils_API/              # Tiling/RTC/平台信息/C++标准库
│   ├── 基础数据结构/           # LocalTensor/GlobalTensor/Layout/Coordinate
│   └── 其他数据类型/           # TensorDesc/TPosition
│
├── basic_knowledge_docs/       — 编程指南 + 算子实践参考 + 入门教程
│   ├── INDEX.md
│   ├── 编程指南/                # 概念原理/编程模型/范式/硬件实现/编译运行/调试
│   ├── 算子实践参考/            # SIMD实现(矢量/矩阵/融合) + 性能优化 + 功能调试
│   └── 入门教程/                # 快速入门/HelloWorld/Add算子
│
├── basic_data_api_docs/        — Host 侧基础数据结构和接口
│   ├── INDEX.md
│   ├── gert命名空间/           # TilingContext/Shape/InferShapeContext/TensorV2
│   └── ge命名空间/             # AscendString/OpRegistrationData/KernelLaunchInfo
│
├── troubleshooting_docs/       — 故障处理
│   ├── INDEX.md
│   ├── 错误码参考/              # GE/RTS/HCCL/AI_CPU/FE/Driver 等模块
│   ├── 典型故障专题/            # AI Core Error/OOM/进程中断/进程卡住
│   └── 故障定位工具/            # asys 工具
│
├── log_reference_docs/         — 日志参考
│   ├── INDEX.md
│   └── FAQ/
│
└── 910D_knowledge_extra/       — 910D (Ascend 351x) 专用文档
    └── *.md                     # ~2000+ 扁平 MD 文件（API/架构/迁移指南）
```

## 检索指南

| 文档集 | 检索指南 | 用途 |
|--------|---------|------|
| `api_reference_docs/` | `references/api-reference.md` | Kernel 侧 API 签名、参数、约束、数据类型、mask、同步控制 |
| `basic_knowledge_docs/` | `references/basic-knowledge.md` | AI Core 架构、编程范式、算子示例、Tiling 策略、性能优化 |
| `basic_data_api_docs/` | `references/basic-data-api.md` | TilingContext、InferShape、Shape、TensorDesc、OpDef 注册 |
| `troubleshooting_docs/` | `references/troubleshooting.md` | 错误码 (EZ/EE/EG/...)、AI Core Error、OOM、进程卡死、asys 工具 |
| `log_reference_docs/` | `references/log-reference.md` | 日志级别配置、plog 框架、日志 FAQ |
| `910D_knowledge_extra/` | — (扁平目录，直接 grep) | 910D/351x 架构、220x→351x 迁移、RegBase 编程、SIMD VF 函数、MicroAPI |

检索指南文件包含 grep/find/cat 示例和常见查询模式。任何 section 先 `cat INDEX.md` 查看目录。

## 快速检索速查

```bash
# 找 API 文档
find "./references/api_reference_docs" -name "DataCopy*.md"
grep -rl "void Sqrt" "./references/api_reference_docs/"

# 找编程概念
grep -rl "DoubleBuffer\|双缓冲" "./references/basic_knowledge_docs/编程指南/"

# 找性能优化
find "./references/basic_knowledge_docs/算子实践参考/SIMD算子性能优化/" -name "*.md"

# 找 Host 侧接口
grep -rl "GetInputShape" "./references/basic_data_api_docs/gert命名空间/"

# 找错误码
grep -rl "EZ9999" "./references/troubleshooting_docs/错误码参考/"

# 看某个 section 的完整索引
cat "./references/api_reference_docs/INDEX.md"

# 找 910D 专用文档
grep -rl "VF函数\|MicroAPI\|RegBase" "./references/910D_knowledge_extra/"
```
