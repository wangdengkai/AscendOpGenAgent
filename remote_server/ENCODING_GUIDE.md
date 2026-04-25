# 文件编码规范

## 问题说明

在跨平台文件操作中，如果不显式指定编码，会导致以下问题：

### Windows vs Linux 默认编码差异

| 平台 | 默认编码 | 问题 |
|------|---------|------|
| Windows | GBK / CP936 | 中文注释乱码 |
| Linux/macOS | UTF-8 | 正常显示 |

### 示例问题

```python
# ❌ 错误写法（使用系统默认编码）
path.write_text(content)  # Windows: GBK, Linux: UTF-8

# ✅ 正确写法（强制 UTF-8）
path.write_text(content, encoding='utf-8')
```

## 修复内容

已修复所有文件读写操作，统一使用 **UTF-8** 编码：

### 1. 文件写入

```python
# model.py
(task_dir / "model.py").write_text(data.model_py, encoding='utf-8')

# kernel files
(kernel_dir / filename).write_text(content, encoding='utf-8')

# config.json
config_path.write_text(
    json.dumps(config, indent=2, ensure_ascii=False), 
    encoding='utf-8'
)

# custom scripts
script_path.write_text(content, encoding='utf-8')
```

### 2. 文件读取

```python
# config.json
config = json.loads(config_file.read_text(encoding='utf-8'))

# created_at
created_at = float(created_at_file.read_text(encoding='utf-8'))
```

### 3. JSON 特殊处理

```python
# ensure_ascii=False 保证中文字符不被转义
json.dumps(config, indent=2, ensure_ascii=False)
```

**对比:**

```python
# ❌ ensure_ascii=True (默认)
{"task_name": "\u6d4b\u8bd5"}  # 中文被转义

# ✅ ensure_ascii=False
{"task_name": "测试"}  # 中文正常显示
```

## 最佳实践

### 规则 1: 所有文本文件使用 UTF-8

```python
# 写入
Path("file.txt").write_text(text, encoding='utf-8')

# 读取
text = Path("file.txt").read_text(encoding='utf-8')
```

### 规则 2: JSON 文件启用 ensure_ascii=False

```python
import json

data = {"name": "测试", "value": 123}

# 写入
Path("data.json").write_text(
    json.dumps(data, indent=2, ensure_ascii=False),
    encoding='utf-8'
)

# 读取
data = json.loads(Path("data.json").read_text(encoding='utf-8'))
```

### 规则 3:  subprocess 输出解码

```python
# 命令输出可能包含非 UTF-8 字符，使用 errors='replace'
stdout.decode('utf-8', errors='replace')
stderr.decode('utf-8', errors='replace')
```

## 检查清单

在添加新的文件操作时，确保：

- [ ] `write_text()` 包含 `encoding='utf-8'`
- [ ] `read_text()` 包含 `encoding='utf-8'`
- [ ] JSON 序列化使用 `ensure_ascii=False`
- [ ] subprocess 输出使用 `errors='replace'`

## 测试验证

### 测试用例 1: 中文注释

```python
# model.py 包含中文注释
model_py = """
import torch

class Model(torch.nn.Module):
    def forward(self, x):
        # 这是一个测试算子
        return torch.nn.functional.elu(x)
"""

# 上传后检查文件内容
content = Path("tasks/xxx/model.py").read_text(encoding='utf-8')
assert "这是一个测试算子" in content  # 应该能正确读取
```

### 测试用例 2: 中文任务名

```python
# 配置中包含中文
config = {
    "task_name": "ELU激活函数",
    "description": "测试中文支持"
}

# 保存并读取
Path("config.json").write_text(
    json.dumps(config, indent=2, ensure_ascii=False),
    encoding='utf-8'
)

loaded = json.loads(Path("config.json").read_text(encoding='utf-8'))
assert loaded["task_name"] == "ELU激活函数"
```

## 相关文档

- [Python pathlib 文档](https://docs.python.org/3/library/pathlib.html#pathlib.Path.write_text)
- [JSON 编码说明](https://docs.python.org/3/library/json.html#basic-usage)
- [UTF-8  Everywhere](https://utf8everywhere.org/)

---

**最后更新**: 2024-04-24  
**状态**: ✅ 已修复所有文件操作的编码问题
