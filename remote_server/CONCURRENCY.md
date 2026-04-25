# 并发安全性说明

本文档详细说明远程服务器的并发安全性和多用户支持能力。

## 📊 并发架构

### 当前设计

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│  User A  │────►│          │     │          │
└──────────┘     │          │     │          │
                 │ FastAPI  │────►│ NPU 0-7  │
┌──────────┐     │ (Async)  │     │          │
│  User B  │────►│          │     │          │
└──────────┘     │          │     └──────────┘
                 │          │
┌──────────┐     │          │     ┌──────────┐
│  User C  │────►│          │     │ Task Dir │
└──────────┘     └──────────┘     │ (UUID)   │
                                  └──────────┘
```

## ✅ 已实现的并发安全机制

### 1. 任务隔离（Task Isolation）

**机制**: 每个任务使用唯一的 UUID 作为目录名

```python
task_id = str(uuid.uuid4())  # 例如: "550e8400-e29b-41d4-a716-446655440000"
task_dir = TASKS_DIR / task_id
```

**安全性**: ✅ 完全隔离
- 不同用户的任务存储在不同目录
- 文件不会互相覆盖
- 支持无限并发任务（受限于磁盘空间）

**示例**:
```
/tmp/ascend_tasks/
├── 550e8400-e29b-41d4-a716-446655440000/  ← User A
├── 6ba7b810-9dad-11d1-80b4-00c04fd430c8/  ← User B
└── 7c9e6679-7425-40de-944b-e07fc1f90ae7/  ← User C
```

---

### 2. NPU 设备调度（NPU Scheduling）

**机制**: 智能负载均衡算法自动分配 NPU 设备

```python
class NPUScheduler:
    def allocate_npu(self, preferred_npu: int = None) -> int:
        # 1. 优先使用用户指定的 NPU（如果负载低）
        # 2. 否则选择负载最低的 NPU
        # 3. 线程安全（使用锁保护）
```

**安全性**: ✅ 避免设备竞争
- 自动检测 NPU 数量
- 跟踪每个 NPU 的活跃任务数
- 负载均衡分配
- 线程安全的状态管理

**示例场景**:
```
时刻 T1:
  User A → NPU 0 (load: 1)
  User B → NPU 1 (load: 1)
  
时刻 T2:
  User C → NPU 2 (load: 1)  # 自动选择空闲设备
  
时刻 T3:
  User D → NPU 0 (load: 2)  # NPU 0 负载仍较低，可复用
  User E → NPU 3 (load: 1)  # 选择新的空闲设备
```

**查询 NPU 状态**:
```bash
curl http://localhost:8080/api/npu_status
```

返回:
```json
{
  "num_npus": 8,
  "npu_load": {
    "0": 2,
    "1": 1,
    "2": 1,
    "3": 1,
    "4": 0,
    "5": 0,
    "6": 0,
    "7": 0
  },
  "total_tasks": 5
}
```

---

### 3. 进程隔离（Process Isolation）

**机制**: 每个命令在独立进程组中运行

```python
process = await asyncio.create_subprocess_exec(
    *cmd,
    preexec_fn=os.setsid  # 创建新进程组
)

# 超时后杀死整个进程组
os.killpg(os.getpgid(process.pid), signal.SIGKILL)
```

**安全性**: ✅ 完全隔离
- 每个编译/验证任务独立进程
- 超时后彻底清理
- 不会遗留僵尸进程

---

### 4. 异步处理（Async Processing）

**机制**: FastAPI 原生异步支持

```python
@app.post("/api/upload_task")
async def upload_task(data: TaskUpload):
    # 非阻塞处理
    
@app.post("/api/build")
async def build_kernel(request: BuildRequest):
    # 可以并发执行
```

**安全性**: ✅ 高并发支持
- 多个请求同时处理
- 不会互相阻塞
- 充分利用 CPU/NPU 资源

---

## ⚠️ 潜在风险及缓解

### 风险 1: 磁盘空间耗尽

**场景**: 大量用户上传大文件

**影响**: 
- 新任务无法上传
- 现有任务可能失败

**缓解措施**:
```python
# 1. 文件大小限制
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB/文件
MAX_TOTAL_SIZE = 200 * 1024 * 1024  # 200MB/任务

# 2. 自动清理过期任务
TASK_TTL = 3600  # 1小时后自动删除

# 3. 监控磁盘使用
def check_disk_space():
    usage = shutil.disk_usage(TASKS_DIR)
    if usage.percent > 90:
        cleanup_old_tasks()
```

**建议**:
- 定期监控磁盘使用
- 调整 `TASK_TTL` 参数
- 考虑使用独立的存储卷

---

### 风险 2: NPU 内存不足

**场景**: 多个大模型同时在同一 NPU 上运行

**影响**:
- OOM (Out Of Memory) 错误
- 任务失败

**缓解措施**:
```python
# NPU 调度器限制单设备并发数
if self.npu_load[preferred_npu] < 2:  # 最多2个任务
    return preferred_npu
```

**建议**:
- 根据 NPU 内存调整负载阈值
- 监控 NPU 内存使用
- 大任务独占 NPU

---

### 风险 3: API 速率限制缺失

**场景**: 单个用户频繁调用 API

**影响**:
- 服务器负载过高
- 其他用户受影响

**缓解措施** (待实现):
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/upload_task")
@limiter.limit("10/minute")  # 每分钟最多10次
async def upload_task(request, data: TaskUpload):
    ...
```

**建议**:
- 添加速率限制
- 实现用户认证
- 设置配额管理

---

### 风险 4: 配置文件并发读写

**场景**: 多个任务同时读取 config.json

**影响**: 
- 理论上可能有竞态条件
- 实际风险很低（只读操作）

**当前状态**: ✅ 安全
- config.json 只在创建时写入
- 后续都是只读访问
- JSON 解析是原子操作

---

## 📈 并发性能测试

### 测试场景

```bash
# 模拟 10 个用户同时上传
for i in {1..10}; do
    curl -X POST "http://localhost:8080/api/upload_task" \
      -H "Content-Type: application/json" \
      -d @task_$i.json &
done

wait
```

### 预期结果

| 指标 | 预期值 |
|------|--------|
| 最大并发任务数 | 取决于 NPU 数量 × 2 |
| 平均响应时间 | < 1s (上传), < 5min (完整流程) |
| 任务成功率 | > 99% |
| NPU 利用率 | 均衡分布 |

---

## 🔧 配置优化建议

### 1. 调整 NPU 数量

```bash
# 方法 1: 环境变量
export NUM_NPUS=8

# 方法 2: 代码中指定
npu_scheduler = NPUScheduler(num_npus=8)
```

### 2. 调整负载阈值

```python
# 在 NPUScheduler.allocate_npu() 中
if self.npu_load[preferred_npu] < 2:  # 改为 1 或 3
```

- **保守模式**: 阈值 = 1（每 NPU 最多 1 任务）
- **平衡模式**: 阈值 = 2（默认）
- **激进模式**: 阈值 = 3+（高密度部署）

### 3. 调整任务 TTL

```python
TASK_TTL = 3600  # 1小时
# 或
TASK_TTL = 7200  # 2小时（保留更久）
```

---

## 🛡️ 生产环境建议

### 必须实施

1. **监控告警**
   ```python
   # 定期检查
   - NPU 负载 > 80%
   - 磁盘使用 > 90%
   - 任务失败率 > 5%
   ```

2. **日志记录**
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   logger.info(f"Task {task_id} allocated to NPU {npu_id}")
   ```

3. **备份策略**
   ```bash
   # 定期备份重要任务
   tar czf backup_$(date +%Y%m%d).tar.gz /tmp/ascend_tasks/
   ```

### 推荐实施

4. **用户认证**
   ```python
   from fastapi import Depends, HTTPException
   
   def verify_api_key(api_key: str = Header(...)):
       if api_key not in VALID_KEYS:
           raise HTTPException(status_code=401)
   ```

5. **速率限制**
   ```python
   @limiter.limit("10/minute")
   ```

6. **健康检查**
   ```python
   @app.get("/health")
   def health_check():
       return {
           "status": "healthy",
           "npu_status": npu_scheduler.get_status()
       }
   ```

---

## 📋 并发安全检查清单

部署前确认：

- [ ] NPU 调度器正常工作
- [ ] 任务目录正确隔离
- [ ] 文件大小限制生效
- [ ] 自动清理功能启用
- [ ] 磁盘空间充足
- [ ] 日志记录完整
- [ ] 监控告警配置
- [ ] 备份策略就绪

---

## 🎯 总结

### 并发安全性评级

| 方面 | 评级 | 说明 |
|------|------|------|
| **任务隔离** | ⭐⭐⭐⭐⭐ | UUID 完全隔离 |
| **NPU 调度** | ⭐⭐⭐⭐⭐ | 智能负载均衡 |
| **进程管理** | ⭐⭐⭐⭐⭐ | 独立进程组 |
| **异步处理** | ⭐⭐⭐⭐⭐ | FastAPI 原生支持 |
| **磁盘管理** | ⭐⭐⭐⭐ | 需监控空间 |
| **速率限制** | ⭐⭐⭐ | 待实现 |

### 结论

✅ **当前设计支持多用户并发使用**

- 任务完全隔离，不会冲突
- NPU 自动调度，负载均衡
- 异步处理，高并发支持
- 需要注意磁盘空间和速率限制

**建议**: 生产环境添加监控、认证和速率限制。

---

**最后更新**: 2024-04-24  
**版本**: 1.0  
**状态**: ✅ 并发安全设计已完成
