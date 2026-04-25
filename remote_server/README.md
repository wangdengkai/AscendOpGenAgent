# AscendC Remote Evaluation Server

远程评估服务器，提供算子编译、验证和性能测试的 HTTP API。

## 快速开始

### 启动服务器

```bash
./start.sh
```

默认情况下，服务器将在 `http://0.0.0.0:8080` 上运行。

## 配置

### 方法 1: 使用 .env 文件（推荐）

1. 复制示例配置文件：
   ```bash
   cp .env.example .env
   ```

2. 编辑 `.env` 文件，修改所需配置：
   ```bash
   # 服务器端口 (默认: 8080)
   SERVER_PORT=9000
   
   # 服务器主机地址 (默认: 0.0.0.0)
   SERVER_HOST=0.0.0.0
   
   # 任务目录 (默认: /tmp/ascend_tasks)
   TASKS_DIR=/tmp/ascend_tasks
   ```

3. 启动服务器：
   ```bash
   ./start.sh
   ```

### 方法 2: 使用环境变量

在启动前设置环境变量：

```bash
export SERVER_PORT=9000
export SERVER_HOST=0.0.0.0
export TASKS_DIR=/tmp/ascend_tasks
./start.sh
```

或者在一行中设置：

```bash
SERVER_PORT=9000 SERVER_HOST=0.0.0.0 ./start.sh
```

### 方法 3: 直接修改代码（不推荐）

如果需要永久更改默认值，可以修改 `app.py` 文件中的默认端口：

```python
if __name__ == "__main__":
    port = int(os.environ.get("SERVER_PORT", "9000"))  # 修改这里的默认值
    host = os.environ.get("SERVER_HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
```

## API 端点

- `POST /api/upload_task` - 上传算子任务
- `POST /api/build` - 编译 kernel
- `POST /api/verify` - 验证精度
- `POST /api/benchmark` - 性能测试
- `POST /api/execute_command` - 执行自定义命令
- `GET /api/task_status/{task_id}` - 查询任务状态
- `GET /api/download_results/{task_id}` - 下载结果
- `WS /ws/task/{task_id}` - WebSocket 实时推送

## 常见问题

### 端口被占用

如果端口已被占用，可以通过以下方式解决：

1. **更改端口**（推荐）：
   ```bash
   SERVER_PORT=9000 ./start.sh
   ```

2. **查找并杀死占用端口的进程**：
   ```bash
   lsof -i :8080
   kill -9 <PID>
   ```

### 查看当前配置

启动服务器时，会显示当前使用的配置：

```
Starting server on 0.0.0.0:8080...
Server URL: http://0.0.0.0:8080
```
