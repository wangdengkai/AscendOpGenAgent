FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    cmake \
    make \
    && rm -rf /var/lib/apt/lists/*

# 复制 requirements
COPY remote_server/requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建日志和任务目录
RUN mkdir -p /app/logs /tmp/ascend_tasks

# 暴露端口
EXPOSE 9002

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:9002/health || exit 1

# 启动命令
CMD ["python", "remote_server/app.py", "--host", "0.0.0.0", "--port", "9002"]
