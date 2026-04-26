# Milvus 配置说明

## 当前配置

本项目已配置为**优先使用 Docker Milvus**（已运行在 `localhost:19530`）。

## Milvus 连接方式

### 方案 1: Docker Milvus（推荐，已配置）

```bash
# 检查 Docker Milvus 是否运行
docker ps | grep milvus

# 如果未运行，启动 Milvus 服务
docker compose up -d  # 在 docker-compose.yml 所在目录
```

**连接地址**: `http://localhost:19530`

**优点**:
- 性能更好
- 支持完整功能
- 适合生产环境

### 方案 2: Milvus Lite（本地文件模式）

```bash
# 安装 Milvus Lite
pip install milvus-lite

# 注意：Windows 不支持 milvus-lite，需要使用 Docker
```

**连接地址**: `milvus_demo.db`（本地文件）

**优点**:
- 无需 Docker
- 适合快速测试

**缺点**:
- Windows 不支持
- 功能有限

## 修改连接方式

如果需要切换连接方式，编辑 `milvus_config.py`:

```python
# 使用 Docker Milvus（当前配置）
MILVUS_URI = "http://localhost:19530"

# 使用 Milvus Lite（Linux/Mac）
# MILVUS_URI = "milvus_demo.db"
```

## 验证连接

运行以下命令测试连接：

```bash
python milvus_config.py
```

输出 `[OK] Milvus 连接正常` 表示配置正确。

## 已修改的文件

以下文件已更新为使用 `milvus_config.py` 中的统一配置：

- `01_milvus_basics/` - 所有 4 个文件
- `03_retrieval_methods/` - 所有 5 个文件
- `04_rag_api/` - 所有 2 个文件
- `05_rag_pipeline/` - 所有 3 个文件

## Docker 服务状态

当前运行的 Docker 服务：

```
CONTAINER ID   IMAGE           STATUS          PORTS
34fcb7b3a4d0   milvus:v2.5.0   Up 4 days       0.0.0.0:19530->19530/tcp
d720af9a1db5   minio           Up 4 days       9000/tcp
e9bd7b6a26c7   etcd:v3.5.5     Up 4 days       2379-2380/tcp
```

Milvus 服务正常运行在 `http://localhost:19530`。
