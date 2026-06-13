# Milvus 配置说明

## 当前配置

本项目通过 **环境变量** 配置 Milvus 连接。复制 `.env.example` 为 `.env` 并填写配置即可。

## 快速开始

### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，设置 MILVUS_URI
```

### 2. 选择 Milvus 部署方式

### 方案 1: Docker Milvus（推荐）

```bash
# 启动 Milvus 服务（在 docker-compose.yml 所在目录）
docker compose up -d
```

**连接地址**: `MILVUS_URI=http://localhost:19530`

**优点**:
- 跨平台支持（Windows / macOS / Linux）
- 支持完整功能（BM25、混合检索、分区等）
- 性能好，适合学习和开发

### 方案 2: Milvus Lite（本地文件模式）

```bash
pip install milvus-lite
```

**连接地址**: `MILVUS_URI=milvus_demo.db`

> ⚠️ **重要提示：Milvus Lite 不支持 Windows 系统！**
> 如果你使用 Windows，请使用方案 1（Docker）或方案 3（远程服务器）。
> 这是 Milvus 官方的限制，非本课程问题。

**优点**:
- 无需 Docker
- 适合 macOS / Linux 快速测试

**缺点**:
- **Windows 不支持**
- 功能有限（不支持 BM25 等高级特性）

### 方案 3: 远程 Milvus 服务器（课堂共享）

```bash
# 在 .env 中设置远程服务器地址
MILVUS_URI=http://your-server-ip:19530
MILVUS_DB_NAME=your_database_name
```

> ⚠️ **多用户共享注意**：如果多个学生共用同一台远程 Milvus 服务器，请注意：
> 1. 使用不同的数据库名（`MILVUS_DB_NAME`）避免数据冲突
> 2. 或者在 Collection 名称中添加个人前缀（如 `student01_document_chunks`）
> 3. 不要随意删除别人创建的 Collection
> 4. 建议优先使用本地 Docker 方案进行学习

## 验证连接

```bash
python milvus_config.py
```

输出 `[OK] Milvus 连接正常` 表示配置正确。

## 已更新的文件

以下文件已更新为使用环境变量和 `milvus_config.py` 统一配置：

- `milvus_config.py` — 统一配置中心（从环境变量读取）
- `01_milvus_basics/` — 所有 4 个文件（含 .py 和 .ipynb）
- `03_retrieval_methods/` — 所有 5 个文件
- `04_rag_api/` — 所有 2 个文件
- `05_rag_pipeline/` — 所有 3 个文件
- `rag_demo/` — 综合实战项目（使用 `rag_demo/config.py`）
