# 00_setup — 环境搭建

> 难度：⭐ | 完成本节后，你将拥有一个可运行的 RAG 开发环境

## 本节目标

- ✅ 安装 Docker 并启动 Milvus 向量数据库
- ✅ 配置 Python 虚拟环境和依赖
- ✅ 获取并配置 API Key（阿里云百炼 + DeepSeek）
- ✅ 验证环境连接正常

---

## 第一步：安装 Docker 并启动 Milvus

### Windows 用户

1. 下载安装 [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
2. 安装完成后重启电脑
3. 打开 PowerShell，验证安装：
   ```powershell
   docker --version
   ```
4. 启动 Milvus（在项目根目录执行）：
   ```bash
   docker compose up -d
   ```
5. 验证 Milvus 运行状态：
   ```bash
   docker ps | grep milvus
   ```

### macOS 用户

```bash
# 1. 安装 Docker Desktop（或使用 brew）
brew install --cask docker

# 2. 启动 Milvus
docker compose up -d

# 3. 验证
docker ps | grep milvus
```

### Linux 用户

```bash
# 1. 安装 Docker
curl -fsSL https://get.docker.com | sh

# 2. 启动 Milvus
docker compose up -d

# 3. 验证
docker ps | grep milvus
```

> ⚠️ **Windows 用户注意**：不要使用 Milvus Lite（`milvus-lite`），它不支持 Windows。Docker 是唯一推荐的本地方案。

---

## 第二步：配置 Python 环境

```bash
# 1. 确保在项目根目录
cd wolin_learn-master

# 2. 创建虚拟环境（如果还没有）
python -m venv .venv

# 3. 激活虚拟环境
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 4. 安装依赖
cd rag_examples
pip install -r requirements.txt
```

---

## 第三步：配置 API Key

### 3.1 获取阿里云百炼 API Key

1. 访问 [阿里云百炼控制台](https://bailian.console.aliyun.com/)
2. 登录后进入「API Key 管理」
3. 创建新的 API Key 并复制

### 3.2 获取 DeepSeek API Key（可选，仅 rag_demo 需要）

1. 访问 [DeepSeek 开放平台](https://platform.deepseek.com/api_keys)
2. 注册/登录后创建 API Key

### 3.3 配置 .env 文件

```bash
# 在 rag_examples/ 目录下
cp .env.example .env

# 编辑 .env 文件，填入 API Key：
# ALIYUN_API_KEY=sk-你的阿里云key
# DEEPSEEK_API_KEY=sk-你的deepseek的key（可选）
# MILVUS_URI=http://localhost:19530
```

---

## 第四步：验证环境

运行以下命令验证一切正常：

```bash
# 在 rag_examples/ 目录下
python milvus_config.py
```

期望输出：
```
[OK] Milvus 连接正常，版本：v2.x.x
```

如果看到 `[ERROR]`，请检查：
1. Docker 是否在运行（`docker ps`）
2. `.env` 文件是否正确配置

---

## 常见问题

### Q: Docker 启动后 Milvus 连接失败？
```bash
# 检查 Milvus 容器是否在运行
docker ps

# 如果没有，重新启动
docker compose up -d

# 等待 10-20 秒让服务完全启动
sleep 20
python milvus_config.py
```

### Q: `pip install` 很慢？
```bash
# 使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q: 虚拟环境激活失败（Windows PowerShell）？
```powershell
# 可能需要先执行
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Q: 我不想用 Docker，有替代方案吗？
- Windows 用户：必须使用 Docker（Milvus Lite 不支持 Windows）
- macOS/Linux 用户：可以使用 Milvus Lite（`pip install milvus-lite`），然后设置 `MILVUS_URI=milvus_demo.db`
- 所有用户：可以连接课堂提供的远程 Milvus 服务器（联系讲师获取地址）

---

## 下一步

环境搭建完成后，进入第一个模块：
- `embedding_examples/01_embedding_basics.py` — 理解 Embedding 概念
- `01_milvus_basics/01_connect_milvus.py` — 连接 Milvus
