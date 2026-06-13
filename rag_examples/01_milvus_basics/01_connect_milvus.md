# 连接 Milvus

## 核心概念

### 什么是 Milvus？

**定义**
- Milvus = 向量数据库
- 专门存储和搜索向量数据的数据库

### 生活化比喻：Milvus = "超级图书馆"

| 传统数据库 | 向量数据库 |
|:---:|:---:|
| 按书名/作者查找（精确匹配） | 按"内容相似"查找（语义匹配） |

**例子**：你找"苹果"相关的书
- 传统搜索：找书名包含"苹果"的书
- 向量搜索：找内容涉及"水果"、"iPhone"、"科技"的书

### Milvus 架构简介

```
┌─────────────┐
│   Client    │  ← Pymilvus（Python 客户端）
├─────────────┤
│   Server    │  ← Milvus 服务
├─────────────┤
│   Storage   │  ← 向量存储 + 标量存储
└─────────────┘
```

### 数据类型
- **向量数据**：Embedding 生成的数字向量（如 `[0.1, -0.5, 0.8, ...]`）
- **标量数据**：ID、文本、数字等传统字段

---

## 运行前检查

1. 已安装依赖：`pip install pymilvus`
2. Milvus 服务（二选一）：
   - **方案 A**: Milvus Lite（`pip install milvus-lite`）- 本地文件模式
   - **方案 B**: Docker Milvus（推荐）- `docker run milvusdb/milvus:v2.5.0`


```python
# 导入必要的库
from pymilvus import MilvusClient
```

---

## 示例 1: 连接本地 Milvus（Milvus Lite）

使用 Milvus Lite（轻量级版本，适合学习和本地开发）

**连接参数说明**：
- `uri` 指定数据库文件路径
- Milvus Lite 会将数据存储在本地文件中


```python
def connect_local_milvus():
    """连接本地 Milvus 服务"""
    
    # 连接参数
    uri = "milvus_demo.db"
    print(f"正在连接到 {uri}...")
    
    # 创建客户端
    client = MilvusClient(uri=uri)
    
    # 获取服务器版本
    version = client.get_server_version()
    print(f"Milvus 版本：{version}")
    print("✓ 连接成功！")
    
    return client

# 运行示例
client = connect_local_milvus()
```

---

## 示例 2: 连接远程 Milvus

适用于生产环境或 Docker 部署的 Milvus


```python
def connect_remote_milvus():
    """连接远程 Milvus 服务"""
    
    # 连接参数
    host = "192.168.142.128"
    port = "19530"  # Milvus 默认端口
    uri = f"http://{host}:{port}"
    
    print(f"正在连接到 {uri}...")
    
    # 创建客户端
    client = MilvusClient(uri=uri)
    
    # 获取服务器版本
    version = client.get_server_version()
    print(f"Milvus 版本：{version}")
    print("✓ 连接成功！")
    
    return client

# 运行示例
client = connect_remote_milvus()
```

    正在连接到 http://192.168.142.128:19530...
    Milvus 版本：2.6.14
    ✓ 连接成功！
    

---

## 示例 3: 带认证的连接

适用于生产环境，需要权限控制的场景


```python
def connect_with_auth():
    """带用户名密码认证的连接"""
    
    # 连接参数
    host = "localhost"
    port = "19530"
    uri = f"http://{host}:{port}"
    
    # 认证信息（实际使用时请替换为真实用户名和密码）
    user = "root"
    password = "Milvus"
    
    print(f"正在连接到 {uri}...")
    print(f"用户：{user}")
    
    # 创建带认证的客户端
    client = MilvusClient(uri=uri, user=user, password=password)
    
    print("✓ 连接成功！")
    
    return client

# 运行示例
client = connect_with_auth()
```

---

## 连接参数详解

| 参数 | 默认值 | 说明 |
|:---:|:---:|:---|
| uri | localhost:19530 | 服务地址，格式：`http://host:port` |
| user | None | 用户名（可选），生产环境建议开启认证 |
| password | None | 密码（可选），与 user 一起使用 |
| token | None | API Token（云服务等场景，如 Zilliz Cloud） |
| db_name | default | 数据库名称，Milvus 2.3+ 支持多数据库 |

### URI 格式说明

```python
# 1. 本地文件（Milvus Lite）
uri = "milvus_demo.db"

# 2. HTTP 连接
uri = "http://localhost:19530"

# 3. HTTPS 连接（云服务）
uri = "https://your-cluster.zillizcloud.com"

# 4. 完整格式
uri = "http://user:password@localhost:19530/dbname"
```

---

## 示例 4: 检查连接健康状态

确保服务正常运行


```python
def check_connection_health():
    """检查 Milvus 连接健康状态"""
    
    # 优先使用远程连接（Docker Milvus）
    uri = "http://192.168.142.128:19530"
    client = MilvusClient(uri=uri)
    
    # 检查项列表
    checks = {
        "版本检查": lambda: client.get_server_version(),
        "连接检查": lambda: client.list_collections() is not None,
    }
    
    print("执行健康检查...\n")
    
    for name, check_func in checks.items():
        try:
            result = check_func()
            print(f"[OK] {name}: {result}")
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
    
    print("\n[OK] 连接状态正常")
    
    return True

# 运行健康检查
check_connection_health()
```

---

## 下一步

连接成功后，接下来学习：
- `02_create_collection.py` - 创建集合
- 掌握如何定义向量字段和标量字段
