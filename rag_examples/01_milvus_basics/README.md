# 01_milvus_basics — Milvus 基础操作

> 难度：⭐⭐ | 前置：了解 Embedding 概念 | 后续：02_document_chunking 文档切片

## 🎯 本节目标

学完本模块后，你将能够：

- ✅ 用多种方式连接 Milvus 服务（本地 Docker / 远程 / 带认证）
- ✅ 创建不同复杂度的 Collection（简单 / 自定义字段 / 多向量 / 动态字段）
- ✅ 单条或批量插入数据，理解 `auto_id` 的使用场景
- ✅ 为 Collection 创建合适的索引加速检索（FLAT / IVF_FLAT / HNSW）
- ✅ 理解度量类型的含义并能做出正确选择（COSINE / L2 / IP）

## 🗺️ 学习路径

```
连接 Milvus → 创建 Collection → 插入数据 → 创建索引
    ↑              ↑              ↑            ↑
  第1步          第2步          第3步        第4步

文件：01_connect_milvus.py → 02_create_collection.py
                              → 03_insert_data.py → 04_create_index.py
```

## 📁 文件说明

| # | 文件 | 内容 | 难度 | 核心收获 |
|---|------|------|------|---------|
| 1 | [01_connect_milvus.py](01_connect_milvus.py) | 连接 Milvus | ⭐ | 本地/远程/认证连接、健康检查、多数据库 |
| 2 | [02_create_collection.py](02_create_collection.py) | 创建 Collection | ⭐⭐ | 简单建表/自定义字段/多向量/动态字段 |
| 3 | [03_insert_data.py](03_insert_data.py) | 插入数据 | ⭐⭐ | 单条插入/批量插入/自定义 ID |
| 4 | [04_create_index.py](04_create_index.py) | 创建索引 | ⭐⭐⭐ | FLAT/IVF_FLAT/HNSW 索引对比 |

## 📖 学习指导

### 第 1 步：连接 Milvus（01_connect_milvus.py）

**本节讲什么：**
- Milvus 的 4 种连接方式（Lite / Docker / 远程 / 带认证）
- URI 格式解析（`http://localhost:19530` 等）
- 数据库级别操作（创建/列出/删除数据库）
- 连接健康检查

**核心概念：**

```
MilvusClient 连接方式：

1. Milvus Lite（本地文件）
   uri = "milvus_demo.db"

2. Docker 本地服务
   uri = "http://localhost:19530"          ← 本课程默认

3. 远程服务器
   uri = "http://47.x.x.x:19530"

4. 带认证
   MilvusClient(uri=..., user="root", password="Milvus")
```

**关键提醒：**
- ⚠️ 连接地址通过 `milvus_config.py` 的 `MILVUS_URI` 统一读取，来自环境变量
- ⚠️ 多数据库功能仅 Milvus 2.3+ Docker 版本支持，Lite 版不支持

---

### 第 2 步：创建 Collection（02_create_collection.py）

**本节讲什么：**
- 什么是 Collection（类比 Excel 表格）
- 简单建表：只需指定 `dimension`
- 自定义建表：定义字段 Schema（VARCHAR / INT64 / FLOAT_VECTOR）
- 多向量 Collection（文本 + 图像等多模态场景）
- 动态字段（Milvus 2.3+ 新特性，无需预定义 Schema 即可插入任意字段）

**核心概念：**

```
Collection 字段类型：

┌─────────────┬────────────────────────────────────┐
│  字段类型    │  说明                               │
├─────────────┼────────────────────────────────────┤
│ INT64       │ 整数（用于主键 ID、浏览量等）          │
│ VARCHAR     │ 字符串（用于内容、标题、分类等）        │
│ FLOAT_VECTOR│ 向量（用于 Embedding 输出）           │
└─────────────┴────────────────────────────────────┘

简单建表 vs 自定义建表：

简单建表：                            自定义建表：
client.create_collection(            fields = [
    collection_name="docs",              FieldSchema(name="id", ...),
    dimension=1024,                      FieldSchema(name="content", ...),
    auto_id=True,                        FieldSchema(name="title", ...),
    metric_type="COSINE",                FieldSchema(name="vector", ...),
)                                    ]
                                     schema = CollectionSchema(fields=fields)
                                     client.create_collection(
                                         collection_name="docs",
                                         schema=schema,
                                     )
```

**度量类型选择（⭐ 重要）：**

| 类型 | 何时用 | 判断标准 |
|------|--------|---------|
| **COSINE**（推荐） | 绝大多数场景 | 值越大越相似，范围 -1 ~ 1 |
| L2 | 向量未归一化时 | 值越小越相似 |
| IP | 与 COSINE 归一化后等价 | 值越大越相似 |

---

### 第 3 步：插入数据（03_insert_data.py）

**本节讲什么：**
- Embedding 的原理和模拟生成
- 单条插入 vs 批量插入（推荐批量）
- 自动分配 ID（`auto_id=True`）vs 手动指定 ID（`auto_id=False`）
- 自定义字段 Collection 的完整插入

**核心概念：**

```
数据插入流程：

1. 准备文本数据
   documents = [{"content": "..."}, {"content": "..."}]

2. 生成 Embedding 向量（用真实 API 或模拟）
   vectors = generate_embedding([d["content"] for d in documents])

3. 组装数据 + 向量
   data = [{"content": doc["content"], "vector": vectors[i]} for i, doc in enumerate(documents)]

4. 插入 Collection
   client.insert(collection_name=collection_name, data=data)
```

**最佳实践（⭐ 重要）：**

| 建议 | 原因 |
|------|------|
| 用批量插入代替单条插入 | 减少网络开销，性能更好 |
| 批量大小 100-1000 条/批 | 太少网络开销大，太多内存占用高 |
| 插入完成后再建索引 | 避免每次插入都更新索引 |
| 向量维度必须匹配 Collection | 1024 维向量插入非 1024 维 Collection 会失败 |
| 用 `auto_id=True` | 无需手动管理 ID，推荐 |

---

### 第 4 步：创建索引（04_create_index.py）

**本节讲什么：**
- 什么是索引（类比"书的目录"，让查找从逐页翻变成按目录找）
- FLAT 索引：暴力搜索，100% 精度
- IVF_FLAT 索引：分簇搜索，平衡速度和精度
- HNSW 索引：图导航，高精度低延迟
- 索引性能对比和选择建议

**核心概念：**

```
索引类型选择：

数据量 < 1 万      → FLAT（精确，无需训练）
数据量 1 万-100 万 → IVF_FLAT（nlist = √N）
数据量 > 100 万    → IVF_PQ（压缩存储）
高精度要求         → HNSW（精度高，速度快，内存大）
内存受限           → IVF_SQ8（8bit 压缩）

IVF_FLAT 参数：
  nlist = √N       （向量总数 N 的平方根）
  nprobe = √nlist  （搜索时探查的分组数，越多越准但越慢）

HNSW 参数：
  M = 16            （每个节点的连接数，默认 16）
  efConstruction = 200（构建时的搜索深度）
  ef = 64           （搜索时的探查深度，越大越准但越慢）
```

**索引性能对比（100 万向量，1024 维）：**

| 索引 | 构建时间 | 检索延迟 | 召回率 |
|------|---------|---------|--------|
| FLAT | 无需构建 | ~500ms | 100% |
| IVF_FLAT | ~30 秒 | ~50ms | 95-98% |
| HNSW | ~60 秒 | ~10ms | 98-99% |

---

## 🧠 核心知识点总结

### 1. 完整操作流程

```
连接 Milvus ──→ 创建 Collection ──→ 插入数据 ──→ 创建索引
     │                │                  │              │
  MilvusClient   create_collection     insert       create_index
  (uri=MILVUS_URI) (dim=1024)          (data=list)   (index_type="IVF_FLAT")
```

### 2. Collection 类比

| Milvus 概念 | 类比 | 说明 |
|------------|------|------|
| Collection | Excel 表格 | 存储数据的容器 |
| Field | 表格的列 | 定义数据类型和约束 |
| Primary Key | 主键列 | 唯一标识每行数据 |
| Vector | 数字指纹列 | Embedding 生成的语义向量 |
| Index | 图书索引 | 加速查找的数据结构 |

### 3. 关键配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dimension` | 1024 | 必须与 Embedding 模型输出维度一致 |
| `metric_type` | "COSINE" | 推荐默认，适用于语义检索 |
| `auto_id` | True | Milvus 自动分配 ID |

### 4. 索引 vs 检索速度

```
无索引（FLAT）:  需要比较 N 次      → 100 万条 = 500ms
IVF_FLAT:        需要比较 √N 次    → 100 万条 = 50ms
HNSW:            图导航只需几步     → 100 万条 = 10ms
```

---

## 🏃 快速开始

```bash
# 确保 Milvus 已启动
docker ps | grep milvus

# 从第 1 个文件开始
python 01_milvus_basics/01_connect_milvus.py
python 01_milvus_basics/02_create_collection.py
python 01_milvus_basics/03_insert_data.py
python 01_milvus_basics/04_create_index.py
```

## ⚠️ 常见问题

### Q: 连接 Milvus 报错 "connection refused"？
A: 检查 Docker 是否在运行：`docker ps | grep milvus`。如果没启动，执行 `docker compose up -d`，等待 15 秒后再试。

### Q: 插入数据时维度不匹配报错？
A: 检查 `milvus_config.py` 中的 `DEFAULT_DIMENSION` 是否为 1024，与你的 Embedding 模型输出维度一致。

### Q: Collection 创建后查询返回 0 条数据？
A: 确认是否执行了 `insert` 操作。Collection 创建后是空的，需要插入数据才会有内容。

### Q: 创建索引需要多久？
A: 少量数据（几百条）几乎瞬间完成。百万级数据需要几秒到几分钟，取决于索引类型和硬件性能。

---

**作者**: Luke
**版本**: 1.0
**最后更新**: 2026-06-16
