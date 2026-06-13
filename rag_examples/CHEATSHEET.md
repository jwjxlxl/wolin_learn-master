# RAG + Milvus 速查表

> 快速查阅：Embedding 模型、Milvus 索引、切片策略、常见错误

---

## 一、Embedding 模型速查

| 模型 | 维度 | 语言 | 提供方 | 推荐场景 |
|------|------|------|--------|----------|
| `text-embedding-v4` | **1024** | 中/英 | 阿里云百炼 | ★ 本课程默认，性价比高 |
| `text-embedding-v3` | 1024/1536/2048 | 中/英 | 阿里云百炼 | 需要更高维度时 |
| `bge-large-zh-v1.5` | 1024 | 中文 | BAAI（开源） | 本地部署、隐私敏感 |
| `bge-m3` | 1024 | 多语言 | BAAI（开源） | 多语言场景 |
| `m3e-base` | 768 | 中文 | 开源 | 轻量级本地部署 |

> ⚠️ **本课程统一使用 1024 维**，配置在 `milvus_config.py` → `DEFAULT_DIMENSION = 1024`

---

## 二、Milvus 索引类型决策树

```
数据量 < 1 万？
  ├── 是 → FLAT（精确搜索，100% 精度，无需索引）
  └── 否 → 数据量 1 万 ~ 100 万？
            ├── 是 → IVF_FLAT（nlist = √N，nprobe = √nlist）
            └── 否 → 需要最高精度？
                      ├── 是 → HNSW（M=16, efConstruction=200, ef=64）
                      └── 否 → IVF_PQ（节省内存，精度略低）
```

| 索引 | 适用量级 | 速度 | 精度 | 内存 | 参数 |
|------|---------|------|------|------|------|
| **FLAT** | <1万 | 慢(500ms) | 100% | 低 | 无 |
| **IVF_FLAT** | 1万~100万 | 快(50ms) | 95-98% | 中 | `nlist`, `nprobe` |
| **IVF_PQ** | >100万 | 很快(20ms) | 90-95% | 低 | `nlist`, `m`, `nbits` |
| **HNSW** | 不限 | 最快(10ms) | 98-99% | 高 | `M`, `efConstruction`, `ef` |
| **AUTOINDEX** | 不限 | 自动 | 自动 | 自动 | 无需手动配置 |

---

## 三、度量类型选择

| 类型 | 公式 | 判断 | 使用场景 |
|------|------|------|----------|
| **COSINE** ★ | cos(θ) | 越大越相似 | 语义检索（推荐默认） |
| **L2** | √(Σ(xi-yi)²) | 越小越相似 | 向量未归一化 |
| **IP** | Σ(xi*yi) | 越大越相似 | 归一化后等价 COSINE |

---

## 四、文档切片策略速查

| 方法 | 难度 | 适用场景 | 推荐参数 |
|------|------|----------|----------|
| **固定字符** | ⭐ | 快速原型 | chunk=500, overlap=50 |
| **滑动窗口** | ⭐⭐ | 保持上下文 | window=500, step=250 (50%) |
| **AI 切片** | ⭐⭐⭐ | 语义完整性 | 规则预处理 + LLM 精切 |
| **摘要切片** | ⭐⭐⭐ | 长文档、多文档 | 摘要检索 → 原文返回 |

### 推荐参数速查

| 文档类型 | chunk_size | chunk_overlap | 方法 |
|----------|-----------|---------------|------|
| 短文本(<1000字) | 200 | 50 | 固定/滑动 |
| 中文本(1000-5000字) | 500 | 100 | 滑动窗口 |
| 长文本(>5000字) | 800 | 150 | 滑动窗口 |
| 法律/合同 | 300 | 100 | AI 切片 |
| 对话记录 | 200 | 50 | 按句子边界 |

---

## 五、检索方法速查

```
检索请求
  ├── 精确匹配（ID/类别/日期） → 标量查询 Scalar Query
  ├── 语义相似 → 向量检索 Vector Search
  ├── 关键词匹配 → BM25 关键词检索
  ├── 综合最优 → 混合检索 Hybrid Search (RRF 融合)
  └── 精益求精 → 混合检索 + Rerank
```

### RRF vs 加权排序

| 方法 | 优点 | 缺点 | 使用场景 |
|------|------|------|----------|
| **RRF** | 无需调权重、鲁棒性好 | 无法控制比例 | 通用场景（推荐） |
| **加权排序** | 可精确控制比例 | 需调参、需归一化 | 明确知道权重时 |

```python
# RRF（推荐，Milvus 原生支持）
ranker = Function(name="rrf", function_type=FunctionType.RERANK,
                  params={"reranker": "rrf", "k": 100})

# 加权排序（精细控制）
ranker = Function(name="weighted", function_type=FunctionType.RERANK,
                  params={"reranker": "weighted", "weights": [0.6, 0.4]})
```

---

## 六、常见错误速查

| 错误信息 | 原因 | 解决方案 |
|----------|------|----------|
| `dimension mismatch` | 向量维度与 Collection 不匹配 | 检查嵌入模型维度 = Collection 维度（本课程：**1024**） |
| `collection not found` | Collection 不存在 | 先运行 `create_collection()` 或 `vdb_init_milvus.py` |
| `index not found` | 未创建索引 | `client.create_index()` 或 `prepare_index_params()` |
| `field xxx not exist` | 输出字段名拼写错误 | 检查 `output_fields` 与 Schema 定义一致 |
| `Connection refused` | Milvus 服务未启动 | `docker compose up -d` 或检查 `MILVUS_URI` |
| `API key invalid` | 环境变量未设置 | 复制 `.env.example` → `.env` 并填入 API Key |
| `Milvus Lite not supported` | Windows 不支持 Milvus Lite | 使用 Docker 方式（见 `MILVUS_CONFIG.md`） |
| `BM25 function not supported` | Milvus 版本过低 | 需要 Milvus 2.4+，升级或使用 Docker |

---

## 七、环境变量清单

```bash
# 必备（rag_examples + rag_demo）
ALIYUN_API_KEY=sk-xxx        # 阿里云百炼 API Key

# rag_demo 额外需要
DEEPSEEK_API_KEY=sk-xxx      # DeepSeek API Key

# Milvus 连接（可选，有默认值）
MILVUS_URI=http://localhost:19530   # Milvus 地址
MILVUS_DB_NAME=default              # 数据库名
```

---

## 八、课程学习路径

```
00_setup          → 环境搭建（Docker + Python + API Key）
embedding_examples → Embedding 基础（API + 本地 + 对比）
01_milvus_basics  → Milvus 基础（连接/集合/插入/索引）
02_document_chunking → 文档切片（固定/滑动/AI/摘要）
03_retrieval_methods → 检索方法（标量/向量/关键词/混合/Rerank）
04_rag_api        → RAG API 封装
05_rag_pipeline   → 完整 RAG 管道
06_rag_advanced   → 高级检索（BM25 Function/稀疏向量/双集合/RRF vs 加权）
06_rag_evaluation → RAG 评估（RAGAS/忠实度/相关性）
rag_demo          → 综合实战项目
```

---

## 九、常用代码片段

### 连接 Milvus
```python
from rag_examples.milvus_config import MILVUS_URI, get_milvus_client
client = get_milvus_client()
```

### 生成 Embedding
```python
from rag_demo.util.embedding import generate_embedding
vector = generate_embedding("你好世界")  # 返回 1024 维向量
```

### 创建简单 Collection
```python
client.create_collection(
    collection_name="my_docs",
    dimension=1024,
    auto_id=True,
    metric_type="COSINE"
)
```

### 混合检索（稠密 + BM25 + RRF）
```python
from pymilvus import AnnSearchRequest, Function, FunctionType

req_dense = AnnSearchRequest(data=[dense_vector], anns_field="dense_vector",
                              param={"nprobe": 10}, limit=5)
req_sparse = AnnSearchRequest(data=[query_text], anns_field="sparse_vector",
                               param={"metric_type": "BM25"}, limit=5)
ranker = Function(name="rrf", function_type=FunctionType.RERANK,
                  params={"reranker": "rrf", "k": 100})

results = client.hybrid_search(
    collection_name="my_docs",
    reqs=[req_dense, req_sparse],
    ranker=ranker, limit=5,
    output_fields=["text", "title"]
)
```

### 文本切片
```python
from rag_demo.util.text_splitter import split_text
chunks = split_text(long_text, chunk_size=500, chunk_overlap=50)
```
