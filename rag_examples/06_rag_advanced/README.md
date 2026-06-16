# 06_rag_advanced — 高级检索技术

> 难度：⭐⭐⭐⭐ | 前置：03_retrieval_methods 检索方法 + 05_rag_pipeline 完整流水线 | 后续：rag_demo 实战项目

## 🎯 本节目标

学完本模块后，你将能够：

- ✅ 在 Milvus 中使用 BM25 Function 自动生成稀疏向量
- ✅ 掌握 `AnnSearchRequest` + `hybrid_search()` 原生混合检索 API
- ✅ 设计双 Collection 的 RAG 检索架构（文档切片 + 问答对）
- ✅ 理解从 mock 示例代码到生产代码的跨越步骤
- ✅ 使用 `UniversalEmbedder` 统一管理 mock/real 模式切换

## 🗺️ 学习路径

```
BM25 Function + 原生混合检索 → 双集合架构设计 → Mock → Real 过渡
        ↑                        ↑                    ↑
   Milvus 2.4+ 高级 API      生产级架构模式      教学代码工程化
   (⭐⭐⭐⭐)                  (⭐⭐⭐⭐)            (⭐⭐⭐)

文件：01_hybrid_search_advanced.py → 02_dual_collection_design.py
                                     → 03_from_mock_to_real.py
```

## 📁 文件说明

| # | 文件 | 内容 | 难度 | 核心收获 |
|---|------|------|------|---------|
| 1 | [01_hybrid_search_advanced.py](01_hybrid_search_advanced.py) | BM25 Function + 混合检索 API | ⭐⭐⭐⭐ | AnnSearchRequest、RRF/加权融合 |
| 2 | [02_dual_collection_design.py](02_dual_collection_design.py) | 双集合 RAG 架构 | ⭐⭐⭐⭐ | 文档切片 + QA 对并行检索 |
| 3 | [03_from_mock_to_real.py](03_from_mock_to_real.py) | Mock → Real 过渡 | ⭐⭐⭐ | UniversalEmbedder、迁移检查清单 |

---

## 📖 学习指导

### 本节在课程中的位置

```
前期学习（教学代码）：
  01_milvus_basics     → Milvus 基本操作
  02_document_chunking → 文档切片
  03_retrieval_methods → 各种检索方法
  04_rag_api           → API 封装
  05_rag_pipeline      → 完整流水线

本节（高级技术）：
  06_rag_advanced      → 生产级架构 ← 你在这里

最终目标：
  rag_demo 实战项目    → 综合实战应用
```

---

### 第 1 步：BM25 Function 与原生混合检索（01_hybrid_search_advanced.py）

**本节讲什么：**
- Milvus 2.4+ 的 BM25 Function——让 Milvus 自动从文本生成稀疏向量
- `enable_analyzer` + `enable_match` 两个开关的作用
- `AnnSearchRequest`——多路检索请求对象
- `hybrid_search()`——原生混合检索 API
- RRF vs 加权排序两种融合策略的代码对比

**核心概念：**

```
Milvus 2.4 之前的混合检索：
  1. 手动用 jieba 分词
  2. 手动用 rank-bm25 计算稀疏向量
  3. 手动插入稀疏向量到 Collection
  4. 手动在应用层融合结果

Milvus 2.4+ 之后的混合检索：
  1. Schema 中定义 BM25 Function
  2. 插入数据时只提供 text，sparse_vector 自动生成
  3. 调用 hybrid_search() API，结果自动融合

本质：把 BM25 计算从"应用层"移到了"数据库层"
```

**BM25 Function 的两个关键开关：**

```python
# 在 VARCHAR 字段上开启：
schema.add_field(
    field_name="text", datatype=DataType.VARCHAR, max_length=2000,
    enable_analyzer=True,   # ← 开关1：启用分词器
    enable_match=True,      # ← 开关2：启用 BM25 匹配
)

# 然后定义 BM25 Function：
bm25_function = Function(
    name="text_bm25",
    input_field_names=["text"],           # ← 从哪个字段读取文本
    output_field_names=["sparse_vector"], # ← 输出到哪个稀疏向量字段
    function_type=FunctionType.BM25,
)
schema.add_function(bm25_function)

# 最后为稀疏向量字段创建索引：
index_params.add_index(
    field_name="sparse_vector",
    index_type="SPARSE_INVERTED_INDEX",   # ← 专用于 BM25 的索引
    metric_type="BM25",
)
```

**完整 Collection 结构：**

```
支持混合检索的 Collection Schema：

┌─────────────────┬───────────────────┬────────────────────┐
│  字段名          │  数据类型          │  说明               │
├─────────────────┼───────────────────┼────────────────────┤
│  id             │  INT64 (主键)      │  自动递增 ID        │
│  text           │  VARCHAR + analyzer│  原始文本（可分词）  │
│  title          │  VARCHAR           │  标题               │
│  dense_vector   │  FLOAT_VECTOR      │  稠密语义向量       │
│  sparse_vector  │  SPARSE_FLOAT_VEC  │  BM25 稀疏向量      │
└─────────────────┴───────────────────┴────────────────────┘

BM25 Function：text 字段 → 自动填充 sparse_vector 字段

两个索引：
  dense_vector  → AUTOINDEX (COSINE)
  sparse_vector → SPARSE_INVERTED_INDEX (BM25)
```

**三种检索方式对比（同一查询）：**

```python
查询："向量数据库有哪些特点？"

纯稠密向量检索（语义）：
  → "向量数据库"的语义 → 找语义相似的文档
  → 可能找到"数据库"相关但不含"向量"的文档

纯 BM25 检索（关键字）：
  → 查找包含"向量"、"数据库"字符的文档
  → 精确匹配，但不懂"特点"的含义

混合检索（语义 + 关键字 + RRF）：
  → 两路同时检索，RRF 融合排名
  → 既找到语义相关的，又保证关键字命中
```

**hybrid_search() 完整代码：**

```python
# 1. 创建稠密检索请求
req_dense = AnnSearchRequest(
    data=[query_vector],        # 查询向量
    anns_field="dense_vector",  # 在哪个字段检索
    param={"nprobe": 10},       # 索引搜索参数
    limit=5,
)

# 2. 创建稀疏检索请求（注意：这里直接传原始文本！）
req_sparse = AnnSearchRequest(
    data=[query],               # 原始文本，不是向量
    anns_field="sparse_vector", # BM25 Function 自动处理
    param={"metric_type": "BM25"},
    limit=5,
)

# 3. RRF 融合
ranker = Function(
    name="rrf",
    function_type=FunctionType.RERANK,
    params={"reranker": "rrf", "k": 100},
)

# 4. 执行混合检索
results = client.hybrid_search(
    collection_name=collection_name,
    reqs=[req_dense, req_sparse],  # 多路请求
    ranker=ranker,
    limit=5,
    output_fields=["title", "text"],
)
```

**RRF vs 加权排序代码对比：**

```python
# RRF（更简单，更鲁棒，推荐默认）
ranker = Function(
    name="rrf", function_type=FunctionType.RERANK,
    params={"reranker": "rrf", "k": 100}
)

# 加权排序（更灵活，需要调参）
ranker = Function(
    name="weighted", function_type=FunctionType.RERANK,
    params={
        "reranker": "weighted",
        "weights": [0.6, 0.4],    # 稠密 : 稀疏 = 6 : 4
        "norm_score": True         # 必须归一化到同一量纲
    }
)
```

**何时用哪种融合策略：**

| 场景 | 推荐方法 | 原因 |
|------|---------|------|
| 不确定权重 | RRF | 不需要调参，鲁棒性好 |
| 语义更重要 | 加权 [0.7, 0.3] | 明确偏重语义 |
| 关键字更重要 | 加权 [0.3, 0.7] | 明确偏重关键字 |
| 通用场景 | RRF 或 加权 [0.6, 0.4] | 平衡 |

---

### 第 2 步：双集合 RAG 架构（02_dual_collection_design.py）

**本节讲什么：**
- 为什么单一 Collection（只有文档切片）不够用
- 双集合设计：`document_chunks`（文档切片）+ `qa_pairs`（问答对）
- 并行检索两个集合，合并上下文
- 构建带引用来源的 RAG 提示词
- 这是 `rag_demo/core/rag_query.py` 的教学简化版

**核心概念：**

```
单一集合的局限：

  只有文档切片 → 只能检索到原始文本片段
  → 缺少"已整理的知识"（如 FAQ 中的标准问答）
  → 缺少"推理过程"等半结构化信息

双集合架构：

  Collection 1: document_chunks（文档切片）
    ├── text: 原始文档文本片段（"证据"）
    ├── file_name: 来源文件名
    └── chunk_index: 切片位置

  Collection 2: qa_pairs（问答对）
    ├── question: 标准问题
    ├── answer: 标准答案
    └── reasoning: 推理过程（比单纯问答更有价值！）

  为什么分开？
    文档切片 → 提供原始证据，覆盖全面
    问答对 → 提供已整理的知识，精准命中
    两者互补，并行检索提高召回率
```

**完整检索流程：**

```
用户问题："Milvus 支持什么检索方式？"
     ↓
  ┌──────┬──────┐
  │ 并行检索     │
  ├─────────────┤
  │document_    │qa_pairs       │
  │chunks       │混合检索       │
  │混合检索     │               │
  └─────┬───────┴──────┬───────┘
        ↓              ↓
  文档片段结果     问答对结果
  [相似度][文本]   [相似度][问题][答案][推理]
        ↓              ↓
  ┌─────┴──────┬──────┴──────┐
  │  上下文合并 + 来源标记      │
  └────────────┬─────────────┘
               ↓
        LLM 生成带引用的答案
```

**双集合 Schema 设计要点：**

```python
# 文档切片集合
doc_schema.add_field("text", VARCHAR, enable_analyzer=True, enable_match=True)
doc_schema.add_field("file_name", VARCHAR)           # ← 溯源字段
doc_schema.add_field("chunk_index", INT32)           # ← 位置字段
doc_schema.add_field("dense_vector", FLOAT_VECTOR)
doc_schema.add_field("sparse_vector", SPARSE_FLOAT_VECTOR)
# BM25 Function: text → sparse_vector

# 问答对集合
qa_schema.add_field("question", VARCHAR, enable_analyzer=True, enable_match=True)
qa_schema.add_field("answer", VARCHAR)               # ← 答案字段
qa_schema.add_field("reasoning", VARCHAR)            # ← 推理过程
qa_schema.add_field("dense_vector", FLOAT_VECTOR)
qa_schema.add_field("sparse_vector", SPARSE_FLOAT_VECTOR)
# BM25 Function: question → sparse_vector  ← 注意：对 question 字段做 BM25
```

**关键设计决策：**

| 决策 | 选择 | 原因 |
|------|------|------|
| QA 对的 BM25 对哪个字段 | `question` | 检索时用用户问题匹配 QA 问题 |
| 两个集合并行还是串行 | 并行 | 不互相等待，减少延迟 |
| 合并上下文时如何区分 | Markdown 标题 | LLM 能理解结构化的来源标记 |
| 是否需要 source 字段 | 是 | 方便验证 LLM 答案的准确性 |

**RAG 提示词构建示例：**

```
你是一个基于检索增强生成（RAG）的智能问答助手。
请根据以下检索到的上下文回答用户的问题。

## 检索到的上下文

[文档片段1] Milvus 向量数据库支持混合检索...
来源：milvus_intro.txt

[问答对1]
问：Milvus 支持哪些检索方式？
答：Milvus 支持标量查询、稠密向量检索...
推理：Milvus 2.4+ 引入了 BM25 Function...

用户问题：什么是 RAG？
```

---

### 第 3 步：从 Mock 到 Real 的过渡（03_from_mock_to_real.py）

**本节讲什么：**
- Mock 向量的局限性和教学价值
- 如何将代码中的 mock 替换为真实 Embedding API
- `UniversalEmbedder` 类——统一管理 mock/real 两种模式
- 从"教学代码"到"生产代码"的迁移检查清单

**核心概念：**

```
为什么课程早期用 Mock 向量？
  1. 降低入门门槛——无需 API Key 就能跑通流程
  2. 聚焦核心概念——学生先理解"流程"而非"API 调用"
  3. 离线可运行——不依赖网络和 API 服务

Mock 向量的致命局限：
  向量是随机生成的 → 没有语义信息 → 检索结果毫无意义
  相同文本 → 相同向量（seed 保证可重复）
  不同文本 → 向量距离随机（不代表语义距离）

什么时候必须切换到 Real？
  → 当你需要验证检索效果时
  → 当你构建实际项目时
  → 当你评估不同切片/检索参数时
```

**Mock vs Real 对比：**

| 维度 | Mock | Real |
|------|------|------|
| 需要 API Key | ❌ | ✅ |
| 语义正确性 | ❌ 无 | ✅ 有 |
| 检索效果 | ❌ 随机排列 | ✅ 相关文档在前 |
| 代码调试 | ✅ 方便 | ❌ 依赖网络 |
| 适用场景 | 概念演示 | 实战项目 |

**UniversalEmbedder——统一接口：**

```python
class UniversalEmbedder:
    """mock/real 统一的 Embedding 接口"""

    def __init__(self, mode="auto", dim=1024):
        # mode="auto" → 自动检测 API Key，有则 real，无则 mock
        # mode="mock" → 强制使用 mock
        # mode="real" → 强制使用 real

    def embed(self, text: str) -> list:
        """对单个文本生成向量"""

    def embed_batch(self, texts: list) -> list:
        """批量生成向量（每批 25 条，避免 API 限制）"""
```

**使用方式：**

```python
# 教学演示（无需 API Key）
embedder = UniversalEmbedder(mode="mock")
vectors = embedder.embed_batch(documents)

# 自动模式（推荐：有 Key 就用 real，没有就 mock）
embedder = UniversalEmbedder(mode="auto")
# 自动检测到 ALIYUN_API_KEY → 使用真实 API
vectors = embedder.embed_batch(documents)

# 生产环境（明确指定 real）
embedder = UniversalEmbedder(mode="real")
vectors = embedder.embed_batch(documents)
```

**5 步替换流程：**

```
第 1 步：安装依赖
  pip install openai python-dotenv

第 2 步：配置 API Key
  在 .env 文件中设置 ALIYUN_API_KEY=sk-xxx

第 3 步：找到代码中的 mock 函数
  搜索：generate_mock_embeddings() / random.seed(hash(text))

第 4 步：替换为真实 Embedding 调用
  from rag_demo.util.embedding import generate_embedding
  vectors = [generate_embedding(doc) for doc in documents]

第 5 步：运行测试，验证效果
  好的结果：相关文档排在前面
  差的结果：随机排列（说明 Embedding 有问题）
```

**迁移检查清单：**

```
□ 1. Embedding 替换
     □ 找到所有 mock_embedding() 调用
     □ 替换为真实 Embedding
     □ 验证检索结果相关性

□ 2. 配置管理
     □ API Key 通过 .env 管理
     □ MILVUS_URI 通过环境变量配置
     □ 不硬编码任何凭证

□ 3. 错误处理
     □ API 调用失败的降级策略
     □ Milvus 连接失败的提示
     □ 数据格式校验

□ 4. 性能优化
     □ 批量插入（batch_size=100~1000）
     □ 使用 AUTOINDEX
     □ 大量数据考虑分区

□ 5. 质量评估
     □ 构建测试问题集
     □ 人工评估前 10 条结果的相关性
     □ 考虑引入 RAGAS 评估框架
```

---

## 🧠 核心知识点总结

### 本节在 RAG 架构中的位置

```
知识库构建：
  加载文档 → 切片 → Embedding → 存入 Milvus
                                ↑
                     BM25 Function 自动生成稀疏向量

检索流程：
  用户问题 → Embedding
     ├──→ document_chunks 混合检索（dense + sparse + RRF）
     └──→ qa_pairs 混合检索（dense + sparse + RRF）
              ↓
         上下文合并 → LLM 生成
```

### 三种检索架构对比

| 架构 | 复杂度 | 效果 | 适用 |
|------|--------|------|------|
| 单集合向量检索 | ⭐ | 一般 | 快速原型 |
| 单集合混合检索 | ⭐⭐⭐ | 良好 | 生产环境首选 |
| 双集合混合检索 | ⭐⭐⭐⭐ | 优秀 | 高质量 RAG |

### 关键 API 速查

| API | 用途 | Milvus 版本 |
|-----|------|-------------|
| `enable_analyzer=True` | 启用内置分词器 | 2.4+ |
| `enable_match=True` | 启用 BM25 匹配 | 2.4+ |
| `Function(FunctionType.BM25)` | 定义 BM25 Function | 2.4+ |
| `AnnSearchRequest` | 创建多路检索请求 | 2.3+ |
| `client.hybrid_search()` | 执行混合检索 | 2.3+ |
| `SPARSE_INVERTED_INDEX` | 稀疏向量索引 | 2.4+ |

---

## 🏃 快速开始

```bash
# 确保 Milvus 2.4+ 已启动
docker ps | grep milvus

# 从 BM25 Function 开始
python 06_rag_advanced/01_hybrid_search_advanced.py
python 06_rag_advanced/02_dual_collection_design.py
python 06_rag_advanced/03_from_mock_to_real.py
```

## ⚠️ 常见问题

### Q: 我的 Milvus 版本不支持 BM25 Function 怎么办？
A: BM25 Function 需要 Milvus 2.4+。检查版本：`docker ps | grep milvus`。如果版本较低，升级 Docker 镜像。或者在应用层手动实现 BM25（参考 03_retrieval_methods 中的教学代码）。

### Q: `hybrid_search()` 报错 "sparse_vector field not found"？
A: 检查 Schema 中是否正确定义了 `SPARSE_FLOAT_VECTOR` 类型的字段，以及是否为该字段创建了 `SPARSE_INVERTED_INDEX` 索引。

### Q: 双集合架构会不会增加太多延迟？
A: 不会。两个集合是**并行检索**的（同时发起两个请求），总延迟约等于单次检索的延迟，而不是两倍。

### Q: QA 对的 reasoning 字段有什么用？
A: 不仅返回答案，还返回"为什么是这个答案"的推理过程。这能帮助 LLM 更好地理解上下文，生成更有说服力的回答。同时也能让用户了解答案的来源。

### Q: mock 模式下检索结果是随机的，怎么判断代码逻辑是否正确？
A: mock 模式用于验证**流程**（插入→检索→返回结构）是否正确。语义正确性需要用 real 模式验证。用 `UniversalEmbedder(mode="auto")` 可以自动切换。

---

**作者**: Luke
**版本**: 1.0
**最后更新**: 2026-06-16
