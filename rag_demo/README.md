# RAG 综合实战项目 — 基于 Milvus 的双集合混合检索问答系统

> 🎓 **课程定位**：综合实战项目（难度 ⭐⭐⭐⭐），建议学完 `rag_examples/` 全部模块后再学习本项目

---

## 📖 项目概述

`rag_demo` 是一个**生产级 RAG（检索增强生成）综合实战项目**，将 `rag_examples/` 课程中分散学习的知识点（Embedding、Milvus、文档切片、混合检索、RAG 管道）串联为一个完整的、可运行的智能问答系统。

### 核心特性

| 特性 | 说明 |
|------|------|
| **双集合架构** | `document_chunks`（文档切片）+ `qa_pairs`（问答对）两张表并行检索 |
| **混合检索** | 稠密向量（COSINE 语义匹配）+ 稀疏向量（BM25 关键词匹配），RRF 融合排序 |
| **BM25 原生支持** | 利用 Milvus 2.4+ `FunctionType.BM25` + `enable_analyzer` + `SPARSE_FLOAT_VECTOR` |
| **模块化设计** | `config.py` 统一配置、`util/` 共享工具包、`core/` 核心业务逻辑 |
| **真实 API** | 阿里云 DashScope `text-embedding-v4`（1024 维）+ DeepSeek LLM 生成答案 |
| **完整测试** | 40 个 pytest 测试用例（文本解析、切片、Embedding） |

### 实战语料

- 📚 **原始文档**：`datas/三国演义.txt`（1.75 MB 完整小说）
- ❓ **问答对**：`datas/qa_sanguo.json`（50 条三国知识 QA）+ `datas/qa_sanguo_additional.json`（3 条增量 QA）

---

## 🗺️ 与 rag_examples 的关系

```
rag_examples/（分散知识点 — 从零学习）        rag_demo/（综合实战 — 融会贯通）
─────────────────────────────────────        ─────────────────────────────
00_setup/          环境搭建            ──→    .env.example 环境配置
embedding_examples/  Embedding 基础     ──→    util/embedding.py 共享模块
01_milvus_basics/  Milvus 连接/集合     ──→    db/vdb_init_milvus.py Schema 设计
02_document_chunking/ 文档切片方法       ──→    util/text_splitter.py 递归切片
03_retrieval_methods/ 检索方法          ──→    core/file_chunk_retrieval.py 混合检索
04_rag_api/        API 封装思路         ──→    core/rag_query.py RAG 问答封装
05_rag_pipeline/   完整管道            ──→    整个项目的端到端流程
06_rag_evaluation/ RAG 评估            ──→    可用来评测本项目的检索/生成质量
06_rag_advanced/   Milvus 高级特性     ──→    BM25 Function、双集合、hybrid_search
```

> 💡 **建议学习路径**：先学完 `rag_examples/` 各模块（至少到 05），再回到本项目，你会发现"原来每个知识点是这样组合起来的"。

---

## 🚀 快速开始

### 1. 前置条件

- ✅ 已完成 `rag_examples/00_setup/` 的环境搭建
- ✅ Python 3.10+，已安装依赖（`rag_examples/requirements.txt`）
- ✅ Milvus 服务运行中（本地 Docker 或远程服务器）
- ✅ 已获取阿里云 DashScope API Key 和 DeepSeek API Key

### 2. 配置环境变量

```bash
# 在 rag_demo/ 目录下
cp .env.example .env

# 编辑 .env 文件，填入你的 API Key：
#   ALIYUN_API_KEY=你的阿里云API密钥
#   DEEPSEEK_API_KEY=你的DeepSeek API密钥
#   MILVUS_URI=http://localhost:19530
```

### 3. 初始化数据库

```bash
# 第一步：创建集合并插入数据（只需执行一次）
# 取消 vdb_init_milvus.py 中 __main__ 的注释，然后运行：
python db/vdb_init_milvus.py
```

### 4. 运行 RAG 问答

```bash
# 直接运行，使用默认测试问题
python core/rag_query.py

# 或在 Python 中调用
from rag_demo.core.rag_query import rag_ask
result = rag_ask("诸葛亮北伐失败的原因是什么？")
print(result["answer"])
```

### 5. 运行测试

```bash
cd rag_demo
pytest tests/ -v
```

---

## 📁 目录结构

```
rag_demo/
├── README.md                          # ← 本文件
├── .env.example                       # 环境变量模板
├── config.py                          # 共享配置中心（Milvus 客户端、集合名常量）
│
├── core/                              # 核心业务逻辑
│   ├── __init__.py                    # 导出 rag_ask 和 search_file_chunks
│   ├── rag_query.py                   # ★ RAG 问答入口（双集合混合检索 + LLM 生成）
│   └── file_chunk_retrieval.py        # 文档块混合检索（稠密 + BM25 + RRF）
│
├── db/                                # 数据库初始化
│   └── vdb_init_milvus.py             # ★ Milvus Schema 设计 + 数据摄入
│
├── util/                              # 共享工具模块
│   ├── __init__.py                    # 统一导出
│   ├── embedding.py                   # ★ 共享 Embedding 模块（text-embedding-v4, 1024 维）
│   ├── text_parser.py                 # TXT / PDF / DOCX 文件解析
│   └── text_splitter.py               # 基于 LangChain 的递归文本切片
│
├── tests/                             # 测试（40 个用例）
│   ├── conftest.py                    # pytest fixtures
│   ├── test_text_parser.py            # 文本解析测试（11 个）
│   ├── test_text_splitter.py          # 文本切片测试（18 个）
│   └── test_embedding.py              # Embedding 测试（11 个，含真实 API）
│
└── datas/                             # 数据文件
    ├── 三国演义.txt                    # 原始语料（1.75 MB）
    ├── qa_sanguo.json                  # 50 条三国 QA 对
    └── qa_sanguo_additional.json       # 3 条增量测试 QA
```

---

## 🔄 数据流架构

```
用户提问: "诸葛亮北伐失败的原因是什么？"
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  rag_ask(query)                                     │
│                                                     │
│  ┌─────────────────┐   ┌──────────────────┐        │
│  │ document_chunks  │   │   qa_pairs       │        │
│  │ (文档切片表)      │   │   (问答对表)      │        │
│  │                  │   │                  │        │
│  │ 稠密向量检索      │   │ 稠密向量检索      │        │
│  │   + BM25 检索    │   │   + BM25 检索    │        │
│  │   ↓ RRF 融合     │   │   ↓ RRF 融合     │        │
│  └────────┬─────────┘   └────────┬─────────┘        │
│           │                      │                   │
│           └──────────┬───────────┘                   │
│                      ▼                               │
│              合并检索结果 + 构建上下文                  │
│                      │                               │
│                      ▼                               │
│              DeepSeek LLM 生成答案                    │
└─────────────────────────────────────────────────────┘
    │
    ▼
  {
    "answer": "诸葛亮北伐失败的主要原因是...",
    "references": [ ... ]
  }
```

---

## 🧩 核心模块详解

### `config.py` — 共享配置中心

```python
from rag_demo.config import get_milvus_client, DOCUMENT_CHUNKS_COLLECTION, QA_PAIRS_COLLECTION

# 所有模块通过此文件获取配置，避免硬编码和重复
client = get_milvus_client()        # Milvus 客户端（从环境变量读取 URI）
collection = DOCUMENT_CHUNKS_COLLECTION  # "document_chunks"
```

### `util/embedding.py` — 共享 Embedding

```python
from rag_demo.util.embedding import generate_embedding, DEFAULT_EMBEDDING_DIMENSION

vec = generate_embedding("人工智能")  # → 1024 维浮点数列表
assert len(vec) == DEFAULT_EMBEDDING_DIMENSION  # 1024
```

### `util/text_splitter.py` — 文本切片

```python
from rag_demo.util.text_splitter import TextChunker

chunker = TextChunker(chunk_size=500, chunk_overlap=50)
chunks = chunker.split(text, metadata={"file_name": "三国演义.txt"})
# → [{"content": "...", "metadata": {"chunk_index": 0, ...}}, ...]
```

### `core/file_chunk_retrieval.py` — 文档检索

```python
from rag_demo.core.file_chunk_retrieval import search_file_chunks

results = search_file_chunks("桃园三结义是哪三个人？", top_k=5)
# → [{"text": "...", "file_name": "...", "chunk_index": 0, "score": 0.95}, ...]
```

### `core/rag_query.py` — RAG 问答

```python
from rag_demo.core.rag_query import rag_ask

result = rag_ask("诸葛亮北伐失败的原因是什么？", doc_top_k=3, qa_top_k=3)
print(result["answer"])       # DeepSeek 生成的回答
print(result["references"])   # 检索到的引用来源
```

---

## 🧪 测试

```bash
# 运行全部测试（40 个）
pytest rag_demo/tests/ -v

# 仅运行 mock 单元测试（不需要 API Key）
pytest rag_demo/tests/ -v -k "not Integration and not real"

# 仅运行真实 API 测试（需要 ALIYUN_API_KEY）
pytest rag_demo/tests/ -v -k "Integration"
```

| 测试文件 | 用例数 | 说明 |
|---------|--------|------|
| `test_text_parser.py` | 11 | TXT 编码/回退/空白处理、PDF/DOCX 导入验证 |
| `test_text_splitter.py` | 18 | 默认参数、自定义参数、元数据、边界条件、英文文本 |
| `test_embedding.py` | 11 | 8 个 mock 单元测试 + 3 个真实 API 语义验证 |

---

## 💡 设计决策

### 为什么用双集合（document_chunks + qa_pairs）？

- **文档切片**适合回答"XX 事件经过是怎样的"（需要检索原文段落）
- **问答对**适合回答"XX 是谁"（有标准答案的短问题）
- 两者互补，覆盖不同类型的用户提问

### 为什么用 RRF 而非加权排序？

- RRF（Reciprocal Rank Fusion）不需要调权重，开箱即用
- 对于稠密向量 + BM25 两种异构分数，RRF 比加权求和更稳定
- 加权排序的注释代码保留在 `rag_query.py` 中供学习对比

### 为什么用 text-embedding-v4（1024 维）？

- 阿里云 DashScope 提供，中文效果优秀
- 1024 维是默认输出维度，无需额外配置
- 与 `rag_examples/milvus_config.py` 的 `DEFAULT_DIMENSION` 保持一致

---

## 🔗 相关资源

- [rag_examples 主课程](../rag_examples/) — 前置知识学习
- [rag_examples CHEATSHEET](../rag_examples/CHEATSHEET.md) — 速查表
- [Milvus 官方文档](https://milvus.io/docs/zh/)
- [阿里云 DashScope Embedding](https://help.aliyun.com/zh/model-studio/text-embedding)
- [DeepSeek API 文档](https://platform.deepseek.com/api-docs/)

---

**作者**: Ric
**版本**: 1.0
**最后更新**: 2026-06-12
