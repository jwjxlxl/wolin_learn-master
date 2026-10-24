# 04_rag_api — 已合并（本模块已移除）

> 本模块原有的两个 API 文件（`rag_retrieval_api.py`、`rag_qna_api.py`）已删除，
> 其内容分别并入「03 检索」与「05 RAG Pipeline」，本目录仅保留此指引。

## 原来想教什么

04 原本想做两件事：把「Milvus 检索」和「检索 + LLM 生成」各封装成一个类，方便复用。

这两件事现在都有了更规范、更完整的实现，所以 04 不再需要。

## 去哪了（对照表）

| 原来在 04 | 现在去哪学 |
|-----------|-----------|
| 建库 / 插入 / 检索的封装 | `utils/hybrid_retriever.py` 的 `HybridRetriever`（混合检索，比 04 的单向量更强） |
| 向量生成 | `embedding_examples/` 的 `AliyunEmbedding`（统一入口，不再手写 OpenAI 客户端） |
| 文档切片 | `02_document_chunking/` 模块 |
| Rerank 重排序 | `03_retrieval_methods/05_rerank.py` |
| 检索 → 组装 Prompt → 生成 → 引用 | `05_rag_pipeline/rag_hybrid_pipeline.py`（更完整：引用溯源 + 防注入） |

## 学习路径（合并后）

```
01_milvus_basics        → 会用 Milvus 建库、插数据
        ↓
02_document_chunking    → 学会把文档切成片段
        ↓
03_retrieval_methods    → 学会各种检索 + HybridRetriever + Rerank
        ↓
05_rag_pipeline         → 把检索接上 LLM，跑通完整 RAG（引用溯源 + 防注入）
        ↓
06_rag_advanced         → 双集合设计、Mock→Real 进阶
```

## 两个核心类的新用法

**原 `RAGRetriever`（检索）→ `HybridRetriever`**

```python
from pymilvus import MilvusClient
from rag_examples.milvus_config import MILVUS_URI, MILVUS_DB_NAME
from rag_examples.utils.hybrid_retriever import HybridRetriever
from rag_examples.embedding_examples.aliyun_embedding import AliyunEmbedding

client = MilvusClient(uri=MILVUS_URI, db_name=MILVUS_DB_NAME)
emb = AliyunEmbedding()
retriever = HybridRetriever(client, "demo", emb)

retriever.create_collection(dim=1024)
retriever.insert(documents, dense_vectors)
results = retriever.retrieve("机器学习需要什么基础？", top_k=3)  # → [{"index","text","score"}]
```

**原 `RAGQnA`（问答）→ `05_rag_pipeline/rag_hybrid_pipeline.py`**

```python
from rag_examples.milvus_config import MILVUS_URI, MILVUS_DB_NAME
from rag_examples.utils.hybrid_retriever import HybridRetriever
from rag_examples.embedding_examples.aliyun_embedding import AliyunEmbedding
from rag_examples.05_rag_pipeline.rag_hybrid_pipeline import rag_answer, print_answer_with_sources
from utils.model_utils import get_qwen_client

client = MilvusClient(uri=MILVUS_URI, db_name=MILVUS_DB_NAME)
retriever = HybridRetriever(client, "demo", AliyunEmbedding())

answer, contexts = rag_answer("什么是机器学习？", retriever, get_qwen_client())
print_answer_with_sources(answer, contexts)
```

完整演示直接运行 `python -m rag_examples.05_rag_pipeline.rag_hybrid_pipeline`。
