# 04_rag_api - RAG API 封装

本模块封装两个核心 API，用于 RAG 应用开发。

## API 列表

| API | 功能 | 难度 |
|------|------|------|
| [rag_retrieval_api.py](rag_retrieval_api.py) | 检索文件 API | ⭐⭐⭐ |
| [rag_qna_api.py](rag_qna_api.py) | RAG 问答 API | ⭐⭐⭐ |

## 检索文件 API

```python
# 用法示例
from rag_retrieval_api import RAGRetriever

retriever = RAGRetriever(
    milvus_uri="milvus_demo.db",
    embedding_model="bge-large-zh-v1.5"
)

# 检索相关文档
results = retriever.search(
    query="什么是机器学习？",
    top_k=5,
    use_rerank=True
)

for doc in results:
    print(f"相似度：{doc['score']}")
    print(f"内容：{doc['content']}")
```

## RAG 问答 API

```python
# 用法示例
from rag_qna_api import RAGQnA

qna = RAGQnA(
    milvus_uri="milvus_demo.db",
    llm_provider="ollama",
    embedding_model="bge-large-zh-v1.5"
)

# 问答
answer = qna.ask(
    question="机器学习需要什么基础？",
    top_k=5,
    use_rerank=True
)

print(f"答案：{answer['answer']}")
print(f"引用：{answer['sources']}")
```

## 运行前准备

```bash
pip install -r ../requirements.txt
```

## 学习建议

1. 先学习检索 API（更基础）
2. 再学习问答 API（在检索基础上增加 LLM 生成）
3. 理解整个 RAG 流程
