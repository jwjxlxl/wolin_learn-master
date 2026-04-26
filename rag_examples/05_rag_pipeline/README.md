# 05_rag_pipeline - RAG 完整流程

本模块演示完整的 RAG 流程闭环，将之前学的所有知识点串联起来。

## 完整流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAG 完整流程                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  【知识库构建阶段】                                                   │
│                                                                     │
│       ▼                                                             │
│   ┌─────────┐                                                       │
│   │ 1. 文档加载 │  读取 TXT/PDF/Word 等格式文档                       │
│   └─────┬───┘                                                       │
│         │                                                           │
│         ▼                                                           │
│   ┌─────────┐                                                       │
│   │ 2. 文档切片 │  滑动窗口/Fixed/AI 切片                             │
│   └─────┬───┘                                                       │
│         │                                                           │
│         ▼                                                           │
│   ┌─────────┐                                                       │
│   │ 3. Embedding│  调用模型生成向量（关键步骤）                      │
│   └─────┬───┘                                                       │
│         │                                                           │
│         ▼                                                           │
│   ┌─────────┐                                                       │
│   │ 4. 存入 Milvus│  创建 Collection+Index，插入向量                 │
│   └─────┬───┘                                                       │
│                                                                     │
│  【检索问答阶段】                                                     │
│                                                                     │
│         ▼                                                           │
│   ┌─────────┐                                                       │
│   │ 5. 用户提问 │  输入问题                                          │
│   └─────┬───┘                                                       │
│         │                                                           │
│         ▼                                                           │
│   ┌─────────┐                                                       │
│   │ 6. 问题 Embedding│  同样模型生成查询向量                         │
│   └─────┬───┘                                                       │
│         │                                                           │
│         ▼                                                           │
│   ┌─────────┐                                                       │
│   │ 7. 向量检索 │  Milvus 检索 Top-K 相似文档                         │
│   └─────┬───┘                                                       │
│         │                                                           │
│         ▼                                                           │
│   ┌─────────┐                                                       │
│   │ 8. Rerank │  (可选) 重排序提升精度                               │
│   └─────┬───┘                                                       │
│         │                                                           │
│         ▼                                                           │
│   ┌─────────┐                                                       │
│   │ 9. LLM 生成  │  基于检索结果生成答案                             │
│   └─────┬───┘                                                       │
│         │                                                           │
│         ▼                                                           │
│      输出答案 + 引用来源                                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 课程文件

| 文件 | 内容 | 难度 |
|------|------|------|
| [rag_full_pipeline.py](rag_full_pipeline.py) | 完整流程封装 | ⭐⭐⭐ |
| [rag_step_by_step.py](rag_step_by_step.py) | 分步详解版 | ⭐⭐⭐⭐ |
| [rag_minimal.py](rag_minimal.py) | 最小可运行版 | ⭐⭐ |

## 快速开始

```python
from rag_full_pipeline import RAGPipeline

# 1. 初始化
pipeline = RAGPipeline(
    milvus_uri="milvus_demo.db",
    embedding_model="local",  # 或 "aliyun"
    dim=1024
)

# 2. 构建知识库
pipeline.run_full_pipeline(
    file_paths=["doc1.txt", "doc2.txt"],
    chunk_size=500,
    chunk_overlap=50
)

# 3. 问答
result = pipeline.ask("你的问题？", top_k=5)
print(result['answer'])
```

## 核心代码逻辑

### Embedding 在流程中的位置

```python
# 知识库构建阶段
for chunk in chunks:
    # ↓↓↓ Embedding 向量化 ↓↓↓
    vector = embedding_model.encode(chunk['text'])

    # 存入 Milvus
    client.insert(collection_name, {
        "content": chunk['text'],
        "embedding": vector  # ← 向量存入这里
    })

# 检索问答阶段
# ↓↓↓ 同样的 Embedding 模型 ↓↓↓
query_vector = embedding_model.encode(question)

# 向量检索
results = client.search(
    collection_name,
    data=[query_vector],  # ← 用查询向量检索
    limit=top_k
)
```

### 关键点

1. **Embedding 模型必须一致**
   - 文档向量化用什么模型，查询也要用同一模型
   - 否则向量不在同一空间，相似度计算无意义

2. **向量维度要匹配**
   - bge-large-zh-v1.5 → dim=1024
   - bge-base-zh-v1.5 → dim=768
   - text-embedding-v3 → dim=1024/1536

3. **切片大小影响效果**
   - 太小：语义不完整
   - 太大：包含过多噪声
   - 建议：300-500 字符，重叠 50-100

## 运行前准备

```bash
pip install -r ../requirements.txt
```

## 学习建议

1. 先运行 `rag_minimal.py`（最小可运行版，快速建立信心）
2. 再运行 `rag_full_pipeline.py`（完整版，学习完整流程）
3. 最后看 `rag_step_by_step.py`（分步详解，理解每个细节）
