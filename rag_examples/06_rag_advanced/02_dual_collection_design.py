# =============================================================================
# 02_dual_collection_design — 双集合 RAG 检索架构
# =============================================================================
# 用途：学习如何设计多集合的 RAG 检索架构
# 难度：⭐⭐⭐⭐（4 星）
#
# 核心概念：
#   1. 双集合设计：document_chunks（文档切片）+ qa_pairs（问答对）
#   2. 并行检索：同时对两个集合执行混合检索
#   3. 上下文合并：将不同来源的检索结果合并到 LLM 提示词中
#   4. 引用来源标记：让 LLM 标注回答的来源
#
# 这是 rag_demo/core/rag_query.py 的教学简化版。
# =============================================================================

import os
import random
from dotenv import load_dotenv
load_dotenv()

from rag_examples.milvus_config import MILVUS_URI, DEFAULT_DIMENSION
from pymilvus import (
    MilvusClient, DataType, Function, FunctionType,
    AnnSearchRequest,
)


# =============================================================================
# 第一部分：理解双集合架构
# =============================================================================

def explain_dual_collection():
    """讲解双集合 RAG 架构的设计思路"""
    print("=" * 60)
    print("第一部分：双集合 RAG 架构")
    print("=" * 60)

    print(f"""
┌─────────────────────────────────────────────────────────┐
│ 为什么需要双集合？                                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 单一集合（文档切片 only）的局限：                         │
│   - 只能检索到原始文本片段                               │
│   - 无法利用已有的结构化问答对                            │
│   - 缺失"推理过程"等半结构化信息                         │
│                                                         │
│ 双集合架构的优势：                                        │
│                                                         │
│   Collection 1: document_chunks（文档切片）               │
│   ├── text: 原始文档文本片段                             │
│   ├── file_name: 来源文件名                              │
│   └── chunk_index: 切片位置                              │
│                                                         │
│   Collection 2: qa_pairs（问答对）                       │
│   ├── question: 标准问题                                 │
│   ├── answer: 标准答案                                   │
│   └── reasoning: 推理过程（比单纯问答更有价值！）         │
│                                                         │
│   检索流程：                                              │
│                                                         │
│   用户问题                                               │
│      ├──→ document_chunks 混合检索 → 文档片段            │
│      └──→ qa_pairs 混合检索 → 相关问答对                 │
│                ↓                                         │
│         上下文合并 + LLM 生成                             │
│                ↓                                         │
│         带引用来源的答案                                  │
│                                                         │
└─────────────────────────────────────────────────────────┘

💡 设计原则：

1. 不同集合服务于不同的检索场景
   - 文档切片：提供"原始证据"
   - 问答对：提供"已整理的知识"

2. 并行检索提高召回率
   - 避免漏掉任何一种来源的相关信息

3. 来源标记便于溯源
   - 每个检索结果标记 source（"文档"或"问答对"）
   - LLM 回答时可以引用具体来源
""")


# =============================================================================
# 第二部分：创建双集合
# =============================================================================

def create_dual_collections():
    """创建两个 Collection：documents 和 qa_pairs"""
    client = MilvusClient(uri=MILVUS_URI)

    # ── Collection 1: documents ──
    doc_collection = "advanced_documents"

    if client.has_collection(doc_collection):
        client.drop_collection(doc_collection)

    schema = client.create_schema()
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=2000,
                       enable_analyzer=True, enable_match=True)
    schema.add_field(field_name="file_name", datatype=DataType.VARCHAR, max_length=200)
    schema.add_field(field_name="chunk_index", datatype=DataType.INT32)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=DEFAULT_DIMENSION)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    bm25_fn = Function(name="text_bm25", input_field_names=["text"],
                       output_field_names=["sparse_vector"], function_type=FunctionType.BM25)
    schema.add_function(bm25_fn)

    idx = client.prepare_index_params()
    idx.add_index(field_name="dense_vector", index_type="AUTOINDEX", metric_type="COSINE")
    idx.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")

    client.create_collection(collection_name=doc_collection, schema=schema, index_params=idx)
    print(f"✓ 文档切片集合 '{doc_collection}' 创建成功")

    # ── Collection 2: qa_pairs ──
    qa_collection = "advanced_qa_pairs"

    if client.has_collection(qa_collection):
        client.drop_collection(qa_collection)

    schema2 = client.create_schema()
    schema2.add_field(field_name="qa_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema2.add_field(field_name="question", datatype=DataType.VARCHAR, max_length=1000,
                        enable_analyzer=True, enable_match=True)
    schema2.add_field(field_name="answer", datatype=DataType.VARCHAR, max_length=2000)
    schema2.add_field(field_name="reasoning", datatype=DataType.VARCHAR, max_length=2000)
    schema2.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=DEFAULT_DIMENSION)
    schema2.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    bm25_fn2 = Function(name="question_bm25", input_field_names=["question"],
                        output_field_names=["sparse_vector"], function_type=FunctionType.BM25)
    schema2.add_function(bm25_fn2)

    idx2 = client.prepare_index_params()
    idx2.add_index(field_name="dense_vector", index_type="AUTOINDEX", metric_type="COSINE")
    idx2.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")

    client.create_collection(collection_name=qa_collection, schema=schema2, index_params=idx2)
    print(f"✓ 问答对集合 '{qa_collection}' 创建成功")

    return client, doc_collection, qa_collection


# =============================================================================
# 第三部分：插入数据
# =============================================================================

def embed(texts):
    """
    生成文本向量（使用阿里云百炼 Embedding API）

    参数:
        texts: 文本或文本列表
    返回:
        向量列表
    """
    from openai import OpenAI

    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIYUN_API_KEY")

    if isinstance(texts, str):
        texts = [texts]

    # 没有 API Key 时使用模拟向量（降级方案，演示模式）
    if not api_key:
        print("  未找到 API Key，使用模拟向量（演示模式）")
        import random
        random.seed(42)
        return [[random.random() for _ in range(1024)] for _ in texts]

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        # 批量处理（API 限制每次最多 25 条）
        all_embeddings = []
        batch_size = 25

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(
                model="text-embedding-v4",
                input=batch,
                encoding_format="float"
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        print(f"已生成 {len(all_embeddings)} 个 Embedding 向量")
        return all_embeddings

    except Exception as e:
        print(f"  Embedding API 调用失败：{e}，使用模拟向量")
        import random
        random.seed(42)
        return [[random.random() for _ in range(1024)] for _ in texts]

def insert_dual_collection_data(client, doc_collection, qa_collection):
    """向两个集合分别插入测试数据"""
    # 模拟向量生成
    def mock_embedding(text):
        random.seed(hash(text) % 10000)
        return [random.uniform(-1, 1) for _ in range(DEFAULT_DIMENSION)]

    # ── 插入文档切片 ──
    doc_chunks = [
        {"text": "Milvus 向量数据库支持混合检索，结合稠密向量和稀疏向量提高召回率。", "file_name": "milvus_intro.txt", "chunk_index": 0},
        {"text": "RAG 系统的核心是检索相关文档片段，然后让 LLM 基于这些片段生成答案。", "file_name": "rag_guide.txt", "chunk_index": 0},
        {"text": "BM25 算法基于 TF-IDF 改进，考虑了词频饱和度和文档长度归一化。", "file_name": "search_tech.txt", "chunk_index": 0},
    ]

    doc_data = []
    for chunk in doc_chunks:
        doc_data.append({
            "text": chunk["text"],
            "file_name": chunk["file_name"],
            "chunk_index": chunk["chunk_index"],
            "dense_vector": embed(chunk["text"])[0],
        })

    client.insert(collection_name=doc_collection, data=doc_data)
    print(f"✓ 向 '{doc_collection}' 插入了 {len(doc_data)} 条文档切片")

    # ── 插入问答对 ──
    qa_pairs = [
        {"question": "Milvus 支持哪些检索方式？", "answer": "Milvus 支持标量查询、稠密向量检索、稀疏向量（BM25）检索和混合检索。", "reasoning": "Milvus 2.4+ 引入了 BM25 Function 和多路 AnnSearchRequest，支持 RRF 和加权两种融合策略。"},
        {"question": "什么是 RAG？", "answer": "RAG（检索增强生成）是一种结合信息检索和文本生成的 AI 技术。", "reasoning": "RAG 先检索相关知识，再将检索结果作为上下文提供给大语言模型，从而减少幻觉、提高答案的准确性和时效性。"},
        {"question": "BM25 和向量检索有什么区别？", "answer": "BM25 基于关键词匹配，适合精确术语搜索；向量检索基于语义相似度，适合理解同义词和上下文。", "reasoning": "两者互补：BM25 精确匹配关键词（如产品型号），向量检索理解语义（如用户意图）。混合使用效果最佳。"},
    ]

    qa_data = []
    for qa in qa_pairs:
        qa_data.append({
            "question": qa["question"],
            "answer": qa["answer"],
            "reasoning": qa["reasoning"],
            "dense_vector": embed(qa["question"])[0],
        })

    client.insert(collection_name=qa_collection, data=qa_data)
    print(f"✓ 向 '{qa_collection}' 插入了 {len(qa_data)} 条问答对")


# =============================================================================
# 第四部分：双集合并行检索
# =============================================================================

def dual_collection_search(client, doc_collection, qa_collection, query: str):
    """同时对两个集合执行混合检索"""
    # 生成查询向量
    random.seed(hash(query) % 10000)
    query_vector = embed(query)[0]

    def hybrid_search(collection_name, output_fields, top_k=3):
        req_dense = AnnSearchRequest(
            data=[query_vector], anns_field="dense_vector",
            param={"nprobe": 10}, limit=top_k,
        )
        req_sparse = AnnSearchRequest(
            data=[query], anns_field="sparse_vector",
            param={"metric_type": "BM25"}, limit=top_k,
        )
        ranker = Function(
            name="rrf", function_type=FunctionType.RERANK,
            params={"reranker": "rrf", "k": 100},
            input_field_names=[],
        )
        return client.hybrid_search(
            collection_name=collection_name,
            reqs=[req_dense, req_sparse], ranker=ranker,
            limit=top_k,
            output_fields=output_fields
        )

    # 并行检索两个集合
    doc_results = hybrid_search(doc_collection, ["text", "file_name", "chunk_index"], top_k=3)
    qa_results = hybrid_search(qa_collection, ["question", "answer", "reasoning"], top_k=3)

    return doc_results, qa_results


# =============================================================================
# 第五部分：构建 RAG 提示词
# =============================================================================

def build_rag_prompt(query, doc_results, qa_results):
    """将双集合检索结果组装成 LLM 提示词"""
    context_parts = []

    # 文档片段
    for i, hits in enumerate(doc_results):
        for j, hit in enumerate(hits):
            entity = hit.get("entity", {})
            context_parts.append(
                f"[文档片段{j+1}] {entity.get('text', '')}\n"
                f"来源：{entity.get('file_name', '')}"
            )

    # 问答对
    for i, hits in enumerate(qa_results):
        for j, hit in enumerate(hits):
            entity = hit.get("entity", {})
            context_parts.append(
                f"[问答对{j+1}]\n"
                f"问：{entity.get('question', '')}\n"
                f"答：{entity.get('answer', '')}\n"
                f"推理：{entity.get('reasoning', '')}"
            )

    context = "\n\n".join(context_parts)

    system_prompt = (
        "你是一个基于检索增强生成（RAG）的智能问答助手。"
        "请根据以下检索到的上下文回答用户的问题。"
        "如果上下文中没有相关信息，请如实告知。"
    )

    prompt = f"{system_prompt}\n\n## 检索到的上下文\n{context}\n\n用户问题：{query}"
    return prompt


# =============================================================================
# 第六部分：总结
# =============================================================================

def dual_collection_summary():
    """双集合架构的关键要点"""
    print("\n" + "=" * 60)
    print("第六部分：双集合架构关键要点")
    print("=" * 60)

    print("""
┌─────────────────────────────────────────────────────────┐
│ 双集合 RAG 架构 — 关键要点                               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 1. 两个集合各有职责                                       │
│    - document_chunks: 原始文档片段（"证据"）              │
│    - qa_pairs: 结构化问答对（"知识卡片"）                 │
│                                                         │
│ 2. 每个集合都独立支持混合检索                             │
│    - 各自有稠密向量 + 稀疏向量字段                        │
│    - 各自有 BM25 Function                                │
│                                                         │
│ 3. 并行检索，合并上下文                                    │
│    - 不会因为串行等待而增加延迟                            │
│    - 不同来源用 Markdown 标题区分                         │
│                                                         │
│ 4. 引用来源可追溯                                         │
│    - 每个检索结果带上 source 和具体位置                   │
│    - 方便验证 LLM 答案的准确性                            │
│                                                         │
│ 5. QA 对中的 reasoning 字段很有价值                       │
│    - 不仅返回答案，还返回推理过程                         │
│    - 帮助 LLM 理解"为什么是这个答案"                     │
│                                                         │
└─────────────────────────────────────────────────────────┘

📖 下一步：查看 rag_demo/core/rag_query.py 了解生产级实现。
""")


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  双集合 RAG 检索架构")
    print("=" * 70 + "\n")

    # 1. 讲解架构
    # explain_dual_collection()

    # 2. 创建双集合
    client, doc_col, qa_col = create_dual_collections()

    # 3. 插入数据
    insert_dual_collection_data(client, doc_col, qa_col)

    # 4. 加载集合
    client.load_collection(doc_col)
    client.load_collection(qa_col)

    # 5. 并行检索演示
    print("\n" + "=" * 60)
    print("第四部分：双集合并行检索演示")
    print("=" * 60)

    queries = ["Milvus 支持什么检索方式？", "什么是 BM25？"]
    for query in queries:
        print(f"\n查询：{query}")
        doc_results, qa_results = dual_collection_search(client, doc_col, qa_col, query)

        print("  文档片段结果：")
        for hits in doc_results:
            for hit in hits:
                entity = hit.get("entity", {})
                print(f"    [{hit['distance']:.4f}] {entity.get('text', '')[:50]}...")

        print("  问答对结果：")
        for hits in qa_results:
            for hit in hits:
                entity = hit.get("entity", {})
                print(f"    [{hit['distance']:.4f}] Q: {entity.get('question', '')[:40]}...")

    # 6. 展示提示词构建
    doc_results, qa_results = dual_collection_search(client, doc_col, qa_col, "什么是 RAG？")
    prompt = build_rag_prompt("什么是 RAG？", doc_results, qa_results)
    print("\n" + "=" * 60)
    print("第五部分：生成的 RAG 提示词（前 300 字）")
    print("=" * 60)
    print(prompt[:300] + "...")

    # 7. 总结
    # dual_collection_summary()
