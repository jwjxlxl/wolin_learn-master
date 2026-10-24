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
# 复用：建集合 / 插入 / 混合检索统一走 utils/hybrid_retriever.py 的 HybridRetriever，
#       向量生成走 embedding_examples 的 AliyunEmbedding（不再手写 OpenAI 客户端）。
# 依赖：Milvus 服务运行中 + ALIYUN_API_KEY（Embedding）
# 这是 rag_demo/core/rag_query.py 的教学简化版。
# =============================================================================

import sys
from dotenv import load_dotenv
load_dotenv()

from pymilvus import MilvusClient, DataType
from rag_examples.milvus_config import MILVUS_URI, DEFAULT_DIMENSION
from rag_examples.utils.hybrid_retriever import HybridRetriever
from rag_examples.embedding_examples.aliyun_embedding import AliyunEmbedding

# Windows 控制台编码
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


# =============================================================================
# 第一部分：理解双集合架构
# =============================================================================

def explain_dual_collection():
    """讲解双集合 RAG 架构的设计思路"""
    print(f"\n-- 第一部分: 双集合 RAG 架构")

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
# 第二部分：创建双集合（复用 HybridRetriever）
# =============================================================================

def create_dual_collections():
    """创建两个 Collection，各自一个 HybridRetriever 实例。

    双集合与单集合的差别只在两点，交给 HybridRetriever 的两个扩展参数：
      - text_field：文本字段名（文档集合是 text，问答集合是 question）
      - extra_fields：附加标量字段（文档集合是 file_name/chunk_index，
                       问答集合是 answer/reasoning）
    """
    client = MilvusClient(uri=MILVUS_URI)
    emb = AliyunEmbedding()

    # ── Collection 1: documents ──
    doc_retriever = HybridRetriever(
        client, "advanced_documents", emb,
        text_field="text",
        extra_fields=[
            {"name": "file_name", "datatype": DataType.VARCHAR, "max_length": 200},
            {"name": "chunk_index", "datatype": DataType.INT32},
        ],
    )

    # ── Collection 2: qa_pairs ──
    qa_retriever = HybridRetriever(
        client, "advanced_qa_pairs", emb,
        text_field="question",
        extra_fields=[
            {"name": "answer", "datatype": DataType.VARCHAR, "max_length": 2000},
            {"name": "reasoning", "datatype": DataType.VARCHAR, "max_length": 2000},
        ],
    )

    # create_collection 内部：建 schema（dense+sparse+BM25 函数）→ 建索引 → 自动加载
    doc_retriever.create_collection(dim=DEFAULT_DIMENSION)
    qa_retriever.create_collection(dim=DEFAULT_DIMENSION)
    print(f"✓ 双集合创建成功：'advanced_documents' + 'advanced_qa_pairs'")

    return client, emb, doc_retriever, qa_retriever


# =============================================================================
# 第三部分：插入数据
# =============================================================================

def insert_dual_collection_data(emb, doc_retriever, qa_retriever):
    """向两个集合分别插入测试数据（复用 HybridRetriever.insert）"""
    # ── 插入文档切片 ──
    doc_chunks = [
        {"text": "Milvus 向量数据库支持混合检索，结合稠密向量和稀疏向量提高召回率。", "file_name": "milvus_intro.txt", "chunk_index": 0},
        {"text": "RAG 系统的核心是检索相关文档片段，然后让 LLM 基于这些片段生成答案。", "file_name": "rag_guide.txt", "chunk_index": 0},
        {"text": "BM25 算法基于 TF-IDF 改进，考虑了词频饱和度和文档长度归一化。", "file_name": "search_tech.txt", "chunk_index": 0},
    ]

    doc_retriever.insert(
        [c["text"] for c in doc_chunks],
        [emb.embed(c["text"]) for c in doc_chunks],
        extra_data=[
            {"file_name": c["file_name"], "chunk_index": c["chunk_index"]}
            for c in doc_chunks
        ],
    )
    print(f"✓ 向 'advanced_documents' 插入 {len(doc_chunks)} 条文档切片")

    # ── 插入问答对 ──
    qa_pairs = [
        {"question": "Milvus 支持哪些检索方式？", "answer": "Milvus 支持标量查询、稠密向量检索、稀疏向量（BM25）检索和混合检索。", "reasoning": "Milvus 2.4+ 引入了 BM25 Function 和多路 AnnSearchRequest，支持 RRF 和加权两种融合策略。"},
        {"question": "什么是 RAG？", "answer": "RAG（检索增强生成）是一种结合信息检索和文本生成的 AI 技术。", "reasoning": "RAG 先检索相关知识，再将检索结果作为上下文提供给大语言模型，从而减少幻觉、提高答案的准确性和时效性。"},
        {"question": "BM25 和向量检索有什么区别？", "answer": "BM25 基于关键词匹配，适合精确术语搜索；向量检索基于语义相似度，适合理解同义词和上下文。", "reasoning": "两者互补：BM25 精确匹配关键词（如产品型号），向量检索理解语义（如用户意图）。混合使用效果最佳。"},
    ]

    qa_retriever.insert(
        [q["question"] for q in qa_pairs],
        [emb.embed(q["question"]) for q in qa_pairs],
        extra_data=[
            {"answer": q["answer"], "reasoning": q["reasoning"]}
            for q in qa_pairs
        ],
    )
    print(f"✓ 向 'advanced_qa_pairs' 插入 {len(qa_pairs)} 条问答对")


# =============================================================================
# 第四部分：双集合并行检索
# =============================================================================

def dual_collection_search(emb, doc_retriever, qa_retriever, query):
    """同时对两个集合执行混合检索（复用 HybridRetriever.hybrid_search）。

    返回:
        (doc_results, qa_results)：各为一个命中列表，
        每项 {"distance": 分数, "entity": {字段名: 值}}
    """
    query_vec = emb.embed(query)

    doc_results = doc_retriever.hybrid_search(query, query_vec, top_k=3)
    qa_results = qa_retriever.hybrid_search(query, query_vec, top_k=3)

    return doc_results, qa_results


# =============================================================================
# 第五部分：构建 RAG 提示词
# =============================================================================

def build_rag_prompt(query, doc_results, qa_results):
    """将双集合检索结果组装成 LLM 提示词"""
    context_parts = []

    # 文档片段
    for j, hit in enumerate(doc_results, 1):
        entity = hit.get("entity", {})
        context_parts.append(
            f"[文档片段{j}] {entity.get('text', '')}\n"
            f"来源：{entity.get('file_name', '')}"
        )

    # 问答对
    for j, hit in enumerate(qa_results, 1):
        entity = hit.get("entity", {})
        context_parts.append(
            f"[问答对{j}]\n"
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

    return f"{system_prompt}\n\n## 检索到的上下文\n{context}\n\n用户问题：{query}"


# =============================================================================
# 第六部分：总结
# =============================================================================

def dual_collection_summary():
    """双集合架构的关键要点"""
    print(f"\n-- 第六部分: 双集合架构关键要点")

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

    # 1. 讲解架构（纯文字，无依赖）
    # explain_dual_collection()

    # 2. 创建双集合
    client, emb, doc_retriever, qa_retriever = create_dual_collections()

    # 3. 插入数据
    insert_dual_collection_data(emb, doc_retriever, qa_retriever)

    # 4. 并行检索演示
    print(f"\n-- 第四部分: 双集合并行检索演示")
    query = "什么是 RAG？"
    doc_results, qa_results = dual_collection_search(emb, doc_retriever, qa_retriever, query)

    print(f"\n查询：{query}")
    print("\n  文档片段结果：")
    for hit in doc_results:
        entity = hit.get("entity", {})
        print(f"    [{hit['distance']:.4f}] {entity.get('text', '')[:50]}...")

    print("\n  问答对结果：")
    for hit in qa_results:
        entity = hit.get("entity", {})
        print(f"    [{hit['distance']:.4f}] Q: {entity.get('question', '')[:40]}...")

    # 5. 展示提示词构建
    prompt = build_rag_prompt(query, doc_results, qa_results)
    print(f"\n-- 第五部分: 生成的 RAG 提示词（前 500 字）")
    print(prompt[:500] + "...")

    # 6. 总结
    # dual_collection_summary()
