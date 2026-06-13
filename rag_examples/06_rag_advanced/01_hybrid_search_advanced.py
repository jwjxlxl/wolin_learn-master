# =============================================================================
# 01_hybrid_search_advanced — Milvus 原生混合检索与 BM25 Function
# =============================================================================
# 用途：学习 Milvus 2.4+ 的高级混合检索 API
# 难度：⭐⭐⭐⭐（4 星）
#
# 核心概念：
#   1. BM25 Function — 让 Milvus 自动从文本生成稀疏向量
#   2. enable_analyzer + enable_match — 文本字段的搜索引擎能力
#   3. AnnSearchRequest — 多路检索请求
#   4. hybrid_search() — 原生混合检索 API
#   5. RRF vs 加权排序 — 两种融合策略对比
#
# 学完本节后，你将能理解 rag_demo 中的混合检索代码。
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
# 第一部分：理解 BM25 Function
# =============================================================================

def explain_bm25_function():
    """讲解 BM25 Function 的概念和用法

    Milvus 2.4+ 支持在 Schema 中直接定义 BM25 Function，
    自动从文本字段生成稀疏向量（BM25 向量），无需手动计算。

    需要两个开关：
    - enable_analyzer=True  → 对文本进行分词和分析
    - enable_match=True     → 启用文本匹配（BM25）功能
    """
    print("=" * 60)
    print("第一部分：理解 BM25 Function")
    print("=" * 60)

    print("""
┌─────────────────────────────────────────────────────────┐
│ BM25 Function — 让文本字段自带搜索引擎能力                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 在 Milvus 2.4 之前：                                    │
│   需要手动用 jieba + rank-bm25 计算稀疏向量再插入        │
│                                                         │
│ 在 Milvus 2.4+ 之后：                                   │
│   只需在 Schema 中定义 BM25 Function，Milvus 自动处理    │
│                                                         │
│ 两个关键开关（在 VARCHAR 字段上）：                       │
│                                                         │
│ 1. enable_analyzer=True                                 │
│    → 启用内置分词器，自动对中文/英文进行分词             │
│    → "我喜欢人工智能" → ["我","喜欢","人工智能"]          │
│                                                         │
│ 2. enable_match=True                                    │
│    → 启用 BM25 文本匹配功能                              │
│    → 自动计算 TF-IDF 并生成稀疏向量                      │
│                                                         │
│ 3. 定义 BM25 Function：                                 │
│    - input_field_names=["text"]  → 从哪个字段读取文本    │
│    - output_field_names=["sparse"] → 输出到哪个稀疏字段  │
│    - function_type=FunctionType.BM25                    │
│                                                         │
│ 4. 为稀疏向量字段创建索引：                              │
│    - index_type="SPARSE_INVERTED_INDEX"                 │
│    - metric_type="BM25"                                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
""")


# =============================================================================
# 第二部分：创建支持 BM25 的 Collection
# =============================================================================

def create_hybrid_search_collection():
    """创建支持混合检索的 Collection

    Schema 设计要点：
    1. text 字段：VARCHAR + enable_analyzer + enable_match → 支持 BM25
    2. dense_vector 字段：FLOAT_VECTOR → 稠密语义向量
    3. sparse_vector 字段：SPARSE_FLOAT_VECTOR → BM25 稀疏向量
    4. BM25 Function：text → sparse_vector
    5. 两个索引：dense_vector(AUTOINDEX) + sparse_vector(SPARSE_INVERTED_INDEX)
    """
    print("=" * 60)
    print("第二部分：创建支持 BM25 混合检索的 Collection")
    print("=" * 60)

    client = MilvusClient(uri=MILVUS_URI)
    collection_name = "hybrid_search_demo"

    # 建表前先删表（确保可重复运行）
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
        print(f"已删除旧集合：{collection_name}")

    # 定义 Schema
    schema = client.create_schema()

    # 主键
    schema.add_field(
        field_name="id", datatype=DataType.INT64,
        is_primary=True, auto_id=True,
    )

    # 文本字段（开启分词和匹配 → 支持 BM25）
    schema.add_field(
        field_name="text", datatype=DataType.VARCHAR,
        max_length=2000,
        enable_analyzer=True,   # ← 开关 1：启用分词
        enable_match=True,      # ← 开关 2：启用 BM25 匹配
    )

    # 标题字段
    schema.add_field(
        field_name="title", datatype=DataType.VARCHAR,
        max_length=256,
    )

    # 稠密向量字段（语义检索用）
    schema.add_field(
        field_name="dense_vector", datatype=DataType.FLOAT_VECTOR,
        dim=DEFAULT_DIMENSION,
    )

    # 稀疏向量字段（BM25 检索用，由 Function 自动填充）
    schema.add_field(
        field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR,
    )

    # 定义 BM25 Function：自动从 text 生成 sparse_vector
    bm25_function = Function(
        name="text_bm25",
        input_field_names=["text"],          # ← 输入：text 字段
        output_field_names=["sparse_vector"], # ← 输出：sparse_vector 字段
        function_type=FunctionType.BM25,
    )
    schema.add_function(bm25_function)

    # 创建索引
    index_params = client.prepare_index_params()

    # 稠密向量索引（AUTOINDEX 自动选择最优索引）
    index_params.add_index(
        field_name="dense_vector",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )

    # 稀疏向量索引（专用于 BM25）
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
    )

    # 创建 Collection
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )

    print(f"✓ 集合 '{collection_name}' 创建成功！")
    print(f"  - 稠密向量维度：{DEFAULT_DIMENSION}")
    print(f"  - BM25 Function：text → sparse_vector")
    print(f"  - 稠密索引：AUTOINDEX (COSINE)")
    print(f"  - 稀疏索引：SPARSE_INVERTED_INDEX (BM25)")

    return client, collection_name


# =============================================================================
# 第三部分：插入测试数据
# =============================================================================

def insert_demo_data(client, collection_name):
    """插入演示数据"""
    from openai import OpenAI

    # 测试文档
    documents = [
        {"title": "人工智能简介", "text": "人工智能（AI）是模拟人类智能的计算机科学。它涵盖了机器学习、深度学习、自然语言处理等多个子领域。"},
        {"title": "机器学习基础", "text": "机器学习是AI的重要分支，通过统计学方法让计算机从数据中自动学习规律和模式。"},
        {"title": "深度学习进阶", "text": "深度学习使用多层神经网络来模拟人脑的学习过程。在图像识别和语音处理中表现优异。"},
        {"title": "RAG技术详解", "text": "RAG（检索增强生成）结合了信息检索和文本生成技术。先检索相关知识，再由大模型基于检索结果生成答案。"},
        {"title": "Milvus向量数据库", "text": "Milvus是开源的向量数据库，专为大规模向量检索设计。支持亿级向量毫秒级查询，是RAG系统的核心组件。"},
        {"title": "自然语言处理", "text": "自然语言处理（NLP）是AI的重要应用领域。包括文本分类、情感分析、机器翻译、问答系统等任务。"},
        {"title": "BM25算法原理", "text": "BM25是一种经典的文本检索算法。它基于TF-IDF改进，考虑了词频饱和度和文档长度归一化。"},
        {"title": "向量检索vs关键词检索", "text": "向量检索基于语义相似度，能理解同义词和上下文。关键词检索基于精确匹配，适合专业术语和代码搜索。"},
    ]

    print("=" * 60)
    print("第三部分：插入测试数据")
    print("=" * 60)

    # 尝试用真实 Embedding API 生成稠密向量
    try:
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIYUN_API_KEY")
        if api_key:
            print("使用真实 Embedding API 生成稠密向量...")
            embedding_client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            texts = [doc["text"] for doc in documents]
            response = embedding_client.embeddings.create(
                model="text-embedding-v4",
                input=texts,
                encoding_format="float",
            )
            dense_vectors = [item.embedding for item in response.data]
            print(f"  ✓ 生成了 {len(dense_vectors)} 个 {len(dense_vectors[0])} 维向量")
        else:
            raise ValueError("未找到 API Key")
    except Exception as e:
        print(f"API 不可用（{e}），使用模拟向量演示...")
        random.seed(42)
        dense_vectors = [
            [random.random() for _ in range(DEFAULT_DIMENSION)]
            for _ in documents
        ]

    # 插入数据（注意：不需要手动提供 sparse_vector，BM25 Function 会自动生成）
    data = []
    for i, doc in enumerate(documents):
        data.append({
            "text": doc["text"],
            "title": doc["title"],
            "dense_vector": dense_vectors[i],
            # sparse_vector 由 BM25 Function 自动生成，无需提供
        })

    result = client.insert(collection_name=collection_name, data=data)
    client.flush(collection_name=collection_name)
    print(f"✓ 插入了 {result['insert_count']} 条数据（稀疏向量由 BM25 Function 自动生成）")

    return documents


# =============================================================================
# 第四部分：三种检索方式对比
# =============================================================================

def demo_pure_dense_search(client, collection_name):
    """纯稠密向量检索（语义检索）"""
    from openai import OpenAI

    query = "什么是向量数据库？"

    # 生成查询向量
    try:
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIYUN_API_KEY")
        embedding_client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        response = embedding_client.embeddings.create(
            model="text-embedding-v4", input=query, encoding_format="float",
        )
        query_vector = response.data[0].embedding
    except Exception:
        random.seed(hash(query) % 10000)
        query_vector = [random.random() for _ in range(DEFAULT_DIMENSION)]

    results = client.search(
        collection_name=collection_name,
        data=[query_vector],
        anns_field="dense_vector",
        limit=3,
        output_fields=["title", "text"],
    )

    print("\n🔵 纯稠密向量检索（语义）：")
    for i, hit in enumerate(results[0]):
        print(f"  {i+1}. [{hit['distance']:.4f}] {hit['entity']['title']}")


def demo_pure_sparse_search(client, collection_name):
    """纯 BM25 关键词检索"""
    query = "向量数据库"

    results = client.search(
        collection_name=collection_name,
        data=[query],
        anns_field="sparse_vector",
        limit=3,
        output_fields=["title", "text"],
        search_params={"params": {"metric_type": "BM25"}},
    )

    print("\n🟢 纯 BM25 关键词检索：")
    for i, hit in enumerate(results[0]):
        print(f"  {i+1}. [{hit['distance']:.4f}] {hit['entity']['title']}")


def demo_hybrid_search(client, collection_name):
    """混合检索：稠密 + 稀疏 + RRF 融合"""
    from openai import OpenAI

    query = "向量数据库有哪些特点？"

    # 1. 生成稠密查询向量
    try:
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIYUN_API_KEY")
        embedding_client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        response = embedding_client.embeddings.create(
            model="text-embedding-v4", input=query, encoding_format="float",
        )
        query_vector = response.data[0].embedding
    except Exception:
        random.seed(hash(query) % 10000)
        query_vector = [random.random() for _ in range(DEFAULT_DIMENSION)]

    # 2. 创建稠密检索请求
    req_dense = AnnSearchRequest(
        data=[query_vector],
        anns_field="dense_vector",
        param={"nprobe": 10},
        limit=5,
    )

    # 3. 创建稀疏检索请求（直接传入查询文本！BM25 Function 自动处理）
    req_sparse = AnnSearchRequest(
        data=[query],  # 注意：这里传的是原始文本，不是向量
        anns_field="sparse_vector",
        param={"metric_type": "BM25"},
        limit=5,
    )

    # 4. RRF 融合排序
    ranker = Function(
        name="rrf",
        input_field_names=[],  # RRF 不需要输入字段
        function_type=FunctionType.RERANK,
        params={"reranker": "rrf", "k": 100},
    )

    # 5. 执行混合检索
    results = client.hybrid_search(
        collection_name=collection_name,
        reqs=[req_dense, req_sparse],
        ranker=ranker,
        limit=5,
        output_fields=["title", "text"],
    )

    print("\n🟣 混合检索（稠密 + BM25 + RRF 融合）：")
    for i, hit in enumerate(results[0]):
        print(f"  {i+1}. [{hit['distance']:.4f}] {hit['entity']['title']}")


# =============================================================================
# 第五部分：RRF vs 加权排序
# =============================================================================

def explain_ranking_strategies():
    """对比 RRF 和加权排序两种融合策略"""
    print("\n" + "=" * 60)
    print("第五部分：RRF vs 加权排序")
    print("=" * 60)

    print("""
┌─────────────────────────────────────────────────────────┐
│ RRF（倒数排名融合）vs 加权排序                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ RRF（推荐默认使用）：                                    │
│   公式：score(d) = Σ 1/(k + rank_i(d))                  │
│   - 不关心原始分数的绝对值，只看排名                      │
│   - 不需要归一化                                         │
│   - 鲁棒性好，不受异常分数影响                            │
│   - k=100 时效果稳定（k 值越小，排名靠前的越重要）       │
│                                                         │
│ 加权排序（需要明确权重时使用）：                          │
│   公式：score(d) = w1*score1(d) + w2*score2(d)          │
│   - 需要 norm_score=True 归一化                          │
│   - 可以精确控制各路的权重占比                            │
│   - 适合明确知道"语义更重要"或"关键词更重要"的场景       │
│                                                         │
│ 使用建议：                                               │
│   - 不确定权重 → 用 RRF                                   │
│   - 明确语义更重要 → 加权 [0.7, 0.3]                     │
│   - 明确关键词更重要 → 加权 [0.3, 0.7]                   │
│   - 通用场景 → 加权 [0.6, 0.4] 或 RRF                   │
│                                                         │
└─────────────────────────────────────────────────────────┘

代码对比：

# RRF（更简单，更鲁棒）
ranker = Function(
    name="rrf", function_type=FunctionType.RERANK,
    params={"reranker": "rrf", "k": 100}
)

# 加权排序（更灵活，需调参）
ranker = Function(
    name="weighted", function_type=FunctionType.RERANK,
    params={
        "reranker": "weighted",
        "weights": [0.6, 0.4],    # 稠密:稀疏 = 6:4
        "norm_score": True         # 归一化两个分数到同一量纲
    }
)
""")


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  Milvus 高级检索 — 混合检索与 BM25 Function")
    print("=" * 70 + "\n")

    # 1. 讲解 BM25 Function
    explain_bm25_function()

    # 2. 创建 Collection
    client, collection_name = create_hybrid_search_collection()

    # 3. 插入测试数据
    insert_demo_data(client, collection_name)

    # 4. 加载集合（必须先加载才能检索）
    client.load_collection(collection_name)

    # 5. 三种检索方式对比
    print("\n" + "=" * 60)
    print("第四部分：三种检索方式对比")
    print("=" * 60)

    demo_pure_dense_search(client, collection_name)
    demo_pure_sparse_search(client, collection_name)
    demo_hybrid_search(client, collection_name)

    # 6. RRF vs 加权排序
    explain_ranking_strategies()

    print("\n" + "=" * 70)
    print("  学习完成！接下来查看：")
    print("  02_dual_collection_design.py — 双集合设计模式")
    print("  03_from_mock_to_real.py — 从模拟到真实的过渡")
    print("=" * 70)
