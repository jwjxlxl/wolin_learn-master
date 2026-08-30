# =============================================================================
# 03_keyword_search — 关键字检索（Keyword Search / 全文检索）
# =============================================================================
"""
本文件教学演示基于关键词的传统检索方法。
如果不是在Milvus中使用全文检索，可以考虑使用 Elasticsearch 或其他全文检索工具。
需要对原始文本进行分词处理，生成稀疏向量。
需要对检索的问题也要进行分词，得到相同维度的稀疏向量，才能进行相似度计算。
如果使用Milvus进行全文检索，自动把文本转换为稀疏向量，检索的时候也不需要分词处理



核心概念：
  - 全文检索：直接输入原始文本，系统自动按 BM25 算法排序
  - BM25 算法：TF（词频）+ IDF（逆文档频率）+ 文档长度归一化
  - 倒排索引：类似书籍索引，快速定位包含某词的文档

Milvus 全文检索工作流程：
  1. 创建 Collection → 设置 VARCHAR/TEXT 字段 + BM25 函数
  2. 插入数据 → 提供原始文本，Milvus 自动生成稀疏向量
  3. 执行搜索 → 用自然语言查询，返回按相关性排序的结果

对比维度：关键字检索（精确匹配、可解释）vs 向量检索（语义理解）
适用场景：精确匹配（专有名词、人名、品牌名、代码/技术术语）
"""

import os
import sys
import math
import jieba
from dotenv import load_dotenv
from rag_examples.milvus_config import MILVUS_URI, MILVUS_DB_NAME
from pymilvus import MilvusClient

# Windows 控制台编码
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()


# =============================================================================
# 示例 : Milvus 全文检索
# =============================================================================

client = MilvusClient(uri="http://192.168.142.128:19530")
def create_fulltext_collection(collection_name="keyword_demo"):
    """
    创建带 BM25 函数的全文检索 Collection。

    返回:
        MilvusClient: 已配置好的客户端，可直接调用 search_fulltext()
    """
    from pymilvus import DataType, Function, FunctionType

    client.use_database("ai0626")
    # 定义 schema
    schema = client.create_schema()
    schema.add_field(
        field_name="id",
        datatype=DataType.INT64,
        is_primary=True,
        auto_id=True,
    )
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=2000,
        enable_analyzer=True,  # 开启分词器（中文分词必需）
    )
    # 添加一个稀疏向量字段
    schema.add_field(
        field_name="sparse",
        datatype=DataType.SPARSE_FLOAT_VECTOR,
    )

    # 添加 BM25 函数：text → sparse（自动生成稀疏向量）
    bm25_function = Function(
        name="text_bm25_emb",
        input_field_names=["text"],
        output_field_names=["sparse"],
        function_type=FunctionType.BM25,
    )
    schema.add_function(bm25_function)

    # 创建索引
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="sparse",
        index_type="SPARSE_INVERTED_INDEX",
        # 稀疏向量字段的度量类型必须是BM25
        metric_type="BM25",
        params={
            "inverted_index_algo": "DAAT_MAXSCORE",
            "bm25_k1": 1.2,
            "bm25_b": 0.75,
        },
    )

    # 清理旧 Collection（教学演示用）
    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )

    return client


def insert_fulltext_texts(client, texts, collection_name="keyword_demo"):
    """
    向全文检索 Collection 中插入文本数据。

    参数:
        client: MilvusClient 实例
        texts: 文本列表（原始中文）
        collection_name: Collection 名称
    """
    client.insert(collection_name=collection_name, data=texts)


def search_fulltext(query, collection_name="keyword_demo", limit=3):
    """
    对全文检索 Collection 执行搜索。

    参数:
        query: 用户查询文本（原始中文）
        collection_name: Collection 名称
        limit: 返回结果数量

    返回:
        list: 搜索结果列表，每项包含 distance 和 entity.text
    """

    res = client.search(
        collection_name=collection_name,
        data=[query],
        anns_field="sparse",
        output_fields=["text"],
        limit=limit,
    )

    return res[0]


def milvus_fulltext_search():
    """
    使用 Milvus 内置全文检索功能（教学演示）。

    Milvus 全文检索三步走：
    1. 创建 Collection（带 BM25 函数）
    2. 插入原始文本数据（Milvus 自动处理分词和向量化）
    3. 执行搜索（自动按 BM25 排序）
    """
    print(f"\n-- 示例 2: Milvus 全文检索")

    # ---- 第 1 步：创建 Collection ----
    print("\n  第 1 步：创建 Collection（配置 BM25 函数）")
    client = create_fulltext_collection()
    if client is None:
        return
    print(f"  ✓ Collection 'keyword_demo' 创建成功")

    # ---- 第 2 步：插入文本数据 ----
    print("\n  第 2 步：插入原始文本数据")

    texts = [
        {'text': 'information retrieval is a field of study.'},
        {'text': 'information retrieval focuses on finding relevant information in large datasets.'},
        {'text': 'data mining and information retrieval overlap in research.'}
    ]

    insert_fulltext_texts(client, texts)
    print(f"  ✓ 插入 {len(texts)} 篇文档（Milvus 自动生成 BM25 稀疏向量）")


# =============================================================================
# 示例 3: BM25 算法原理（简化实现，帮助理解）
# =============================================================================

def demo_bm25_algorithm():
    """
    简化版 BM25 算法实现（教学用）。

    BM25 评分 = Σ [ IDF(qi) × TF(qi, D) / (k1 × (1 - b + b × |D|/avgLen) + TF(qi, D)) ]

    三个核心要素：
      - TF（词频）：词在文档中出现越多越重要，但有边际递减
      - IDF（逆文档频率）：在越少文档中出现的词越重要
      - 长度归一化：长文档词频天然高，需要"打折"
    """
    print(f"\n-- 示例 3: BM25 算法原理")

    documents = [
        "机器学习是人工智能的核心技术。",
        "深度学习使用机器学习的方法训练神经网络模型。",
        "自然语言处理是人工智能的重要应用方向。",
        "机器学习在医疗和金融领域有广泛应用。",
    ]

    print("  文档库：")
    for i, doc in enumerate(documents):
        print(f"  [{i+1}] {doc}")

    query = "机器学习"
    print(f"\n  查询：'{query}'")

    # 中文分词
    tokenized_docs = [list(jieba.cut(doc)) for doc in documents]

    # 计算文档频率（DF）
    df = {}
    for tokens in tokenized_docs:
        for token in set(tokens):
            df[token] = df.get(token, 0) + 1

    N = len(documents)
    avg_len = sum(len(t) for t in tokenized_docs) / N

    # BM25 评分
    k1, b = 1.5, 0.75
    query_tokens = list(jieba.cut(query))

    print(f"\n  BM25 评分过程：")
    scores = []
    for i, tokens in enumerate(tokenized_docs):
        score = 0
        doc_len = len(tokens)
        for term in query_tokens:
            if term not in df:
                continue
            # IDF = log((N - df + 0.5) / (df + 0.5) + 1)
            idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
            # TF 评分
            freq = tokens.count(term)
            numerator = freq * (k1 + 1)
            denominator = freq + k1 * (1 - b + b * doc_len / avg_len)
            tf_score = numerator / denominator
            score += idf * tf_score
        scores.append((i + 1, score, documents[i]))
        print(f"  文档{i+1}: IDF × TF = {score:.4f}")

    # 排序
    scores.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  排序结果：")
    for num, score, doc in scores:
        print(f"  [{num}] {score:.4f}  {doc}")

# =============================================================================
# 示例 5: 最佳实践
# =============================================================================

def keyword_search_best_practices():
    """关键字检索的最佳实践总结"""
    print(f"\n-- 示例 5: 最佳实践")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │ Milvus 全文检索要点                                     │
  ├─────────────────────────────────────────────────────────┤
  │                                                         │
  │ 1. 字段配置                                             │
  │    - 文本字段：VARCHAR（短文本）或 TEXT（长文本）       │
  │    - 必须开启：enable_analyzer=True（中文分词）         │
  │    - 必须添加：BM25 Function（text → sparse）           │
  │                                                         │
  │ 2. 索引参数                                             │
  │    - index_type: SPARSE_INVERTED_INDEX                  │
  │    - metric_type: BM25                                  │
  │    - bm25_k1: 1.2-2.0（词频饱和度，默认 1.2）           │
  │    - bm25_b: 0.5-0.8（长度归一化，默认 0.75）           │
  │                                                         │
  │ 3. 搜索调用                                             │
  │    - data: 直接传原始查询文本                           │
  │    - anns_field: BM25 稀疏向量字段名                    │
  │    - output_fields: 不能包含 sparse 向量字段            │
  │                                                         │
  │ 4. 注意事项                                             │
  │    - 无法直接读取 BM25 生成的稀疏向量                   │
  │    - 每个文本字段需定义独立的 BM25 函数                 │
  │    - 中文分词依赖 Milvus 内置 analyzer                  │
  │                                                         │
  └─────────────────────────────────────────────────────────┘

  混合检索策略（推荐）：

  ```python
  # 关键字检索 + 向量检索 = 混合检索
  keyword_results = client.search(
      collection_name="my_collection",
      data=[query],
      anns_field="sparse",     # BM25 稀疏向量
      limit=10,
  )

  vector_results = client.search(
      collection_name="my_collection",
      data=[embedding],
      anns_field="dense",      # 语义向量
      limit=10,
  )

  # 使用 RRF（倒数排名融合）合并结果
  final = reciprocal_rank_fusion(keyword_results, vector_results)
  ```
""")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    # milvus_fulltext_search()   # 需要 Milvus 服务运行中
    results = search_fulltext("whats the study", limit=3)
    for hit in results:
        print(hit["distance"], hit["entity"]["text"])

    # demo_bm25_algorithm()
    # keyword_vs_vector_search()
    # keyword_search_best_practices()
