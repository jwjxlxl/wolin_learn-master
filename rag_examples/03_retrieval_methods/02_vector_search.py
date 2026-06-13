# =============================================================================
# 02_vector_search — 向量检索（Vector Search）
# =============================================================================
"""
本文件教学演示基于向量相似度的语义检索。

核心概念：
  - Embedding（文本向量化）
  - 相似度计算（余弦相似度、欧氏距离、内积）
  - 近似最近邻搜索（ANN）

相似度度量：COSINE（余弦）、L2（欧氏距离）、IP（内积）
适用场景：语义匹配、跨语言搜索、容错搜索、多模态搜索
"""

import random
from dotenv import load_dotenv
from pymilvus import MilvusClient
from rag_examples.milvus_config import MILVUS_URI, DEFAULT_DIMENSION

load_dotenv()


# =============================================================================
# 示例 1: 准备测试数据
# =============================================================================

def prepare_test_collection():
    """准备向量检索测试用的 Collection（含语义结构的模拟向量）"""
    random.seed(42)

    client = MilvusClient(uri=MILVUS_URI)
    collection_name = "vector_search_demo"

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        dimension=DEFAULT_DIMENSION,
        auto_id=True,
        metric_type="COSINE"
    )

    documents = [
        {"content": "人工智能是模拟人类智能的计算机科学领域，包括机器学习、深度学习等技术。", "category": "AI"},
        {"content": "机器学习通过训练数据让计算机自动学习规律，无需显式编程。", "category": "AI"},
        {"content": "深度学习使用多层神经网络，在图像识别和自然语言处理领域取得成功。", "category": "AI"},
        {"content": "产品设计需要考虑用户体验、功能完整性和技术可行性。", "category": "Product"},
        {"content": "好的产品设计应该简洁易用，解决用户的实际问题。", "category": "Product"},
        {"content": "Python 是一门流行的编程语言，适合数据科学和人工智能开发。", "category": "Programming"},
        {"content": "编程是将人类思维转化为计算机可执行指令的过程。", "category": "Programming"},
    ]

    def semantic_embedding(category, index):
        """根据类别生成有语义结构的模拟向量"""
        random.seed(category + str(index))
        base = [0.5 if i % 3 == 0 else -0.3 for i in range(DEFAULT_DIMENSION)]
        if category == "AI":
            base[0:100] = [0.8] * 100
        elif category == "Product":
            base[100:200] = [0.8] * 100
        elif category == "Programming":
            base[200:300] = [0.8] * 100
        vector = [b + random.uniform(-0.1, 0.1) for b in base]
        return vector

    data_to_insert = []
    for i, doc in enumerate(documents):
        data_to_insert.append({
            "content": doc["content"],
            "category": doc["category"],
            "vector": semantic_embedding(doc["category"], i)
        })

    client.insert(collection_name, data_to_insert)

    print(f"✓ 已创建测试集合：{collection_name}")
    print(f"  插入文档数：{len(documents)}")
    print(f"  文档类别：AI(3 条), Product(2 条), Programming(2 条)")

    return client, collection_name


# =============================================================================
# 示例 2: 基础向量检索
# =============================================================================

def basic_vector_search(client, collection_name):
    """演示两个不同语义方向的向量检索"""

    # 场景 1: 查询"机器学习是什么"
    print(f"\n-- 示例 2.1: 查询'机器学习是什么'（AI 相关）")
    random.seed(100)
    query_vector = [0.8 if i < 100 else random.uniform(-0.2, 0.2) for i in range(DEFAULT_DIMENSION)]

    results = client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=3,
        output_fields=["content", "category"]
    )

    print("  检索结果（Top-3）：")
    for i, hit in enumerate(results[0]):
        print(f"  [{i+1}] 相似度：{hit['distance']:.4f}")
        print(f"      类别：{hit['entity']['category']}")
        print(f"      内容：{hit['entity']['content'][:50]}...")
        print()

    # 场景 2: 查询"如何设计产品"
    print(f"\n-- 示例 2.2: 查询'如何设计产品'（Product 相关）")
    random.seed(200)
    query_vector = [0.8 if 100 <= i < 200 else random.uniform(-0.2, 0.2) for i in range(DEFAULT_DIMENSION)]

    results = client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=3,
        output_fields=["content", "category"]
    )

    print("  检索结果（Top-3）：")
    for i, hit in enumerate(results[0]):
        print(f"  [{i+1}] 相似度：{hit['distance']:.4f}")
        print(f"      类别：{hit['entity']['category']}")
        print(f"      内容：{hit['entity']['content'][:50]}...")
        print()


# =============================================================================
# 示例 3: 不同度量类型对比
# =============================================================================

def metric_type_comparison():
    """对比 COSINE、L2、IP 三种度量类型的适用场景"""
    print(f"\n-- 示例 3: 不同度量类型对比")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │ 度量类型对比                                            │
  ├──────────────┬────────────────┬─────────────────────────┤
  │    类型       │    计算方式     │         特点             │
  ├──────────────┼────────────────┼─────────────────────────┤
  │ COSINE       │ 余弦相似度      │ 值越大越相似 (范围 -1~1) │
  │              │ cos(θ)         │ 不受向量长度影响         │
  │              │                │ 推荐作为默认选择         │
  ├──────────────┼────────────────┼─────────────────────────┤
  │ L2           │ 欧几里得距离    │ 值越小越相似             │
  │ (EUCLIDEAN)  │ √(Σ(xi-yi)²)   │ 考虑向量绝对位置         │
  │              │                │ 适用于未归一化的向量     │
  ├──────────────┼────────────────┼─────────────────────────┤
  │ IP           │ 内积            │ 值越大越相似             │
  │ (INNER_PRODUCT)│ Σ(xi*yi)     │ 归一化后与 COSINE 等价   │
  │              │                │ 某些场景计算更快         │
  └──────────────┴────────────────┴─────────────────────────┘

  如何选择？

  1. COSINE（推荐）
     ✓ 适用于大多数语义检索场景
     ✓ Embedding 模型通常输出归一化向量
     ✓ 结果直观（-1 到 1 的相似度）

  2. L2
     ✓ 向量未归一化时使用
     ✓ 需要考虑向量绝对大小的场景

  3. IP
     ✓ 与 COSINE 在归一化后等价
     ✓ 追求极致性能时使用
""")


# =============================================================================
# 示例 4: 批量向量检索
# =============================================================================

def batch_vector_search(client, collection_name):
    """一次传入多个查询向量，批量检索"""
    print(f"\n-- 示例 4: 批量向量检索")

    random.seed(42)
    query_vectors = [
        [0.8 if i < 100 else random.uniform(-0.2, 0.2) for i in range(DEFAULT_DIMENSION)],
        [0.8 if 100 <= i < 200 else random.uniform(-0.2, 0.2) for i in range(DEFAULT_DIMENSION)],
        [0.8 if 200 <= i < 300 else random.uniform(-0.2, 0.2) for i in range(DEFAULT_DIMENSION)],
    ]

    print("  批量查询 3 个问题向量...\n")

    results = client.search(
        collection_name=collection_name,
        data=query_vectors,
        limit=2,
        output_fields=["category"]
    )

    categories = ["AI", "产品", "编程"]
    for i, result in enumerate(results):
        print(f"  【查询向量 {i+1}】（{categories[i]}相关）")
        for j, hit in enumerate(result):
            print(f"  [{j+1}] 相似度：{hit['distance']:.4f} | 类别：{hit['entity']['category']}")
        print()


# =============================================================================
# 示例 5: 向量检索参数详解
# =============================================================================

def search_params_explained(client, collection_name):
    """讲解向量检索的关键参数：limit、output_fields、filter、search_params"""
    print(f"\n-- 示例 5: 向量检索参数详解")
    print("""
  向量检索参数说明：

  1. limit (Top-K)
     含义：返回最相似的 K 个结果
     建议：5-20（太多会增加后续处理负担）

  2. output_fields
     含义：返回哪些字段的内容
     建议：只返回需要的字段，减少数据传输

  3. filter（标量过滤）
     含义：在向量检索基础上增加条件筛选
     示例："category == 'AI' and views > 1000"

  4. search_params（索引相关参数）
     IVF_FLAT 索引：{"nprobe": 10}
     HNSW 索引：{"ef": 64}
""")

    random.seed(99)
    query_vector = [random.random() for _ in range(DEFAULT_DIMENSION)]

    print("  实际检索演示：")
    results = client.search(
        collection_name=collection_name,
        data=[query_vector, query_vector],
        limit=5,
        output_fields=["content", "category"],
        search_params={"nprobe": 10}
    )

    print(f"  返回 {len(results[0])} 个结果：")
    for i, hit in enumerate(results[0]):
        print(f"  [{i+1}] 相似度：{hit['distance']:.4f} | 类别：{hit['entity']['category']}")


# =============================================================================
# 示例 6: 使用真实 Embedding 模型
# =============================================================================

def search_with_real_embedding():
    """展示实际项目中生成查询向量的三种方法"""
    print(f"\n-- 示例 6: 使用真实 Embedding 模型")
    print("""
  实际项目中如何生成查询向量？

  方法 1: sentence-transformers（本地模型）
  ────────────────────────────────────────
  from sentence_transformers import SentenceTransformer

  model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
  query = "什么是机器学习"
  query_vector = model.encode(query).tolist()

  方法 2: OpenAI Embedding API
  ────────────────────────────────────────
  from openai import OpenAI

  client = OpenAI(api_key="your-key")
  response = client.embeddings.create(
      model="text-embedding-3-small",
      input="什么是机器学习"
  )
  query_vector = response.data[0].embedding

  方法 3: 阿里云百炼 API
  ────────────────────────────────────────
  from dashscope import TextEmbedding

  result = TextEmbedding.call(
      model='text-embedding-v2',
      input='什么是机器学习'
  )
  query_vector = result.output['embeddings'][0]['embedding']
""")

    print("  模拟流程：")
    queries = [
        "人工智能和机器学习有什么关系？",
        "如何设计一个好的产品？",
        "Python 适合做什么开发？"
    ]

    for query in queries:
        print(f"  用户问题：{query}")
        print(f"  ↓ Embedding 模型")
        print(f"  输出：{DEFAULT_DIMENSION} 维向量 [0.023, -0.045, 0.089, ...]")
        print(f"  ↓ 向量检索")
        print(f"  返回：Top-3 最相似的文档")
        print()


# =============================================================================
# 示例 7: 向量检索最佳实践
# =============================================================================

def vector_search_best_practices():
    """向量检索的最佳实践总结"""
    print(f"\n-- 示例 7: 向量检索最佳实践")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │ 向量检索最佳实践                                        │
  ├─────────────────────────────────────────────────────────┤
  │                                                         │
  │ 1. Embedding 模型选择                                    │
  │    - 中文场景：bge-large-zh, m3e-base                   │
  │    - 多语言：paraphrase-multilingual-MiniLM             │
  │    - 通用：text-embedding-3-small (OpenAI)              │
  │                                                         │
  │ 2. 向量维度匹配                                          │
  │    - 确保查询向量维度与 Collection 定义一致              │
  │    - 常见维度：768 (bert), 1536 (OpenAI), """ + str(DEFAULT_DIMENSION) + """                                                     │
  │                                                         │
  │ 3. 相似度阈值                                            │
  │    - 建议设置最低相似度阈值过滤噪声                     │
  │    - COSINE 场景：threshold=0.5-0.7                      │
  │                                                         │
  │ 4. 结果数量选择                                          │
  │    - Top-K 建议：5-20                                   │
  │    - 太多：增加 LLM 上下文负担                          │
  │    - 太少：可能遗漏相关信息                             │
  │                                                         │
  │ 5. 性能优化                                              │
  │    - 数据量>1 万时建立索引（IVF_FLAT/HNSW）             │
  │    - 大批量查询时分批处理                               │
  │                                                         │
  └─────────────────────────────────────────────────────────┘

  完整的 RAG 检索流程：

  ```python
  def retrieve_context(query, top_k=5):
      query_vector = embedding_model.encode(query).tolist()
      results = client.search(
          collection_name="knowledge_base",
          data=[query_vector],
          limit=top_k,
          output_fields=["content", "title"]
      )
      contexts = []
      for hit in results[0]:
          if hit['distance'] > 0.5:
              contexts.append(hit['entity']['content'])
      return contexts
  ```
""")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  向量检索（Vector Search）")
    print("=" * 70 + "\n")

    client, collection_name = prepare_test_collection()
    basic_vector_search(client, collection_name)
    metric_type_comparison()
    batch_vector_search(client, collection_name)
    search_params_explained(client, collection_name)
    search_with_real_embedding()
    vector_search_best_practices()
