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

import os
import random
import time
from dotenv import load_dotenv
from pymilvus import MilvusClient
from rag_examples.milvus_config import MILVUS_URI, DEFAULT_DIMENSION

load_dotenv()

# Embedding 模式控制
# USE_REAL_EMBEDDING=true 时使用真实 Embedding 模型（需要 ALIYUN_API_KEY）
# 不设置或设为 false 时使用模拟向量（向后兼容，无需 API Key）
USE_REAL_EMBEDDING = os.getenv("USE_REAL_EMBEDDING", "false").lower() == "true"


# =============================================================================
# 示例 1: 准备测试数据
# =============================================================================

def _mock_embedding(text):
    """
    生成模拟向量（教学用，无需 API Key）。

    使用 hash(text) 作为 seed，确保相同文本得到相同向量。
    与原始 semantic_embedding 的机制一致，只是 seed 策略不同。
    """
    random.seed(hash(text) % 10000)
    base = [0.5 if i % 3 == 0 else -0.3 for i in range(DEFAULT_DIMENSION)]
    # 根据文本内容的前几个字符决定"语义段"位置
    seed_val = hash(text) % 700
    base[seed_val:seed_val + 100] = [0.8] * 100
    vector = [b + random.uniform(-0.1, 0.1) for b in base]
    return vector


def _real_embedding(text):
    """
    使用阿里云 text-embedding-v4 生成真实向量（生产用，1024 维）。

    需要设置环境变量 ALIYUN_API_KEY。
    """
    api_key = os.getenv("ALIYUN_API_KEY")
    if not api_key:
        raise ValueError("使用真实 Embedding 需要设置环境变量 ALIYUN_API_KEY。\n"
                         "请在 .env 文件中添加：ALIYUN_API_KEY=your_key\n"
                         "或设置 USE_REAL_EMBEDDING=false 使用模拟向量。")

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("需要安装 openai 库：pip install openai")

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    response = client.embeddings.create(
        model="text-embedding-v4",
        input=text
    )
    return response.data[0].embedding


def get_embedding(text):
    """
    获取文本向量（统一入口）。

    根据 USE_REAL_EMBEDDING 自动选择真实或模拟模式。

    参数:
        text: 需要向量化的文本

    返回:
        list[float]: DEFAULT_DIMENSION (1024) 维向量
    """
    if USE_REAL_EMBEDDING:
        return _real_embedding(text)
    return _mock_embedding(text)


def prepare_test_collection():
    """准备向量检索测试用的 Collection"""
    mode_name = "真实 Embedding" if USE_REAL_EMBEDDING else "模拟向量"

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

    if USE_REAL_EMBEDDING:
        print(f"  使用真实 Embedding 模型（text-embedding-v4）生成向量...")
        start = time.time()

    data_to_insert = []
    for i, doc in enumerate(documents):
        vector = get_embedding(doc["content"])
        data_to_insert.append({
            "content": doc["content"],
            "category": doc["category"],
            "vector": vector
        })

    if USE_REAL_EMBEDDING:
        elapsed = time.time() - start
        print(f"  ✓ 向量生成完成，耗时：{elapsed:.1f}s")

    client.insert(collection_name, data_to_insert)

    print(f"✓ 已创建测试集合：{collection_name}（{mode_name}）")
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
    query_text = "机器学习是什么"
    query_vector = get_embedding(query_text)

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
    query_text = "如何设计一款好用的产品"
    query_vector = get_embedding(query_text)

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

    query_texts = [
        "人工智能的核心技术有哪些？",
        "如何设计一款好用的产品？",
        "Python 编程语言的优势是什么？"
    ]

    if USE_REAL_EMBEDDING:
        start = time.time()

    query_vectors = [get_embedding(text) for text in query_texts]

    if USE_REAL_EMBEDDING:
        elapsed = time.time() - start
        print(f"  ✓ {len(query_texts)} 个向量生成完成，耗时：{elapsed:.1f}s\n")
    else:
        print(f"  批量查询 {len(query_texts)} 个文本向量...\n")

    results = client.search(
        collection_name=collection_name,
        data=query_vectors,
        limit=2,
        output_fields=["content", "category"]
    )

    categories = ["AI", "产品", "编程"]
    for i, result in enumerate(results):
        print(f"  【查询 {i+1}】{query_texts[i]}（{categories[i]}相关）")
        for j, hit in enumerate(result):
            print(f"  [{j+1}] 相似度：{hit['distance']:.4f} | 类别：{hit['entity']['category']} | {hit['entity']['content'][:40]}...")
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
    query_vector = get_embedding("向量检索有哪些关键参数")

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

def embedding_quality_comparison():
    """对比 mock 和 real embedding 的检索效果差异"""
    print(f"\n-- 示例 6: Mock vs Real Embedding 效果对比")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │ Mock 向量 vs 真实 Embedding 检索效果对比                 │
  ├─────────────────────────────────────────────────────────┤
  │                                                         │
  │ Mock 向量（教学用）：                                    │
  │   - 使用 hash(text) 生成确定性伪随机向量                 │
  │   - 特点：无需 API Key，同文本同向量                     │
  │   - 局限：向量没有真实语义信息                           │
  │                                                         │
  │ 真实 Embedding（生产用）：                                │
  │   - 使用 text-embedding-v4 模型（1024 维）               │
  │   - 特点：语义相似的文本向量距离近                       │
  │   - 需要：ALIYUN_API_KEY 环境变量                       │
  │                                                         │
  └─────────────────────────────────────────────────────────┘

  当前模式：""" + ("真实 Embedding" if USE_REAL_EMBEDDING else "模拟向量（USE_REAL_EMBEDDING=true 切换）") + """

  如何切换到真实 Embedding？
  ────────────────────────────────────────
  1. 在 .env 文件中添加：ALIYUN_API_KEY=your_key
  2. 添加：USE_REAL_EMBEDDING=true
  3. 重新运行本文件

  代码对比：
  ────────────────────────────────────────
  # Mock 模式（当前）
  def _mock_embedding(text):
      random.seed(hash(text) % 10000)
      return [random.uniform(-1, 1) for _ in range(1024)]

  # Real 模式（需要 API Key）
  from openai import OpenAI
  def _real_embedding(text):
      client = OpenAI(api_key=os.getenv("ALIYUN_API_KEY"),
          base_url="https://dashscope.aliyuncs.com/...")
      return client.embeddings.create(
          model="text-embedding-v4", input=text).data[0].embedding
""")


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
    embedding_quality_comparison()
    vector_search_best_practices()
