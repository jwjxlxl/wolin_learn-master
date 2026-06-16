# =============================================================================
# 03_insert_data — 插入数据到 Collection
# =============================================================================
# 用途：学习如何向 Milvus Collection 插入数据
# 难度：⭐⭐（2 星）
#
# 核心概念：
#   - 单条插入 vs 批量插入
#   - 向量数据 + 标量数据同时插入
#   - 自增 ID vs 手动指定 ID
#   - Embedding 模型生成向量
# =============================================================================

import os
import random
from dotenv import load_dotenv
load_dotenv()

from rag_examples.milvus_config import MILVUS_URI, DEFAULT_DIMENSION


def explain_embedding():
    """理解什么是 Embedding（嵌入）

    Embedding 是将文本、图像、音频等数据转换为向量的过程。
    相似的内容 → 向量在空间中距离相近。
    """
    print("=" * 60)
    print("示例 1: 理解 Embedding（嵌入）")
    print("=" * 60)

    print(f"""
┌─────────────────────────────────────────────────────────┐
│ 什么是 Embedding（嵌入）？                               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 📊 定义                                                  │
│    Embedding = 将非结构化数据转换为向量（数字列表）     │
│    相似的内容 → 向量在空间中距离相近                    │
│                                                         │
│ 💡 生活化比喻                                            │
│    Embedding = "给数据分配坐标"                          │
│                                                         │
│    图书馆里，相似主题的书放在一起：                       │
│    - AI 书籍 → 坐标 [0.8, 0.2, 0.9, ...]                 │
│    - 烹饪书籍 → 坐标 [0.1, 0.9, 0.3, ...]                │
│    - 历史书籍 → 坐标 [0.3, 0.7, 0.2, ...]                │
│                                                         │
│ 📦 Embedding 模型类型                                    │
│    1. 文本 Embedding 模型                                │
│       - text-embedding-v4 (阿里云百炼，{DEFAULT_DIMENSION} 维)              │
│       - text-embedding-v3 (阿里云百炼)                  │
│       - BGE-M3 (开源)                                   │
│       - m3e-base (开源，中文优化)                        │
│                                                         │
│    2. 图像 Embedding 模型                                │
│       - ResNet50                                        │
│       - ViT (Vision Transformer)                        │
│       - CLIP (图文双模态)                               │
│                                                         │
│ 🔑 为什么需要 Embedding？                                │
│    - 计算机只能理解数字                                  │
│    - 向量空间中的距离 = 语义相似度                       │
│    - 可以用数学方法计算"相似"                            │
│                                                         │
└─────────────────────────────────────────────────────────┘

💡 Embedding 示例（文本 → 向量）:

   "我喜欢猫" → [0.12, -0.34, 0.56, 0.78, ...]  ({DEFAULT_DIMENSION} 维)
   "我讨厌狗" → [0.10, -0.32, 0.54, 0.76, ...]  ({DEFAULT_DIMENSION} 维)
                      ↑
              向量非常接近！

   "今天天气真好" → [0.89, 0.45, -0.12, 0.33, ...] ({DEFAULT_DIMENSION} 维)
                      ↑
              向量距离较远
""")


def generate_embedding_with_llm(texts, model="text-embedding-v4", api_key=None):
    """使用阿里云百炼 DashScope API 生成真实的 Embedding 向量

    Args:
        texts: 文本列表或单个文本
        model: Embedding 模型名称
        api_key: API Key，默认从环境变量读取

    Returns:
        向量列表（每个向量是浮点数列表）
    """
    from openai import OpenAI

    if api_key is None:
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIYUN_API_KEY")
        if not api_key:
            raise ValueError("未找到 API Key，请设置环境变量 DASHSCOPE_API_KEY 或 ALIYUN_API_KEY")
    print("API KEY：", api_key)

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    if isinstance(texts, str):
        texts = [texts]
        single_input = True
    else:
        single_input = False

    print(f"调用 Embedding 模型：{model}")
    print(f"输入文本数量：{len(texts)}")

    try:
        all_embeddings = []
        batch_size = 25

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(
                model=model,
                input=batch,
                encoding_format="float",
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            print(f"  已处理批次 {i//batch_size + 1}: {len(batch_embeddings)} 条")

        if all_embeddings:
            print(f"生成向量维度：{len(all_embeddings[0])} 维")

        return all_embeddings[0] if single_input else all_embeddings

    except Exception as e:
        print(f"Embedding API 调用失败：{e}")
        print("退化为模拟 Embedding 生成...")
        return generate_mock_embeddings(texts if not single_input else texts[0])


def generate_mock_embeddings(texts, dim=DEFAULT_DIMENSION):
    """模拟 Embedding 生成的向量数据

    ⚠️ 仅用于教学演示和 API 不可用时的降级方案。
    实际项目必须使用真实的 Embedding 模型！

    Args:
        texts: 文本列表或单个文本
        dim: 向量维度

    Returns:
        向量列表
    """
    random.seed(42)

    if isinstance(texts, str):
        texts = [texts]

    vectors = []
    for text in texts:
        vector = [random.random() for _ in range(dim)]
        vectors.append(vector)

    return vectors[0] if len(texts) == 1 else vectors


def prepare_test_data():
    """准备测试用的文档数据

    Returns:
        文档列表，包含 content, title, category, views
    """
    documents = [
        {
            "content": "人工智能（AI）是模拟人类智能的计算机科学领域，包括机器学习、深度学习、自然语言处理等技术。",
            "title": "人工智能简介",
            "category": "AI",
            "views": 1000,
        },
        {
            "content": "机器学习是人工智能的分支，通过训练数据让计算机自动学习规律，无需显式编程。",
            "title": "机器学习基础",
            "category": "AI",
            "views": 800,
        },
        {
            "content": "深度学习使用多层神经网络模拟人脑，在图像识别、语音识别等领域取得突破性进展。",
            "title": "深度学习入门",
            "category": "AI",
            "views": 1200,
        },
        {
            "content": "RAG（检索增强生成）结合检索和生成技术，先检索相关知识库，再让大语言模型基于检索结果生成答案。",
            "title": "RAG 技术解析",
            "category": "LLM",
            "views": 600,
        },
        {
            "content": "Milvus 是一个开源的向量数据库，专门用于存储和搜索向量数据，支持亿级向量毫秒级检索。",
            "title": "Milvus 向量数据库",
            "category": "Database",
            "views": 500,
        },
    ]

    return documents


def insert_single_data():
    """演示单条插入数据

    适合实时写入场景，但性能较低。
    """
    print("=" * 60)
    print("示例 5: 单条插入数据")
    print("=" * 60)

    from pymilvus import MilvusClient

    client = MilvusClient(uri=MILVUS_URI)
    collection_name = "simple_docs"

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        dimension=DEFAULT_DIMENSION,
        auto_id=True,
        metric_type="COSINE",
    )

    documents = prepare_test_data()
    vectors = generate_mock_embeddings([d["content"] for d in documents])

    print(f"准备插入 {len(documents)} 条数据...\n")

    for i, doc in enumerate(documents):
        data = {
            "content": doc["content"],
            "vector": vectors[i],
        }
        result = client.insert(collection_name=collection_name, data=data)
        print(f"✓ 插入第 {i+1} 条：{doc['title'][:10]}...")
        print(f"  返回结果：{result}")

    print()
    stats = client.get_collection_stats(collection_name)
    print(f"插入完成，总行数：{stats.get('row_count', 0)}")

    return client, collection_name


def insert_batch_data():
    """演示批量插入数据（推荐）

    一次性插入多条数据，性能更好。推荐方式。
    """
    print("=" * 60)
    print("示例 6: 批量插入数据（推荐）")
    print("=" * 60)

    from pymilvus import MilvusClient

    client = MilvusClient(uri=MILVUS_URI)
    collection_name = "simple_docs"

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        dimension=DEFAULT_DIMENSION,
        auto_id=True,
        metric_type="COSINE",
    )

    documents = prepare_test_data()
    texts = [d["content"] for d in documents]
    vectors = generate_mock_embeddings(texts)

    print(f"准备批量插入 {len(documents)} 条数据...\n")

    data_to_insert = []
    for i, doc in enumerate(documents):
        data_to_insert.append({
            "content": doc["content"],
            "vector": vectors[i],
            "title": doc["title"],
        })

    result = client.insert(collection_name=collection_name, data=data_to_insert)

    print(f"✓ 批量插入完成")
    print(f"  插入数量：{result.get('insert_count', len(data_to_insert))}")
    print(f"  返回 ID 数量：{len(result.get('ids', []))}")

    stats = client.get_collection_stats(collection_name)
    print(f"\n总行数：{stats.get('row_count', 0)}")

    return client, collection_name


def insert_with_custom_fields():
    """向自定义字段的 Collection 插入数据

    演示如何插入包含多个标量字段的数据。
    """
    print("=" * 60)
    print("示例 7: 自定义字段 Collection 插入")
    print("=" * 60)

    from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
    from pymilvus.milvus_client import IndexParams

    client = MilvusClient(uri=MILVUS_URI)
    collection_name = "custom_docs"

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="views", dtype=DataType.INT64),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DEFAULT_DIMENSION),
    ]
    schema = CollectionSchema(fields=fields)

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        metric_type="COSINE",
    )

    documents = prepare_test_data()
    texts = [d["content"] for d in documents]
    vectors = generate_mock_embeddings(texts)

    print(f"准备插入带完整字段的数据...\n")

    data_to_insert = []
    for i, doc in enumerate(documents):
        data_to_insert.append({
            "content": doc["content"],
            "title": doc["title"],
            "category": doc["category"],
            "views": doc["views"],
            "vector": vectors[i],
        })

    result = client.insert(collection_name=collection_name, data=data_to_insert)
    print(f"✓ 插入完成")
    print(f"  插入数量：{result.get('insert_count', 0)}")

    print("\n检查索引...")
    index_list = client.list_indexes(collection_name=collection_name)
    if not index_list:
        index_params = IndexParams()
        index_params.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")
        client.create_index(collection_name=collection_name, index_params=index_params)
        print("索引已创建")
    else:
        print("索引已存在")

    print("加载集合...")
    client.load_collection(collection_name=collection_name)

    print("验证数据（查询前 3 条）：")
    res = client.query(
        collection_name=collection_name,
        filter="",
        output_fields=["title", "category", "views"],
        limit=3,
    )

    for i, item in enumerate(res):
        print(f"  {i+1}. {item['title']} | 类别：{item['category']} | 浏览：{item['views']}")

    return client, collection_name


def insert_with_custom_id():
    """演示手动指定 ID 的插入方式

    适用于需要同步外部 ID 的场景。
    """
    print("=" * 60)
    print("示例 8: 手动指定 ID 插入")
    print("=" * 60)

    from pymilvus import MilvusClient
    from pymilvus.milvus_client import IndexParams

    client = MilvusClient(uri=MILVUS_URI)
    collection_name = "custom_id_docs"

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        dimension=DEFAULT_DIMENSION,
        auto_id=False,
        metric_type="COSINE",
    )

    documents = prepare_test_data()
    texts = [d["content"] for d in documents]
    vectors = generate_mock_embeddings(texts)

    print(f"手动指定 ID 插入数据...\n")

    data_to_insert = []
    custom_ids = [1001, 1002, 1003, 1004, 1005]

    for i, doc in enumerate(documents):
        data_to_insert.append({
            "id": custom_ids[i],
            "content": doc["content"],
            "vector": vectors[i],
        })

    result = client.insert(collection_name=collection_name, data=data_to_insert)
    print(f"✓ 插入完成")
    print(f"  返回的 ID：{result.get('ids', [])}")

    print("\n检查索引...")
    index_list = client.list_indexes(collection_name=collection_name)
    if not index_list:
        index_params = IndexParams()
        index_params.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")
        client.create_index(collection_name=collection_name, index_params=index_params)
        print("索引已创建")
    else:
        print("索引已存在")

    print("加载集合...")
    client.load_collection(collection_name=collection_name)

    print("验证：通过 ID 查询 id=1003 的数据")
    res = client.query(
        collection_name=collection_name,
        filter="id == 1003",
        output_fields=["content"],
    )

    if res:
        print(f"  查询结果：{res[0]['content'][:30]}...")

    return client, collection_name


def insert_best_practices():
    """插入数据的最佳实践建议"""
    print("=" * 60)
    print("示例 9: 插入数据最佳实践")
    print("=" * 60)

    print(f"""
┌─────────────────────────────────────────────────────────┐
│ 插入数据最佳实践                                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 1. 批量插入优于单条插入                                  │
│    - 批量大小建议：100-1000 条/批                         │
│    - 太少：网络开销大                                   │
│    - 太多：内存占用高，失败重试成本大                   │
│                                                         │
│ 2. 向量维度必须匹配                                      │
│    - 插入前检查 Collection 的 dimension                   │
│    - 向量长度 != dimension 会报错（当前：{DEFAULT_DIMENSION} 维）        │
│                                                         │
│ 3. 数据类型要一致                                        │
│    - VARCHAR 字段不能超过 max_length                     │
│    - INT64 字段必须是整数                               │
│                                                         │
│ 4. 错误处理                                              │
│    - 捕获异常并记录失败的 data                          │
│    - 实现重试机制                                       │
│                                                         │
│ 5. 性能优化                                              │
│    - 插入前不创建索引（插入完成后再建索引）             │
│    - 大量数据插入时考虑分区                             │
│                                                         │
│ 6. ID 选择策略                                           │
│    - 无需外部同步：用 auto_id=True（推荐）              │
│    - 需要同步外部 ID：用 auto_id=False                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  Milvus 基础 - 插入数据")
    print("=" * 70 + "\n")

    # explain_embedding()
    # print()

    generate_embedding_with_llm("我是一支笔")

    # insert_single_data()
    # print()
    #
    # insert_batch_data()
    # print()
    #
    # insert_with_custom_fields()
    # print()
    #
    # insert_with_custom_id()
    # print()
    #
    # insert_best_practices()

    print("\n" + "=" * 70)
    print("  数据插入学习完成！接下来：04_create_index.py（创建索引）")
    print("=" * 70)
