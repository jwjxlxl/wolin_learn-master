# =============================================================================
# 01_scalar_query — 标量查询（Scalar Query）
# =============================================================================
"""
本文件教学演示使用 Milvus 标量字段进行条件筛选。

核心概念：
  - 标量字段 vs 向量字段
  - 过滤表达式（Filter Expression）
  - 标量查询的应用场景

标量字段类型：INT64、FLOAT、VARCHAR、BOOL
适用场景：按类别筛选、按时间范围、按数值范围、组合条件查询
"""

import random
import time
from dotenv import load_dotenv
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
from rag_examples.milvus_config import MILVUS_URI, DEFAULT_DIMENSION

load_dotenv()


# =============================================================================
# 示例 1: 准备测试数据
# =============================================================================

def prepare_test_collection():
    """准备标量查询测试用的 Collection"""
    random.seed(42)

    client = MilvusClient(uri=MILVUS_URI)
    collection_name = "scalar_query_demo"

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="zhangliang", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="views", dtype=DataType.INT64),
        FieldSchema(name="price", dtype=DataType.FLOAT),
        FieldSchema(name="is_published", dtype=DataType.BOOL),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DEFAULT_DIMENSION),
    ]
    schema = CollectionSchema(fields=fields)

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        metric_type="COSINE"
    )

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="FLAT",
        metric_type="COSINE"
    )
    client.create_index(
        collection_name=collection_name,
        index_params=index_params
    )

    documents = [
        {"content": "人工智能简介", "title": "AI 入门", "category": "AI", "zhangliang": "深度学习", "views": 1000, "price": 0.0, "is_published": True},
        {"content": "机器学习基础", "title": "ML 教程", "category": "AI", "zhangliang": "强化学习", "views": 800, "price": 99.0, "is_published": True},
        {"content": "深度学习入门", "title": "DL 指南", "category": "AI", "zhangliang": "卷积神经网络", "views": 1200, "price": 129.0, "is_published": True},
        {"content": "自然语言处理", "title": "NLP 详解", "category": "AI", "zhangliang": "Transformer", "views": 600, "price": 89.0, "is_published": False},
        {"content": "计算机视觉", "title": "CV 应用", "category": "AI", "zhangliang": "目标检测", "views": 500, "price": 79.0, "is_published": True},
        {"content": "产品设计方法", "title": "产品指南", "category": "Product", "zhangliang": "用户调研", "views": 300, "price": 0.0, "is_published": True},
        {"content": "用户体验优化", "title": "UX 技巧", "category": "Product", "zhangliang": "交互设计", "views": 450, "price": 59.0, "is_published": True},
        {"content": "市场营销策略", "title": "营销实战", "category": "Marketing", "zhangliang": "数字营销", "views": 200, "price": 149.0, "is_published": False},
        {"content": "数据分析方法", "title": "分析教程", "category": "Data", "zhangliang": "统计分析", "views": 700, "price": 0.0, "is_published": True},
        {"content": "Python 编程", "title": "编程入门", "category": "Programming", "zhangliang": "面向对象", "views": 1500, "price": 79.0, "is_published": True},
    ]

    def mock_embedding():
        return [random.random() for _ in range(DEFAULT_DIMENSION)]

    data_to_insert = []
    for doc in documents:
        data_to_insert.append({
            "content": doc["content"],
            "title": doc["title"],
            "category": doc["category"],
            "zhangliang": doc["zhangliang"],
            "views": doc["views"],
            "price": doc["price"],
            "is_published": doc["is_published"],
            "embedding": mock_embedding()
        })

    client.insert(collection_name, data_to_insert)
    client.flush(collection_name)

    print(f"✓ 已创建测试集合：{collection_name}")
    print(f"  插入文档数：{len(documents)}")

    return client, collection_name


# =============================================================================
# 示例 2: 基础标量查询（单条件）
# =============================================================================

def basic_scalar_query(client, collection_name):
    """
    基础标量查询演示：
      1. 查询类别为'AI'的文档 → category == 'AI'
      2. 查询浏览量>800 的文档 → views > 800
      3. 查询已发布的文档 → is_published == True
    """
    client.load_collection(collection_name)

    print(f"\n-- 示例 2.1: 查询类别为'AI'的文档（category == 'AI'）")
    results = client.query(
        collection_name=collection_name,
        filter="category == 'AI'",
        output_fields=["title", "category", "views"],
        limit=10
    )
    for res in results:
        print(f"  - {res['title']} | 类别：{res['category']} | 浏览：{res['views']}")

    print(f"\n-- 示例 2.2: 查询浏览量>800 的文档（views > 800）")
    results = client.query(
        collection_name=collection_name,
        filter="views > 800",
        output_fields=["title", "views"],
        limit=10
    )
    for res in results:
        print(f"  - {res['title']} | 浏览：{res['views']}")

    print(f"\n-- 示例 2.3: 查询已发布的文档（is_published == True）")
    results = client.query(
        collection_name=collection_name,
        filter="is_published == True",
        output_fields=["title", "is_published"],
        limit=10
    )
    for res in results:
        status = "✓" if res['is_published'] else "✗"
        print(f"  {status} {res['title']}")


# =============================================================================
# 示例 3: 组合条件查询
# =============================================================================

def compound_scalar_query(client, collection_name):
    """演示 and / or / in 等组合过滤条件"""
    print(f"\n-- 示例 3.1: AI 类别且浏览量>600（category == 'AI' and views > 600）")
    results = client.query(
        collection_name=collection_name,
        filter="category == 'AI' and views > 600",
        output_fields=["title", "category", "views"],
        limit=10
    )
    for res in results:
        print(f"  - {res['title']} | 类别：{res['category']} | 浏览：{res['views']}")

    print(f"\n-- 示例 3.2: AI 或 Product 类别（category == 'AI' or category == 'Product'）")
    results = client.query(
        collection_name=collection_name,
        filter="category == 'AI' or category == 'Product'",
        output_fields=["title", "category"],
        limit=10
    )
    for res in results:
        print(f"  - {res['title']} | 类别：{res['category']}")

    print(f"\n-- 示例 3.3: 特定类别（category in ['AI', 'Data']）")
    results = client.query(
        collection_name=collection_name,
        filter="category in ['AI', 'Data']",
        output_fields=["title", "category"],
        limit=10
    )
    for res in results:
        print(f"  - {res['title']} | 类别：{res['category']}")

    print(f"\n-- 示例 3.4: 已发布且（浏览量>500 或价格=0）")
    results = client.query(
        collection_name=collection_name,
        filter="is_published == True and (views > 500 or price == 0)",
        output_fields=["title", "views", "price"],
        limit=10
    )
    for res in results:
        print(f"  - {res['title']} | 浏览：{res['views']} | 价格：{res['price']}")


# =============================================================================
# 示例 4: 范围查询
# =============================================================================

def range_query(client, collection_name):
    """演示数值范围查询（>=、<=、<、>）"""
    print(f"\n-- 示例 4.1: 浏览量在 500-1000 之间（views >= 500 and views <= 1000）")
    results = client.query(
        collection_name=collection_name,
        filter="views >= 500 and views <= 1000",
        output_fields=["title", "views"],
        limit=10
    )
    for res in results:
        print(f"  - {res['title']} | 浏览：{res['views']}")

    print(f"\n-- 示例 4.2: 价格<100 的文档（price < 100）")
    results = client.query(
        collection_name=collection_name,
        filter="price < 100",
        output_fields=["title", "price"],
        limit=10
    )
    for res in results:
        print(f"  - {res['title']} | 价格：¥{res['price']}")


# =============================================================================
# 示例 5: 标量 + 向量混合查询
# =============================================================================

def scalar_plus_vector_search(client, collection_name):
    """演示向量检索与标量过滤结合使用"""
    random.seed(42)
    query_vector = [random.random() for _ in range(DEFAULT_DIMENSION)]

    print(f"\n-- 示例 5.1: 向量检索（无过滤），返回 Top-3")
    results = client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=3,
        output_fields=["title", "category", "views"]
    )
    print(f"  检索结果：")
    for hit in results[0]:
        print(f"  [相似度：{hit['distance']:.4f}] {hit['entity']['title']} | 类别：{hit['entity']['category']}")

    print(f"\n-- 示例 5.2: 向量检索 + 标量过滤（仅 AI 类别）")
    results = client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=3,
        output_fields=["title", "category", "views"],
        filter="category == 'AI'"
    )
    print(f"  检索结果（仅 AI 类别）：")
    for hit in results[0]:
        print(f"  [相似度：{hit['distance']:.4f}] {hit['entity']['title']} | 类别：{hit['entity']['category']}")


# =============================================================================
# 示例 6: 标量查询最佳实践
# =============================================================================

def scalar_query_best_practices():
    """标量查询的最佳实践总结"""
    print(f"\n-- 示例 6: 标量查询最佳实践")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │ 标量查询最佳实践                                        │
  ├─────────────────────────────────────────────────────────┤
  │                                                         │
  │ 1. 过滤表达式语法                                        │
  │    - 等于：field == 'value'                             │
  │    - 不等于：field != 'value'                           │
  │    - 大于/小于：field > 100, field < 100                │
  │    - 大于等于/小于等于：field >= 100, field <= 100      │
  │    - 包含：field in ['a', 'b', 'c']                     │
  │    - 与/或：expr1 and expr2, expr1 or expr2             │
  │                                                         │
  │ 2. 性能优化建议                                          │
  │    - 标量字段建立索引（Milvus 会自动处理）              │
  │    - 避免在向量检索前做复杂标量过滤                     │
  │    - 先用向量检索召回，再用标量过滤精简                 │
  │                                                         │
  │ 3. 常见应用场景                                          │
  │    - 按类别/标签筛选内容                                │
  │    - 按时间范围过滤（最近 7 天、最近 30 天）             │
  │    - 按权限控制（只返回用户可访问的文档）               │
  │    - 按状态筛选（只返回已发布/审核通过的文档）          │
  │                                                         │
  │ 4. 注意事项                                              │
  │    - 字符串比较区分大小写                               │
  │    - 空值处理：field != '' 或 field IS NOT NULL         │
  │    - 特殊字符需要转义                                   │
  │                                                         │
  └─────────────────────────────────────────────────────────┘

  RAG 系统中的典型用法:

  场景：用户问"AI 相关的付费教程有哪些？"
  策略：向量检索 + 标量过滤

  ```python
  results = client.search(
      collection_name="knowledge_base",
      data=[query_vector],           # 问题的向量
      limit=5,
      filter="category == 'AI' and price > 0",  # 标量条件
      output_fields=["title", "content", "price"]
  )
  ```
""")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  标量查询（Scalar Query）")
    print("=" * 70 + "\n")

    client, collection_name = prepare_test_collection()
    basic_scalar_query(client, collection_name)
    compound_scalar_query(client, collection_name)
    range_query(client, collection_name)
    scalar_plus_vector_search(client, collection_name)
    scalar_query_best_practices()
