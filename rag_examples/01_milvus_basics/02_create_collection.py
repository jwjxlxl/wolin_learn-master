# =============================================================================
# 02_create_collection — 创建 Collection（集合）
# =============================================================================
# 用途：学习在 Milvus 中创建集合的各种方式
# 难度：⭐⭐（2 星）
#
# 核心概念：
#   - Collection = 数据库中的"表"（类比 Excel 表格）
#   - Field Schema 定义字段模式
#   - 向量字段 vs 标量字段
#   - 主键（自增 vs 手动指定）
# =============================================================================

from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
from pymilvus.milvus_client import IndexParams
from rag_examples.milvus_config import MILVUS_URI, DEFAULT_DIMENSION, DEFAULT_METRIC_TYPE


def create_simple_collection():
    """创建一个简单的 Collection

    包含最基础的字段：ID（自增）、文本、向量。
    """
    print("=" * 60)
    print("示例 1: 创建简单的 Collection")
    print("=" * 60)

    client = MilvusClient(uri=MILVUS_URI)

    collection_name = "simple_docs"

    if client.has_collection(collection_name):
        print(f"删除已存在的集合：{collection_name}")
        client.drop_collection(collection_name)

    print(f"创建集合：{collection_name}")

    client.create_collection(
        collection_name=collection_name,
        dimension=DEFAULT_DIMENSION,
        auto_id=True,
        metric_type=DEFAULT_METRIC_TYPE,
    )

    print("✓ 创建成功！")
    print()

    info = client.describe_collection(collection_name)
    print(f"集合信息：")
    print(f"  名称：{info['collection_name']}")
    dimension = None
    for field in info.get('fields', []):
        if field.get('type') == 101:
            dimension = field.get('params', {}).get('dim')
            break
    print(f"  向量维度：{dimension}")
    print(f"  度量类型：{DEFAULT_METRIC_TYPE}")
    print(f"  主键自增：{info['auto_id']}")

    return client, collection_name


def create_custom_collection():
    """创建自定义字段的 Collection

    定义多个标量字段，支持更复杂的查询。
    """
    print("=" * 60)
    print("示例 2: 创建自定义字段的 Collection")
    print("=" * 60)

    client = MilvusClient(uri=MILVUS_URI)

    collection_name = "custom_docs"

    if client.has_collection(collection_name):
        print(f"删除已存在的集合：{collection_name}")
        client.drop_collection(collection_name)

    print(f"创建集合：{collection_name}")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="views", dtype=DataType.INT64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DEFAULT_DIMENSION),
    ]

    schema = CollectionSchema(fields=fields)

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        metric_type=DEFAULT_METRIC_TYPE,
    )

    print("✓ 创建成功！")
    print()

    info = client.describe_collection(collection_name)
    print(f"字段列表：")
    for field in info['fields']:
        field_name = field['name']
        field_type = field['type']
        is_primary = field.get('is_primary', False)
        print(f"  - {field_name}: {field_type} {'(主键)' if is_primary else ''}")


def create_multi_vector_collection():
    """创建包含多个向量字段的 Collection

    适用于多模态场景（如同时有文本向量和图像向量）。
    """
    print("=" * 60)
    print("示例 3: 创建多向量字段的 Collection")
    print("=" * 60)

    client = MilvusClient(uri=MILVUS_URI)
    collection_name = "multi_vector_docs"

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    print(f"创建集合：{collection_name}")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="text_embedding", dtype=DataType.FLOAT_VECTOR, dim=DEFAULT_DIMENSION),
        FieldSchema(name="image_embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
    ]

    schema = CollectionSchema(fields=fields)

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        metric_type=DEFAULT_METRIC_TYPE,
    )

    print("✓ 创建成功！")
    print()
    print("💡 多向量字段适用于：")
    print("   - 图文混合检索")
    print("   - 多模态 RAG 系统")
    print("   - 跨模态搜索（以文搜图、以图搜文）")


def create_dynamic_field_collection():
    """创建启用动态字段的 Collection（Milvus 2.3+ 新特性）

    动态字段允许插入数据时自动添加未预定义的字段，
    无需修改 Schema 就能存储不同结构的数据。
    """
    print("=" * 60)
    print("示例 4: 动态字段（Milvus 2.3+ 新特性）")
    print("=" * 60)

    client = MilvusClient(uri=MILVUS_URI)
    collection_name = "dynamic_field_docs"

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DEFAULT_DIMENSION),
    ]

    schema = CollectionSchema(fields=fields, enable_dynamic_field=True)

    print("创建集合（启用动态字段）...")
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        metric_type=DEFAULT_METRIC_TYPE,
    )

    print("✓ 创建成功！")
    print()

    # 演示：插入带有动态字段的数据
    print("插入数据（包含未预定义的动态字段）：")

    data = [
        {
            "content": "人工智能简介",
            "vector": [0.1] * DEFAULT_DIMENSION,
            "author": "张三",
            "tags": ["AI", "科技"],
            "publish_date": "2024-01-15",
        },
        {
            "content": "机器学习基础",
            "vector": [0.2] * DEFAULT_DIMENSION,
            "author": "李四",
            "course_id": 101,
            "difficulty": "中级",
        },
        {
            "content": "深度学习进阶",
            "vector": [0.3] * DEFAULT_DIMENSION,
            "author": "王五",
            "video_url": "https://example.com/video",
            "duration_minutes": 45,
        },
    ]

    client.insert(collection_name=collection_name, data=data)
    print(f"  插入 {len(data)} 条数据，每条数据有不同的动态字段")
    print()

    print("创建索引并加载集合...")
    index_params = IndexParams()
    index_params.add_index(field_name="vector", index_type="FLAT", metric_type=DEFAULT_METRIC_TYPE)
    client.create_index(collection_name=collection_name, index_params=index_params)
    client.load_collection(collection_name=collection_name)

    print("查询所有数据，查看动态字段：")
    results = client.query(
        collection_name=collection_name,
        filter="",
        output_fields=["*"],
        limit=10,
    )

    for i, item in enumerate(results):
        print(f"\n  [{i+1}] ID: {item.get('id')}")
        print(f"      内容：{item.get('content', '')[:20]}...")
        dynamic_keys = [k for k in item.keys() if k not in ['id', 'content', 'vector']]
        if dynamic_keys:
            print(f"      动态字段：{dynamic_keys}")
            for key in dynamic_keys:
                print(f"        - {key}: {item[key]}")

    print("\n[OK] 动态字段演示完成！")

    return client, collection_name


def metric_types_explained():
    """详解 Milvus 支持的度量类型

    COSINE（余弦相似度）、L2（欧几里得距离）、IP（内积）的对比和选择建议。
    """
    print("=" * 60)
    print("示例 5: 度量类型详解")
    print("=" * 60)

    print("""
┌─────────────────────────────────────────────────────────┐
│ Milvus 度量类型对比                                      │
├──────────────┬────────────────┬─────────────────────────┤
│    类型       │    计算方式     │         特点             │
├──────────────┼────────────────┼─────────────────────────┤
│ L2           │ 欧几里得距离    │ 值越小越相似             │
│ (EUCLIDEAN)  │ √(Σ(xi-yi)²)   │ 适用于归一化前的向量     │
├──────────────┼────────────────┼─────────────────────────┤
│ IP           │ 内积            │ 值越大越相似             │
│ (INNER_PRODUCT)│ Σ(xi*yi)     │ 与 COSINE 类似           │
├──────────────┼────────────────┼─────────────────────────┤
│ COSINE       │ 余弦相似度      │ 值越大越相似 (范围 -1~1) │
│              │ cos(θ)         │ 最常用的类型             │
└──────────────┴────────────────┴─────────────────────────┘

💡 如何选择度量类型？

1. COSINE（推荐默认选择）
   ✓ 最常用，适用于大多数语义检索场景
   ✓ 不受向量长度影响，只关注方向
   ✓ Embedding 模型通常输出归一化向量

2. L2（欧几里得距离）
   ✓ 适用于向量未归一化的场景
   ✓ 考虑向量的绝对位置

3. IP（内积）
   ✓ 与 COSINE 在归一化后等价
   ✓ 某些场景下计算更快

📊 示例对比:
假设有两个向量:
A = [1, 2, 3]
B = [2, 4, 6]  (A 的 2 倍)

- COSINE(A, B) = 1.0  (方向相同，完全相似)
- L2(A, B) = √14 ≈ 3.74 (距离不为 0)

结论：COSINE 更关注"语义方向"，L2 更关注"绝对位置"
""")


def collection_operations():
    """Collection 的常见操作

    列出、检查存在、查看详情、统计行数、删除、重命名等。
    """
    print("=" * 60)
    print("示例 6: Collection 操作大全")
    print("=" * 60)

    client = MilvusClient(uri=MILVUS_URI)

    # 1. 列出所有 Collection
    print("1. 列出所有集合：")
    collections = client.list_collections()
    for col in collections:
        print(f"   - {col}")
    print()

    # 2. 检查 Collection 是否存在
    print("2. 检查集合是否存在：")
    test_name = "simple_docs"
    exists = client.has_collection(test_name)
    print(f"   {test_name}: {'存在' if exists else '不存在'}")
    print()

    # 3. 获取 Collection 详细信息
    if exists:
        print("3. 获取集合详细信息：")
        info = client.describe_collection(test_name)
        print(f"   名称：{info['collection_name']}")
        dimension = None
        for field in info.get('fields', []):
            if field.get('type') == 101:
                dimension = field.get('params', {}).get('dim')
                break
        print(f"   维度：{dimension}")
        print(f"   主键自增：{info['auto_id']}")
        print()

    # 4. 获取 Collection 行数
    if exists:
        print("4. 获取集合行数：")
        count = client.get_collection_stats(test_name)
        print(f"   行数：{count.get('row_count', 0)}")
        print()

    # 5. 删除 Collection（仅示例代码）
    print("5. 删除集合（示例代码，不执行）：")
    print("   client.drop_collection('collection_name')")
    print()

    # 6. 重命名 Collection（仅示例代码）
    print("6. 重命名集合（示例代码，不执行）：")
    print("   client.rename_collection('old_name', 'new_name')")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  Milvus 基础 - 创建 Collection")
    print("=" * 70 + "\n")

    print("【说明】")
    print("  本示例演示多种创建 Collection 的方式")
    print("  建表前先删表，确保代码可重复运行")
    print()

    create_simple_collection()
    print()

    create_custom_collection()
    print()

    create_multi_vector_collection()
    print()

    metric_types_explained()
    print()

    collection_operations()

    print()
    print("=" * 70)
    print("  Collection 学习完成！接下来学习：03_insert_data.py（插入数据）")
    print("=" * 70)
