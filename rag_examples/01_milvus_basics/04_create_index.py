# =============================================================================
# 04_create_index — 创建索引（Index）
# =============================================================================
# 用途：学习在 Milvus 中创建索引加速检索
# 难度：⭐⭐⭐（3 星）
#
# 核心概念：
#   - 索引 = 加速检索的数据结构（类比"书的目录"）
#   - FLAT / IVF_FLAT / IVF_PQ / HNSW 索引类型对比
#   - 索引参数调优
# =============================================================================

import random
from pymilvus import MilvusClient
from rag_examples.milvus_config import MILVUS_URI, DEFAULT_DIMENSION


def prepare_test_data():
    """准备测试用的文档数据"""
    documents = [
        {"content": "人工智能是模拟人类智能的计算机科学领域。", "category": "AI"},
        {"content": "机器学习通过训练数据让计算机自动学习规律。", "category": "AI"},
        {"content": "深度学习使用多层神经网络模拟人脑。", "category": "AI"},
        {"content": "RAG 结合检索和生成技术，先检索再生成答案。", "category": "LLM"},
        {"content": "Milvus 是开源的向量数据库，支持亿级向量检索。", "category": "Database"},
        {"content": "自然语言处理让计算机理解和生成人类语言。", "category": "AI"},
        {"content": "计算机视觉让计算机能够'看懂'图像和视频。", "category": "AI"},
        {"content": "大语言模型是基于海量文本训练的深度学习和模型。", "category": "LLM"},
        {"content": "知识图谱用图结构存储和表示知识。", "category": "Knowledge"},
        {"content": "推荐系统根据用户偏好推荐相关内容。", "category": "Application"},
    ]
    return documents


def generate_mock_embeddings(texts, dim=DEFAULT_DIMENSION):
    """模拟 Embedding 生成的向量数据

    ⚠️ 仅用于教学演示。实际项目必须使用真实的 Embedding 模型。
    """
    random.seed(42)

    if isinstance(texts, str):
        texts = [texts]

    vectors = []
    for text in texts:
        vector = [random.random() for _ in range(dim)]
        vectors.append(vector)

    return vectors if len(texts) > 1 else vectors[0]


def insert_test_data(client, collection_name):
    """向 Collection 插入测试数据"""
    documents = prepare_test_data()
    texts = [d["content"] for d in documents]
    vectors = generate_mock_embeddings(texts)

    data_to_insert = []
    for i, doc in enumerate(documents):
        data_to_insert.append({
            "content": doc["content"],
            "category": doc["category"],
            "vector": vectors[i],
        })

    client.insert(collection_name, data_to_insert)
    client.flush(collection_name=collection_name)

    print(f"✓ 已插入 {len(documents)} 条测试数据")


def create_flat_index():
    """创建 FLAT 索引（精确搜索）

    FLAT = 不建索引，暴力搜索。
    适合数据量小（<1 万条）、要求 100% 精度的场景。
    """
    print("=" * 60)
    print("示例 2: FLAT 索引（精确搜索）")
    print("=" * 60)

    client = MilvusClient(uri=MILVUS_URI)
    collection_name = "flat_index_demo"

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        dimension=DEFAULT_DIMENSION,
        auto_id=True,
        metric_type="COSINE",
    )

    insert_test_data(client, collection_name)

    print("\n创建 FLAT 索引...")

    index_list = client.list_indexes(collection_name=collection_name)
    client.release_collection(collection_name)
    if index_list:
        for index_name in index_list:
            client.drop_index(collection_name=collection_name, index_name=index_name)
        print(f"  已删除现有索引: {index_list}")

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="FLAT",
        metric_type="COSINE",
    )

    client.create_index(collection_name=collection_name, index_params=index_params)
    print("✓ FLAT 索引创建完成")

    index_info = client.describe_index(collection_name, "vector")
    print(f"  索引类型：{index_info['index_type']}")
    print(f"  度量类型：{index_info['metric_type']}")

    print("\n测试检索（FLAT 索引）：")
    test_vector = generate_mock_embeddings(["Milvus"])

    client.load_collection(collection_name)

    res = client.search(
        collection_name=collection_name,
        data=[test_vector],
        limit=3,
        output_fields=["content"],
    )

    for i, hit in enumerate(res[0]):
        print(f"  {i+1}. 相似度：{hit['distance']:.4f} | 内容：{hit['entity']['content'][:20]}...")

    return client, collection_name


def create_ivf_flat_index():
    """创建 IVF_FLAT 索引（倒排文件索引）

    将向量空间分成 nlist 个簇，只在相关簇内搜索。
    适合 1 万-100 万条数据的中等规模场景。
    """
    print("=" * 60)
    print("示例 3: IVF_FLAT 索引（倒排文件索引）")
    print("=" * 60)

    client = MilvusClient(uri=MILVUS_URI)
    collection_name = "ivf_flat_index_demo"

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        dimension=DEFAULT_DIMENSION,
        auto_id=True,
        metric_type="COSINE",
    )

    insert_test_data(client, collection_name)

    print("\n创建 IVF_FLAT 索引...")

    client.release_collection(collection_name=collection_name)

    index_list = client.list_indexes(collection_name=collection_name)
    if index_list:
        for index_name in index_list:
            client.drop_index(collection_name=collection_name, index_name=index_name)
        print(f"  已删除现有索引: {index_list}")

    nlist = 4
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": nlist},
    )

    client.create_index(collection_name=collection_name, index_params=index_params)

    print(f"✓ IVF_FLAT 索引创建完成（nlist={nlist}）")

    client.load_collection(collection_name=collection_name)

    index_info = client.describe_index(collection_name, "vector")
    print(f"  索引类型：{index_info['index_type']}")
    print(f"  度量类型：{index_info['metric_type']}")

    print("\n测试检索（IVF_FLAT 索引）：")
    test_vector = generate_mock_embeddings(["测试查询"])

    res = client.search(
        collection_name=collection_name,
        data=[test_vector],
        limit=3,
        output_fields=["content"],
        search_params={"params": {"nprobe": 3}},
    )

    for i, hit in enumerate(res[0]):
        print(f"  {i+1}. 相似度：{hit['distance']:.4f} | 内容：{hit['entity']['content'][:20]}...")

    return client, collection_name


def create_hnsw_index():
    """创建 HNSW 索引（图索引，高精度）

    HNSW = Hierarchical Navigable Small World。
    通过构建多层图结构实现快速导航搜索。
    适合高精度、低延迟场景。
    """
    print("=" * 60)
    print("示例 4: HNSW 索引（图索引，高精度）")
    print("=" * 60)

    client = MilvusClient(uri=MILVUS_URI)
    collection_name = "hnsw_index_demo"

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={
            "M": 16,
            "efConstruction": 200,
        },
    )

    client.create_collection(
        collection_name=collection_name,
        dimension=DEFAULT_DIMENSION,
        auto_id=True,
        metric_type="COSINE",
        index_params=index_params,
    )

    print("✓ 集合和 HNSW 索引创建完成")

    insert_test_data(client, collection_name)

    index_info = client.describe_index(collection_name, "vector")
    print(f"  索引类型：{index_info['index_type']}")
    print(f"  度量类型：{index_info['metric_type']}")

    print("\n测试检索（HNSW 索引）：")
    test_vector = generate_mock_embeddings(["测试查询"])

    res = client.search(
        collection_name=collection_name,
        data=[test_vector],
        limit=3,
        output_fields=["content"],
        search_params={"params": {"ef": 64}},
    )

    for i, hit in enumerate(res[0]):
        print(f"  {i+1}. 相似度：{hit['distance']:.4f} | 内容：{hit['entity']['content'][:20]}...")

    return client, collection_name


def index_types_explained():
    """详解 Milvus 支持的索引类型"""
    print("=" * 60)
    print("示例 5: 索引类型详解")
    print("=" * 60)

    print("""
┌─────────────────────────────────────────────────────────┐
│ Milvus 索引类型对比                                      │
├──────────────┬────────────────┬─────────────────────────┤
│    索引类型   │    适用场景     │         特点             │
├──────────────┼────────────────┼─────────────────────────┤
│ FLAT         │ <1 万条数据     │ 精确搜索，无需训练       │
│              │ 小数据量场景    │ 100% 精度，无误差         │
├──────────────┼────────────────┼─────────────────────────┤
│ IVF_FLAT     │ 1 万 -100 万条   │ 倒排文件索引             │
│              │ 通用场景       │ 将向量分 nlist 组         │
│              │                │ 只在相关组内搜索         │
├──────────────┼────────────────┼─────────────────────────┤
│ IVF_PQ       │ >100 万条       │ 乘积量化索引             │
│              │ 超大数据量     │ 向量压缩存储             │
│              │                │ 有一定精度损失           │
├──────────────┼────────────────┼─────────────────────────┤
│ IVF_SQ8      │ 内存受限场景    │ 标量量化索引             │
│              │                │ 8bit 压缩存储             │
├──────────────┼────────────────┼─────────────────────────┤
│ HNSW         │ 高精度场景      │ 可导航小世界图           │
│              │ 对延迟敏感     │ 精度最高，速度最快       │
│              │                │ 内存占用较高             │
├──────────────┼────────────────┼─────────────────────────┤
│ ANNOY        │ 读多写少场景    │ 近似最近邻搜索           │
│              │                │ 不支持动态增删           │
└──────────────┴────────────────┴─────────────────────────┘

💡 索引选择建议:

1. 数据量 < 1 万：→ FLAT
2. 数据量 1 万 - 100 万：→ IVF_FLAT（nlist = √N）
3. 数据量 > 100 万：→ IVF_PQ
4. 高精度要求：→ HNSW
5. 内存受限：→ IVF_SQ8

📊 参数调优:

IVF_FLAT:
  - nlist: √N（N 为向量数）
  - nprobe: √nlist（搜索时探查的分组数）

HNSW:
  - M: 16-64（默认 16）
  - efConstruction: 200-400（构建时）
  - ef: 64-256（搜索时）
""")


def index_management():
    """索引的管理操作

    查看、删除、重建索引。
    """
    print("=" * 60)
    print("示例 6: 索引管理操作")
    print("=" * 60)

    client = MilvusClient(uri=MILVUS_URI)
    collection_name = "flat_index_demo"

    print("1. 查看索引信息：")
    index_list = client.list_indexes(collection_name)
    if index_list:
        index_info = client.describe_index(collection_name, "vector")
        print(f"   索引类型：{index_info['index_type']}")
        print(f"   字段名：{index_info['field_name']}")
    else:
        print("   暂无索引")
    print()

    print("2. 删除索引（示例代码）：")
    print("   client.drop_index(collection_name, 'vector')")
    print()

    print("3. 重建索引场景：")
    print("""
   何时需要重建索引？
   - 数据大量更新后
   - 索引参数需要调整
   - 索引损坏或异常

   重建步骤：
   1. 删除旧索引：client.drop_index(...)
   2. 创建新索引：client.create_index(...)
   3. 等待索引构建完成
""")


def index_performance_comparison():
    """索引性能对比说明"""
    print("=" * 60)
    print("示例 7: 索引性能对比")
    print("=" * 60)

    print(f"""
┌─────────────────────────────────────────────────────────┐
│ 不同索引类型性能对比（100 万向量，{DEFAULT_DIMENSION} 维）                   │
├──────────────┬─────────────┬────────────┬───────────────┤
│    索引类型   │  构建时间    │  检索延迟   │   召回率      │
├──────────────┼─────────────┼────────────┼───────────────┤
│ FLAT         │ 无需构建     │  ~500ms    │   100%       │
│ IVF_FLAT     │   ~30 秒     │  ~50ms     │   95-98%     │
│ IVF_PQ       │   ~20 秒     │  ~20ms     │   90-95%     │
│ HNSW         │   ~60 秒     │  ~10ms     │   98-99%     │
└──────────────┴─────────────┴────────────┴───────────────┘

💡 检索速度对比（100 万向量）:
- FLAT:    需要比较 100 万次 → 500ms
- IVF_FLAT: 只需比较 1 万 次 → 50ms (10 倍加速)
- HNSW:    图导航只需几步 → 10ms (50 倍加速)
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  Milvus 基础 - 创建索引")
    print("=" * 70 + "\n")

    create_flat_index()
    print()

    create_ivf_flat_index()
    print()

    create_hnsw_index()
    print()

    index_types_explained()
    print()

    index_management()
    print()

    index_performance_comparison()

    print("\n" + "=" * 70)
    print("  索引学习完成！接下来：02_document_chunking（文档切片）")
    print("=" * 70)
