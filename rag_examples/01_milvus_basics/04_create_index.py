# =============================================================================
# 04_create_index — 创建索引（Index）
# =============================================================================
# 用途：学习在 Milvus 中创建索引加速检索
# 难度：⭐⭐⭐（3 星）
#
# 核心概念：
#   - 索引 = 加速检索的数据结构（类比"书的目录"）
#   - FLAT / IVF_FLAT / HNSW 三种常用索引的创建与检索
#   - 索引选型与参数调优
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

    print(f"  ✓ 已插入 {len(documents)} 条测试数据")


def _demo_index(index_type, index_params=None, search_params=None, query="测试查询"):
    """三种索引共用的演示骨架：建集合 → 插数据 → 建索引 → 检索。

    FLAT / IVF_FLAT / HNSW 的流程完全一致，只有索引类型和参数不同，
    故抽成这一个函数，避免三处复制粘贴。

    关键点：`create_collection(dimension=...)` 会自动建一个 AUTOINDEX 索引并加载集合，
    要换成指定索引类型，需先 release + drop 掉 AUTOINDEX，再 create_index。

    参数:
        index_type: 索引类型（FLAT / IVF_FLAT / HNSW）
        index_params: 建索引参数（如 {"nlist": 4} 或 {"M": 16, "efConstruction": 200}）
        search_params: 检索参数（如 {"params": {"nprobe": 3}} 或 {"params": {"ef": 64}}）
        query: 检索用的查询词
    """
    collection_name = f"{index_type.lower()}_index_demo"
    client = MilvusClient(uri=MILVUS_URI)

    # 1. 建集合（先删旧集合，保证可重复运行）
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        dimension=DEFAULT_DIMENSION,
        auto_id=True,
        metric_type="COSINE",
    )

    # 2. 插入测试数据
    insert_test_data(client, collection_name)

    # 3. 换掉自动建的 AUTOINDEX，改成指定索引类型
    client.release_collection(collection_name=collection_name)
    for index_name in client.list_indexes(collection_name=collection_name):
        client.drop_index(collection_name=collection_name, index_name=index_name)

    index_params_obj = client.prepare_index_params()
    add_index_kwargs = {}
    if index_params:
        add_index_kwargs["params"] = index_params
    index_params_obj.add_index(
        field_name="vector",
        index_type=index_type,
        metric_type="COSINE",
        **add_index_kwargs,
    )
    client.create_index(collection_name=collection_name, index_params=index_params_obj)

    index_info = client.describe_index(collection_name, "vector")
    print(f"  ✓ {index_type} 索引创建完成（度量：{index_info['metric_type']}）")

    # 4. 加载并检索
    client.load_collection(collection_name=collection_name)
    test_vector = generate_mock_embeddings([query])

    search_kwargs = {}
    if search_params:
        search_kwargs["search_params"] = search_params
    res = client.search(
        collection_name=collection_name,
        data=[test_vector],
        limit=3,
        output_fields=["content"],
        **search_kwargs,
    )

    for i, hit in enumerate(res[0]):
        print(f"  {i+1}. 相似度：{hit['distance']:.4f} | 内容：{hit['entity']['content'][:20]}...")

    return client, collection_name


def create_flat_index():
    """FLAT 索引（精确搜索）—— 不建索引，暴力比较，100% 精度"""
    print(f"\n-- 示例 1: FLAT 索引（精确搜索）")
    print("   FLAT 不建索引、暴力比对，适合数据量小（<1 万条）、要求 100% 精度的场景。")
    return _demo_index("FLAT")


def create_ivf_flat_index():
    """IVF_FLAT 索引（倒排文件）—— 把向量分 nlist 个簇，只在相关簇内搜索"""
    print(f"\n-- 示例 2: IVF_FLAT 索引（倒排文件索引）")
    print("   把向量空间分成 nlist 个簇，只在相关簇内搜索，适合 1 万-100 万条。")
    return _demo_index(
        "IVF_FLAT",
        index_params={"nlist": 4},
        search_params={"params": {"nprobe": 3}},
    )


def create_hnsw_index():
    """HNSW 索引（图索引）—— 多层图导航，高精度低延迟"""
    print(f"\n-- 示例 3: HNSW 索引（图索引，高精度）")
    print("   可导航小世界图，多层图快速导航，精度最高、延迟最低，但内存占用较高。")
    return _demo_index(
        "HNSW",
        index_params={"M": 16, "efConstruction": 200},
        search_params={"params": {"ef": 64}},
    )


def index_types_explained():
    """索引选型表 + 参数调优"""
    print(f"\n-- 示例 4: 索引选型与参数调优")

    print(f"""
┌────────────┬──────────────┬──────────────────────┬───────────────┬────────┐
│ 索引类型    │ 适用数据量    │ 特点                  │ 延迟(100万向量) │ 召回率 │
├────────────┼──────────────┼──────────────────────┼───────────────┼────────┤
│ FLAT       │ <1 万        │ 精确暴力搜索，无需训练  │ ~500ms        │ 100%   │
│ IVF_FLAT   │ 1万-100万    │ 倒排文件，分 nlist 簇  │ ~50ms         │ 95-98% │
│ IVF_PQ     │ >100 万      │ 乘积量化，压缩存储      │ ~20ms         │ 90-95% │
│ HNSW       │ 高精度低延迟  │ 图导航，精度高速度快    │ ~10ms         │ 98-99% │
└────────────┴──────────────┴──────────────────────┴───────────────┴────────┘

💡 选型建议：
  1. <1 万条、要 100% 精度          → FLAT
  2. 1万-100万条、通用场景          → IVF_FLAT（nlist = √N）
  3. >100万条、可接受轻微精度损失    → IVF_PQ
  4. 高精度、低延迟、内存充裕        → HNSW
  5. 不确定怎么选                    → AUTOINDEX（Milvus 自动选型，本示例 1-3 里被换掉的那个）

📊 参数调优：
  IVF_FLAT:  nlist = √N（分簇数，N 为向量总数）；nprobe = √nlist（搜索时探查的簇数，越大越准越慢）
  HNSW:      M = 16-64（每个点的连接数）；efConstruction = 200-400（构建时）；ef = 64-256（搜索时）
""")

    print("⚡ 加速原理（100 万向量）：")
    print("   FLAT：    逐个比较 100 万次 → ~500ms")
    print("   IVF_FLAT：只比较相关簇，约 1 万次 → ~50ms（约 10 倍加速）")
    print("   HNSW：    图导航只需几步 → ~10ms（约 50 倍加速）")


def index_management():
    """索引管理操作：查看 / 删除 / 重建"""
    print(f"\n-- 示例 5: 索引管理操作")

    print("1. 查看索引：")
    print("   client.list_indexes(collection_name)            # 列出所有索引名")
    print("   client.describe_index(collection_name, 'vector')  # 查看某索引详情")

    print("\n2. 删除索引（需先 release，不能删除已加载集合上的索引）：")
    print("   client.release_collection(collection_name)")
    print("   client.drop_index(collection_name, 'vector')")

    print("\n3. 何时需要重建索引：")
    print("   - 数据大量更新后")
    print("   - 需要调整索引参数（如增大 nlist）")
    print("   - 索引损坏或异常")
    print("   重建 = 删除旧索引 → create_index 建新索引 → load_collection")


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

    # index_management()

    print("\n" + "=" * 70)
    print("  索引学习完成！接下来：02_document_chunking（文档切片）")
    print("=" * 70)
