# =============================================================================
# 04_hybrid_search — 混合检索（Hybrid Search）
# =============================================================================
# 用途：学习结合向量检索和关键字检索的混合检索方法
# 难度：⭐⭐⭐（3 星）
# =============================================================================
# 核心概念：
#   - 多路召回：同时使用多种检索方法
#   - 结果融合：RRF（倒数排名融合）、加权平均
#   - 互补优势：向量检索（语义）+ 关键字检索（精确）
# =============================================================================

import random
from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# 简单混合检索
# =============================================================================
# 作用：用模拟分数演示混合检索的核心思想
#       1. 展示两种检索方法各自的打分结果
#       2. 用加权平均公式融合两种分数，得到最终排名
# 教学目的：让学生理解"多路召回 + 结果融合"的概念
#           不依赖真实向量模型/BM25 库，用模拟分数直观说明融合逻辑
# =============================================================================

def simple_hybrid_search():
    """简单的混合检索演示 — 向量检索 + 关键字检索，然后融合结果

    工作流程：
        第 1 步：准备一个小型文档库（5 条文档）
        第 2 步：分别展示向量检索和 BM25 关键字检索各自的 Top-3 打分
        第 3 步：用加权平均公式 fused = α × vector_score + (1-α) × bm25_score
                融合两种分数，得到最终排名

    ⚠️ 本示例使用模拟分数，目的是让学生先理解"融合"的概念，
       再去示例 2/3 中学习真实的 RRF 融合和 HybridSearcher 封装。
    """
    print(f"\n-- 示例 1: 简单混合检索")

    # ── 1. 准备文档库 ──
    # 选择 5 条 AI 主题文档，覆盖不同子领域
    # 索引 0 和 1 都包含"机器学习"关键词，是本次查询的相关文档
    documents = [
        "机器学习是人工智能的核心技术，通过数据训练模型。",       # idx=0: 含"机器学习"
        "深度学习使用神经网络，是机器学习的重要分支。",             # idx=1: 含"机器学习"
        "自然语言处理让计算机理解人类语言，属于 AI 领域。",         # idx=2
        "计算机视觉让计算机看懂图像，应用包括人脸识别。",           # idx=3
        "推荐系统根据用户行为推荐内容，使用协同过滤算法。",         # idx=4
    ]

    # 打印文档库内容
    print("文档库：")
    for i, doc in enumerate(documents):
        print(f"  [{i+1}] {doc}")
    print()

    # ── 2. 设置查询词 ──
    # 选择"机器学习"作为查询，文档 0 和 1 都包含该词
    query = "机器学习"
    print(f"查询：'{query}'")
    print()

    # ── 3. 模拟两种检索方法的打分 ──
    # ⚠️ 实际项目中，这些分数来自真实的 Embedding 相似度计算和 BM25 算法
    #    这里用预定义分数简化演示，让学生聚焦"融合逻辑"而非检索实现

    # 向量检索分数（余弦相似度，范围 0~1）
    # 文档 0 最相关（语义最接近），文档 1 次之（同属 AI 领域但语义稍远）
    vector_scores = [0.9, 0.85, 0.5, 0.3, 0.2]
    # BM25 关键字检索分数
    # 文档 0 和 1 都包含"机器学习"关键词，分数高；其他文档不含该词，分数极低
    bm25_scores = [0.95, 0.88, 0.1, 0.05, 0.1]

    # ── 4. 分别展示各方法的检索结果 ──
    print("1. 各方法单独检索结果：")
    print("-" * 50)

    # 向量检索 Top-3（按分数降序排序，取前 3）
    # enumerate(vector_scores) → (0, 0.9), (1, 0.85), (2, 0.5), (3, 0.3), (4, 0.2)
    # sorted(reverse=True)     → [(0, 0.9), (1, 0.85), (2, 0.5), ...]
    print("向量检索 Top-3:")
    vector_ranked = sorted(enumerate(vector_scores), key=lambda x: x[1], reverse=True)[:3]
    for doc_idx, score in vector_ranked:
        print(f"  [{score:.2f}] {documents[doc_idx][:30]}...")

    # 关键字检索 Top-3（同样按分数降序排序，取前 3）
    print("\n关键字检索 Top-3:")
    bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:3]
    for doc_idx, score in bm25_ranked:
        print(f"  [{score:.2f}] {documents[doc_idx][:30]}...")

    # ── 5. 融合结果：加权平均 ──
    # 公式：fused_score[i] = α × vector_score[i] + (1-α) × bm25_score[i]
    # 这里 α = 0.6（向量权重 60%），1-α = 0.4（关键字权重 40%）
    #
    # 为什么用加权平均而不是直接相加？
    #   → 向量检索分数和 BM25 分数的量纲不同，加权平均可以调节两者的相对重要性
    #   → α 越大，向量检索的语义理解影响越大
    #   → α 越小，关键字的精确匹配影响越大
    print("\n2. 融合结果（加权平均：向量 0.6 + 关键字 0.4）：")
    print("-" * 50)

    # 计算每篇文档的融合分数
    # 示例：文档 0: 0.6 × 0.9 + 0.4 × 0.95 = 0.54 + 0.38 = 0.92
    fused_scores = []
    for i in range(len(documents)):
        fused = 0.6 * vector_scores[i] + 0.4 * bm25_scores[i]
        fused_scores.append((i, fused))

    # 按融合分数降序排序，取 Top-3 作为最终结果
    fused_ranked = sorted(fused_scores, key=lambda x: x[1], reverse=True)[:3]
    for doc_idx, score in fused_ranked:
        print(f"  [{score:.2f}] {documents[doc_idx][:30]}...")


# =============================================================================
# RRF（倒数排名融合）
# =============================================================================

def rrf_fusion():
    """RRF（Reciprocal Rank Fusion）倒数排名融合"""
    print(f"\n-- 示例 2: RRF 倒数排名融合")

    documents = [
        "机器学习是人工智能的核心技术。",
        "深度学习使用神经网络。",
        "自然语言处理是 AI 的重要应用。",
        "计算机视觉用于图像识别。",
        "推荐系统使用协同过滤算法。",
        "知识图谱用图结构表示知识。",
    ]

    # 模拟两种检索的排名
    vector_ranking = [0, 1, 2, 4, 3, 5]
    keyword_ranking = [0, 2, 1, 5, 3, 4]

    print(f"模拟排名：")
    print(f"  向量检索：{[documents[i][:10] for i in vector_ranking[:3]]}...")
    print(f"  关键字检索：{[documents[i][:10] for i in keyword_ranking[:3]]}...")
    print()

    k = 60
    print(f"RRF 公式：score = 1 / ({k} + rank)")
    print("-" * 50)

    rrf_scores = {}
    for doc_idx in range(len(documents)):
        vector_rank = vector_ranking.index(doc_idx) + 1
        keyword_rank = keyword_ranking.index(doc_idx) + 1
        rrf_score = (1 / (k + vector_rank)) + (1 / (k + keyword_rank))
        rrf_scores[doc_idx] = rrf_score

        print(f"文档{doc_idx}: 向量排名={vector_rank}, 关键字排名={keyword_rank}")
        print(f"         RRF 分数 = {rrf_score:.6f}")

    print("\nRRF 融合后的最终排名：")
    print("-" * 50)
    final_ranking = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (doc_idx, score) in enumerate(final_ranking, 1):
        print(f"  [{rank}] 分数：{score:.6f} | {documents[doc_idx][:30]}...")

    print("\nRRF 优势：")
    print("   - 不受原始分数影响，只看排名")
    print("   - 不同检索方法的分数可能不可比，但排名可比")
    print("   - k 值调节：k 大→各方法权重接近；k 小→排名靠前优势大")



def hybrid_search_best_practices():
    """混合检索最佳实践"""
    print(f"\n-- 示例 4: 混合检索最佳实践")

    print("""
┌─────────────────────────────────────────────────────────┐
│ 混合检索最佳实践                                        │
├─────────────────────────────────────────────────────────┤
│ 1. 检索方法组合                                         │
│    推荐：向量检索 + BM25 关键字检索                       │
│    可选 + 标量过滤（类别、时间等）                      │
│                                                         │
│ 2. 结果融合方法                                         │
│    - RRF（推荐）：不受原始分数影响，稳定                │
│    - 加权平均：需要调优权重，灵活                       │
│                                                         │
│ 3. 权重设置                                             │
│    - 通用场景：向量 0.5 : 关键字 0.5                    │
│    - 语义理解：向量 0.7 : 关键字 0.3                    │
│    - 精确匹配：向量 0.3 : 关键字 0.7                    │
│                                                         │
│ 4. RRF 参数 k                                            │
│    - k=60（默认，适用于大多数场景）                     │
│    - k 小（20-40）：排名靠前优势更大                     │
│    - k 大（80-100）：各方法权重更接近                    │
│                                                         │
│ 5. 后处理                                               │
│    - Rerank 重排序：用更强模型对 Top-K 精排             │
│    - 去重：移除内容高度相似的结果                       │
│    - 多样性：确保结果覆盖不同角度                       │
└─────────────────────────────────────────────────────────┘
""")


def milvus_hybrid_search_demo():
    """Milvus 原生混合检索 — 稠密向量 + BM25 稀疏向量 + RRF/加权融合


    工作流程：
        第 1 步：用 AliyunEmbedding 生成稠密向量（text-embedding-v4，1024 维）
        第 2 步：创建同时含稠密 + 稀疏双向量字段的 Collection
                （BM25 Function 自动把 text 转成稀疏向量）
        第 3 步：插入数据（稠密向量手动提供，稀疏向量由 Milvus 自动生成）
        第 4 步：分别跑 纯向量 / 纯 BM25 / RRF 混合 / 加权混合，对比结果

    需要：Milvus 服务运行中 + ALIYUN_API_KEY（用于生成稠密向量）
    """
    from pymilvus import MilvusClient
    from rag_examples.embedding_examples.aliyun_embedding import AliyunEmbedding
    from rag_examples.milvus_config import MILVUS_URI, MILVUS_DB_NAME
    from rag_examples.utils.hybrid_retriever import HybridRetriever

    print(f"\n-- 示例 5: Milvus 原生混合检索")

    collection_name = "hybrid_demo"
    client = MilvusClient(uri=MILVUS_URI, db_name=MILVUS_DB_NAME)
    emb = AliyunEmbedding()
    retriever = HybridRetriever(client, collection_name, emb)

    # ── 1. 准备文档 + 生成稠密向量 ──
    documents = [
        "机器学习是人工智能的核心技术，通过数据训练模型预测。",
        "深度学习使用多层神经网络，是机器学习的重要分支。",
        "自然语言处理让计算机理解和生成人类语言。",
        "计算机视觉用于图像识别、人脸识别、自动驾驶。",
        "推荐系统根据用户历史行为推荐相关内容。",
        "知识图谱用图结构存储和表示领域知识。",
    ]

    print(f"\n  第 1 步：生成稠密向量（text-embedding-v4）")
    dense_vectors = [emb.embed(doc) for doc in documents]
    dim = len(dense_vectors[0])
    print(f"  ✓ 已为 {len(documents)} 篇文档生成 {dim} 维稠密向量")

    # ── 2. 创建 Collection（复用 HybridRetriever）──
    print(f"\n  第 2 步：创建 Collection（dense + sparse 双向量字段）")
    retriever.create_collection(dim=dim)
    print(f"  ✓ Collection '{collection_name}' 创建成功")

    # ── 3. 插入数据 ──
    print(f"\n  第 3 步：插入数据（dense 手动提供，sparse 由 BM25 自动生成）")
    n = retriever.insert(documents, dense_vectors)
    print(f"  ✓ 插入 {n} 篇文档并 flush 完成")

    # ── 4. 检索对比 ──
    query = "机器学习"
    query_vec = emb.embed(query)
    print(f"\n  查询：'{query}'\n")

    # 4.1 纯向量检索（语义）
    print("  [1] 纯向量检索（语义，anns_field='dense'）：")
    for i, hit in enumerate(retriever.search_dense(query_vec, top_k=3), 1):
        print(f"      [{i}] {hit['distance']:.3f}  {hit['entity']['text'][:28]}")

    # 4.2 纯 BM25 关键字检索（精确匹配）
    print("\n  [2] 纯 BM25 关键字检索（精确匹配，anns_field='sparse'）：")
    for i, hit in enumerate(retriever.search_sparse(query, top_k=3), 1):
        print(f"      [{i}] {hit['distance']:.3f}  {hit['entity']['text'][:28]}")

    # 4.3 RRF 混合检索（只看排名，不依赖分数量纲）
    print("\n  [3] RRF 混合检索（RRFRanker, k=60）：")
    for i, hit in enumerate(retriever.hybrid_search(query, query_vec, top_k=3, ranker="rrf"), 1):
        print(f"      [{i}] {hit['distance']:.3f}  {hit['entity']['text'][:28]}")

    # 4.4 加权融合（向量 0.6 + 关键字 0.4，可调权重）
    print("\n  [4] 加权混合检索（WeightedRanker, 向量 0.6 + 关键字 0.4）：")
    for i, hit in enumerate(retriever.hybrid_search(query, query_vec, top_k=3, ranker="weighted", vector_weight=0.6), 1):
        print(f"      [{i}] {hit['distance']:.3f}  {hit['entity']['text'][:28]}")

    print("""
  对比小结：
    - 纯向量：按语义相似度排序，可能召回不含查询词的文档
    - 纯 BM25：精确匹配查询词，语义相近但用词不同的文档召回不到
    - RRF：只看排名、不依赖分数量纲，能稳定兼顾两路结果
    - 加权：可调节向量/关键字的相对权重，需按业务场景调参
""")


# =============================================================================
# 主程序入口
# =============================================================================
# 运行顺序：
#   示例1 simple_hybrid_search()       → 理解混合检索的"加权平均"概念
#   示例2 rrf_fusion()                 → 理解 RRF 倒数排名融合的原理
#   示例3 demo_hybrid_searcher()       → 看封装好的 HybridSearcher 类的使用
#   示例4 hybrid_search_best_practices() → 了解生产环境的最佳实践
#   示例5 milvus_hybrid_search_demo()  → Milvus 原生混合检索（真实 API）
# =============================================================================

if __name__ == '__main__':

    # simple_hybrid_search()
    # rrf_fusion()
    # hybrid_search_best_practices()
    milvus_hybrid_search_demo()

