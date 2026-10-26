# =============================================================================
# 05_rerank — Rerank 重排序（Re-ranking）
# =============================================================================
# 用途：学习"召回 → 精排"两阶段检索，用 Rerank 模型提升检索精度
# 难度：⭐⭐⭐（3 星）
# =============================================================================
# 核心概念：
#   - 召回（Bi-Encoder）：query 与文档各编各的向量，快但粗
#   - 精排（Cross-Encoder）：query 与文档拼在一起深度交互，准但慢
#   - 两阶段：Bi 捞 Top-N 候选，Cross 精排出 Top-K 结果
# =============================================================================

import sys
from dotenv import load_dotenv
load_dotenv()

# Windows 控制台编码
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

_reranker = None  # 缓存 Rerank 模型，进程内只加载一次


def get_reranker():
    """加载中文 Rerank 模型（BAAI/bge-reranker-base）。进程内缓存，避免重复下载。"""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder('BAAI/bge-reranker-base')
    return _reranker


# =============================================================================
# 示例 1: 原理
# =============================================================================

def rerank_principle():
    """通俗讲解：为什么需要 Rerank（Bi-Encoder vs Cross-Encoder）"""
    print(f"\n-- 示例 1: 为什么需要 Rerank")
    print("""
一句话：检索用「快而粗」的模型先捞候选，Rerank 用「慢而准」的模型精排。

【类比：面试筛选】
  召回（Bi-Encoder）   ≈ 人事快速扫简历，先筛出 50 份「可能合适」的
  精排（Cross-Encoder）≈ 面试官逐份细读，最终定出「最合适」的 5 人

【根本区别】
  Bi-Encoder：query 和文档【各编各的向量】，再算余弦相似度
             → 两边编码时互不见面，只能隔空比划 → 快，但精度有限
  Cross-Encoder：把 query 和文档【拼成一句话】一起送进模型
             → 模型逐字交互，判断「文档到底答没答到问题」→ 准，但每对都要算一遍

【为什么不能只用一个】
  全用 Bi：百万文档也能快速过滤，但 Top-1 常跑偏
  全用 Cross：精度高，但百万对 (query, doc) 逐个过模型，算不动
  → 折中：Bi 召回 Top-50，Cross 精排 Top-5，又快又准
""")


# =============================================================================
# 示例 2: 真实 Rerank（中文）
# =============================================================================

def bge_reranker_demo():
    """真实 Rerank：用 BGE-Reranker 给候选文档重新打分"""
    print(f"\n-- 示例 2: 真实 Rerank（BGE-Reranker）")

    query = "机器学习需要什么基础？"
    # 从向量库中检索出的 top-N 候选文档
    candidates = [
        "深度学习是机器学习的子集，使用神经网络。",
        "机器学习通过训练数据让计算机自动学习。",
        "Python 是常用的编程语言，用于 AI 开发。",
        "数据结构和算法是编程的基础。",
        "机器学习需要数学基础，包括线性代数和概率统计。",
    ]

    try:
        reranker = get_reranker()
        # 注意：传 list[tuple]，不要传生成器
        scores = reranker.predict([(query, doc) for doc in candidates])
    except ImportError:
        print("需要安装：pip install sentence-transformers")
        return
    except Exception as e:
        print(f"模型加载/推理失败（多为网络无法下载模型）：{e}")
        print("提示：请手动下载 BAAI/bge-reranker-base，或用 HuggingFace 镜像。")
        return

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    print(f"查询：{query}\n")
    print("Rerank 结果（按相关性降序）：")
    print("-" * 50)
    for rank, (idx, score) in enumerate(ranked, 1):
        print(f"  [{rank}] {score:.4f}  {candidates[idx]}")


# =============================================================================
# 示例 3: 真实 Rerank（英文）
# =============================================================================

def cross_encoder_rerank():
    """真实 Rerank：英文场景用英文 CrossEncoder"""
    print(f"\n-- 示例 3: 真实 Rerank（CrossEncoder，英文场景）")

    query = "What basics are needed to learn machine learning?"
    candidates = [
        "Deep learning is a subset of machine learning that uses neural networks.",
        "Machine learning teaches computers to learn from data automatically.",
        "Python is a popular programming language used for AI development.",
        "Data structures and algorithms are the foundation of programming.",
        "Machine learning requires math foundations including linear algebra and probability.",
    ]

    try:
        from sentence_transformers import CrossEncoder
        # 英文文本配英文模型，勿用英文模型跑中文
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
        scores = model.predict([(query, doc) for doc in candidates])
    except ImportError:
        print("需要安装：pip install sentence-transformers")
        return
    except Exception as e:
        print(f"模型加载/推理失败（多为网络无法下载模型）：{e}")
        return

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    print(f"查询：{query}\n")
    print("Rerank 结果：")
    print("-" * 50)
    for rank, (idx, score) in enumerate(ranked, 1):
        print(f"  [{rank}] {score:.4f}  {candidates[idx]}")


# =============================================================================
# 示例 4: 两阶段实战（Milvus 召回 + Rerank 精排）
# =============================================================================

def milvus_rerank_pipeline():
    """两阶段实战：Milvus 召回 Top-N → Rerank 精排 Top-K

    需要：Milvus 服务运行中 + ALIYUN_API_KEY + sentence-transformers
    """
    from pymilvus import MilvusClient
    from rag_examples.milvus_config import MILVUS_URI, MILVUS_DB_NAME
    from rag_examples.utils.hybrid_retriever import HybridRetriever
    from rag_examples.embedding_examples.aliyun_embedding import AliyunEmbedding

    print(f"\n-- 示例 4: 两阶段实战（Milvus 召回 + Rerank 精排）")

    query = "机器学习需要什么基础？"
    documents = [
        "深度学习是机器学习的子集，使用多层神经网络。",
        "机器学习通过训练数据让计算机自动学习。",
        "Python 是常用的编程语言，广泛用于 AI 开发。",
        "数据结构和算法是编程的基础。",
        "机器学习需要数学基础，包括线性代数和概率统计。",
        "神经网络通过反向传播算法更新权重参数。",
    ]

    # 1. 建库（复用 04 的 HybridRetriever）
    client = MilvusClient(uri=MILVUS_URI, db_name=MILVUS_DB_NAME)
    emb = AliyunEmbedding()
    retriever = HybridRetriever(client, "rerank_demo", emb)
    dense_vectors = [emb.embed(d) for d in documents]
    retriever.create_collection(dim=len(dense_vectors[0]))
    retriever.insert(documents, dense_vectors)

    # 2. 召回：先多捞一些候选（Top-N）
    recall = retriever.retrieve(query, top_k=5)
    print(f"\n[召回阶段] 混合检索 Top-{len(recall)}：")
    for c in recall:
        print(f"  [{c['index']}] {c['text']}")

    # 3. 精排：Rerank 模型对候选重新打分
    try:
        reranker = get_reranker()
        scores = reranker.predict([(query, c['text']) for c in recall])
    except ImportError:
        print("\n需要安装：pip install sentence-transformers")
        return
    except Exception as e:
        print(f"\n模型加载/推理失败（多为网络无法下载模型）：{e}")
        return

    reranked = sorted(zip(recall, scores), key=lambda x: x[1], reverse=True)

    print(f"\n[精排阶段] Rerank 后 Top-3：")
    for rank, (c, score) in enumerate(reranked[:3], 1):
        print(f"  [{rank}] {score:.4f}  {c['text']}")

    print("\n[对比] 召回名次 → 精排名次：")
    recall_rank = {c['index']: i for i, c in enumerate(recall, 1)}
    for rank, (c, score) in enumerate(reranked, 1):
        old = recall_rank[c['index']]
        arrow = "↑" if rank < old else "↓" if rank > old else "="
        print(f"  {old}{arrow}{rank}  {c['text'][:24]}")


# =============================================================================
# 示例 5: 最佳实践
# =============================================================================

def rerank_best_practices():
    """Rerank 最佳实践"""
    print(f"\n-- 示例 5: Rerank 最佳实践")

    print("""
1. 何时使用 Rerank
   ✓ 对检索精度要求高的场景
   ✓ 候选集较小（<100 条）
   ✗ 实时性要求极高（Rerank 增加 100-500ms 延迟）
   ✗ 候选集太大（计算成本过高）

2. 模型选择
   中文：BAAI/bge-reranker-large（或更轻量的 -base / -v2-m3）
   英文：cross-encoder/ms-marco-MiniLM-L-12-v2
   API：Cohere Rerank API

3. 参数建议
   - 召回数量：50-100 条（给 Rerank 足够选择）
   - 返回数量：5-10 条（LLM 上下文限制）
   - 分数阈值：过滤掉 <0.3 的低相关结果

4. 性能优化
   - 用 fp16 推理省显存
   - 批量预测多对 (query, doc)
   - 缓存热点查询的 Rerank 结果

典型延迟：
  Bi-Encoder 检索 ~10ms | Cross-Encoder Rerank ~100-500ms | LLM 生成 ~1-5s
""")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  Rerank 重排序（Re-ranking）")
    print("=" * 70 + "\n")

    # rerank_principle()           # 原理，无依赖
    # bge_reranker_demo()          # 需要 sentence-transformers + 模型下载
    # cross_encoder_rerank()     # 需要 sentence-transformers + 模型下载
    milvus_rerank_pipeline()     # 需要 Milvus + ALIYUN_API_KEY + sentence-transformers
    # rerank_best_practices()
