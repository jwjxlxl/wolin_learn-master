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


# =============================================================================
# HybridSearcher 类
# =============================================================================

class HybridSearcher:
    """混合检索器封装 — 支持向量检索 + BM25 关键字检索 + 结果融合"""

    def __init__(self, documents):
        self.documents = documents

        random.seed(42)
        self.vectors = []
        for doc in documents:
            vec = self._mock_embedding(doc)
            self.vectors.append(vec)

        self._build_bm25_index()

    def _mock_embedding(self, text):
        """模拟 Embedding（实际应该用真实模型）"""
        random.seed(hash(text) % 10000)
        vec = [random.uniform(-0.5, 0.5) for _ in range(128)]

        if "机器学习" in text:
            vec[0:20] = [0.8] * 20
        elif "深度学习" in text:
            vec[20:40] = [0.8] * 20
        elif "自然语言" in text:
            vec[40:60] = [0.8] * 20

        return vec

    def _build_bm25_index(self):
        """构建 BM25 索引（简化版）"""
        self.keyword_index = {}
        for i, doc in enumerate(self.documents):
            for char in set(doc.lower()):
                if char not in self.keyword_index:
                    self.keyword_index[char] = []
                self.keyword_index[char].append(i)

    def _vector_search(self, query_vector, top_k):
        """向量检索"""
        scores = []
        for i, vec in enumerate(self.vectors):
            dot = sum(a * b for a, b in zip(query_vector, vec))
            norm_q = sum(a * a for a in query_vector) ** 0.5
            norm_d = sum(a * a for a in vec) ** 0.5
            score = dot / (norm_q * norm_d) if norm_q > 0 and norm_d > 0 else 0
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _keyword_search(self, query, top_k):
        """关键字检索（简化版）"""
        scores = [0] * len(self.documents)
        for char in query.lower():
            if char in self.keyword_index:
                for doc_idx in self.keyword_index[char]:
                    scores[doc_idx] += 1

        max_score = max(scores) if scores else 1
        scores = [s / max_score for s in scores]
        scored = list(enumerate(scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def search(self, query, query_vector=None, top_k=5, method="rrf", alpha=0.5):
        """混合检索

        参数:
            query: 查询文本
            query_vector: 查询向量（可选）
            top_k: 返回结果数
            method: 融合方法 ("rrf" 或 "weighted")
            alpha: 向量检索权重（0-1）
        """
        if query_vector is None:
            random.seed(hash(query) % 10000)
            query_vector = [random.uniform(-0.5, 0.5) for _ in range(128)]
            if "机器学习" in query:
                query_vector[0:20] = [0.8] * 20

        vector_results = self._vector_search(query_vector, top_k * 2)
        keyword_results = self._keyword_search(query, top_k * 2)

        if method == "rrf":
            return self._rrf_fusion(vector_results, keyword_results, top_k)
        else:
            return self._weighted_fusion(vector_results, keyword_results, top_k, alpha)

    def _rrf_fusion(self, vector_results, keyword_results, top_k):
        """RRF 融合"""
        k = 60
        vector_rank = {doc_idx: rank + 1 for rank, (doc_idx, _) in enumerate(vector_results)}
        keyword_rank = {doc_idx: rank + 1 for rank, (doc_idx, _) in enumerate(keyword_results)}

        all_docs = set(vector_rank.keys()) | set(keyword_rank.keys())
        rrf_scores = {}
        for doc_idx in all_docs:
            v_rank = vector_rank.get(doc_idx, len(vector_results) + 1)
            k_rank = keyword_rank.get(doc_idx, len(keyword_results) + 1)
            score = 1 / (k + v_rank) + 1 / (k + k_rank)
            rrf_scores[doc_idx] = score

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(idx, score, self.documents[idx]) for idx, score in ranked]

    def _weighted_fusion(self, vector_results, keyword_results, top_k, alpha):
        """加权融合"""
        vector_scores = {doc_idx: score for doc_idx, score in vector_results}
        keyword_scores = {doc_idx: score for doc_idx, score in keyword_results}

        max_v = max(vector_scores.values()) if vector_scores else 1
        max_k = max(keyword_scores.values()) if keyword_scores else 1

        all_docs = set(vector_scores.keys()) | set(keyword_scores.keys())
        fused_scores = {}
        for doc_idx in all_docs:
            v_score = vector_scores.get(doc_idx, 0) / max_v
            k_score = keyword_scores.get(doc_idx, 0) / max_k
            fused = alpha * v_score + (1 - alpha) * k_score
            fused_scores[doc_idx] = fused

        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(idx, score, self.documents[idx]) for idx, score in ranked]


# =============================================================================
# 混合检索器演示
# =============================================================================
# 作用：演示如何使用 HybridSearcher 封装类进行真实混合检索
#       1. 创建 HybridSearcher 实例（内置模拟 Embedding + 关键字索引）
#       2. 用 RRF 融合方法检索
#       3. 用加权融合方法检索
#       4. 对比两种融合方法的差异
# 教学目的：让学生看到一个封装好的混合检索器的使用方式
#           对比 RRF vs 加权两种融合策略的效果
# =============================================================================

def demo_hybrid_searcher():
    """演示混合检索器 — 使用 HybridSearcher 类封装的真实混合检索

    工作流程：
        第 1 步：准备文档库（6 条 AI 主题文档）
        第 2 步：创建 HybridSearcher 实例
                → 内部自动为每条文档生成模拟 Embedding 向量
                → 内部自动构建字符级关键字倒排索引
        第 3 步：使用 RRF（倒数排名融合）方法检索，返回 Top 4
        第 4 步：使用加权融合方法检索（向量权重 0.7 + 关键字权重 0.3）
        第 5 步：对比两种方法的结果差异

    ⚠️ HybridSearcher 使用模拟向量（_mock_embedding），
       实际项目中应替换为真实的 Embedding 模型和 BM25 库。
    """
    print(f"\n-- 示例 3: 混合检索器封装")

    # ── 1. 准备文档库 ──
    # 6 条文档覆盖 AI 的不同子领域
    # 文档 0 和 1 包含"机器学习"关键词，是查询"机器学习"的相关文档
    documents = [
        "机器学习是人工智能的核心技术，通过数据训练模型预测。",       # idx=0: 含"机器学习"
        "深度学习使用多层神经网络，是机器学习的重要分支。",             # idx=1: 含"机器学习"
        "自然语言处理让计算机理解和生成人类语言。",                   # idx=2
        "计算机视觉用于图像识别、人脸识别、自动驾驶。",                # idx=3
        "推荐系统根据用户历史行为推荐相关内容。",                      # idx=4
        "知识图谱用图结构存储和表示领域知识。",                        # idx=5
    ]

    # 打印文档库概览
    print("文档库：")
    for i, doc in enumerate(documents):
        print(f"  [{i+1}] {doc[:30]}...")
    print()

    # ── 2. 创建混合检索器 ──
    # HybridSearcher 构造函数内部做了两件事：
    #   a) 为每条文档生成 128 维模拟 Embedding 向量（_mock_embedding）
    #      → "机器学习"相关文档的 vec[0:20] = [0.8] * 20
    #      → "深度学习"相关文档的 vec[20:40] = [0.8] * 20
    #      → "自然语言"相关文档的 vec[40:60] = [0.8] * 20
    #   b) 构建字符级倒排索引（_build_bm25_index）
    #      → 每个唯一字符映射到包含该字符的文档 ID 列表
    searcher = HybridSearcher(documents)

    # ── 3. 设置查询词 ──
    query = "机器学习"
    print(f"查询：'{query}'")
    print()

    # ── 4. RRF 融合检索 ──
    # method="rrf"：使用倒数排名融合（Reciprocal Rank Fusion）
    #   → 公式：score = 1/(k + vector_rank) + 1/(k + keyword_rank)
    #   → k=60（默认），只看排名，不受原始分数影响
    #   → 优势：不同检索方法的分数可能不可比，但排名可比
    print("1. RRF 融合结果：")
    print("-" * 50)
    results = searcher.search(query, top_k=4, method="rrf")
    for i, (idx, score, doc) in enumerate(results):
        print(f"  [{i+1}] 分数：{score:.6f}")
        print(f"      {doc}")
    print()

    # ── 5. 加权融合检索 ──
    # method="weighted"：使用加权平均
    #   → 公式：fused = α × vector_score_normalized + (1-α) × keyword_score_normalized
    #   → alpha=0.7：向量检索权重 70%，关键字权重 30%
    #   → 先归一化：各自除以该方法内最大分数，统一到 0~1 范围
    #   → 优势：可以灵活调节两种方法的相对重要性
    #   → 注意：权重设置需要根据业务场景调优
    print("2. 加权融合结果（向量 0.7 + 关键字 0.3）：")
    print("-" * 50)
    results = searcher.search(query, top_k=4, method="weighted", alpha=0.7)
    for i, (idx, score, doc) in enumerate(results):
        print(f"  [{i+1}] 分数：{score:.6f}")
        print(f"      {doc}")


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


def milvus_hybrid_search_info():
    """Milvus 混合检索说明"""
    print(f"\n-- 示例 5: Milvus 混合检索")

    print("""
Milvus 2.3+ 版本开始支持原生混合检索（hybrid_search API）：

  - AnnSearchRequest：向量检索请求
  - FullTextSearchRequest：全文检索请求
  - WeightedRanker / RRFRanker：结果融合策略

使用混合检索需要：
  1. Collection 支持全文索引
  2. pymilvus 2.3+ 版本
  3. 如需兼容旧版本，可在应用层手动实现融合

推荐查阅 Milvus 官方文档获取最新 API 用法。
""")


# =============================================================================
# 主程序入口
# =============================================================================
# 运行顺序：
#   示例1 simple_hybrid_search()     → 理解混合检索的"加权平均"概念
#   示例2 rrf_fusion()               → 理解 RRF 倒数排名融合的原理
#   示例3 demo_hybrid_searcher()     → 看封装好的 HybridSearcher 类的使用
#   示例4 hybrid_search_best_practices()  → 了解生产环境的最佳实践
#   示例5 milvus_hybrid_search_info()     → 了解 Milvus 原生混合检索 API
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  混合检索（Hybrid Search）")
    print("=" * 70 + "\n")

    # simple_hybrid_search()
    # rrf_fusion()
    # demo_hybrid_searcher()
    hybrid_search_best_practices()
    # milvus_hybrid_search_info()

