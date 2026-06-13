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

def simple_hybrid_search():
    """简单的混合检索演示 — 向量检索 + 关键字检索，然后融合结果"""
    print(f"\n-- 示例 1: 简单混合检索")

    documents = [
        "机器学习是人工智能的核心技术，通过数据训练模型。",
        "深度学习使用神经网络，是机器学习的重要分支。",
        "自然语言处理让计算机理解人类语言，属于 AI 领域。",
        "计算机视觉让计算机看懂图像，应用包括人脸识别。",
        "推荐系统根据用户行为推荐内容，使用协同过滤算法。",
    ]

    print("文档库：")
    for i, doc in enumerate(documents):
        print(f"  [{i+1}] {doc}")
    print()

    query = "机器学习"
    print(f"查询：'{query}'")
    print()

    # 模拟向量检索分数
    vector_scores = [0.9, 0.85, 0.5, 0.3, 0.2]
    # 模拟 BM25 分数
    bm25_scores = [0.95, 0.88, 0.1, 0.05, 0.1]

    print("1. 各方法单独检索结果：")
    print("-" * 50)
    print("向量检索 Top-3:")
    vector_ranked = sorted(enumerate(vector_scores), key=lambda x: x[1], reverse=True)[:3]
    for doc_idx, score in vector_ranked:
        print(f"  [{score:.2f}] {documents[doc_idx][:30]}...")

    print("\n关键字检索 Top-3:")
    bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:3]
    for doc_idx, score in bm25_ranked:
        print(f"  [{score:.2f}] {documents[doc_idx][:30]}...")

    # 融合：加权平均
    print("\n2. 融合结果（加权平均：向量 0.6 + 关键字 0.4）：")
    print("-" * 50)

    fused_scores = []
    for i in range(len(documents)):
        fused = 0.6 * vector_scores[i] + 0.4 * bm25_scores[i]
        fused_scores.append((i, fused))

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


def demo_hybrid_searcher():
    """演示混合检索器"""
    print(f"\n-- 示例 3: 混合检索器封装")

    documents = [
        "机器学习是人工智能的核心技术，通过数据训练模型预测。",
        "深度学习使用多层神经网络，是机器学习的重要分支。",
        "自然语言处理让计算机理解和生成人类语言。",
        "计算机视觉用于图像识别、人脸识别、自动驾驶。",
        "推荐系统根据用户历史行为推荐相关内容。",
        "知识图谱用图结构存储和表示领域知识。",
    ]

    print("文档库：")
    for i, doc in enumerate(documents):
        print(f"  [{i+1}] {doc[:30]}...")
    print()

    searcher = HybridSearcher(documents)
    query = "机器学习"
    print(f"查询：'{query}'")
    print()

    print("1. RRF 融合结果：")
    print("-" * 50)
    results = searcher.search(query, top_k=4, method="rrf")
    for i, (idx, score, doc) in enumerate(results):
        print(f"  [{i+1}] 分数：{score:.6f}")
        print(f"      {doc}")
    print()

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


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  混合检索（Hybrid Search）")
    print("=" * 70 + "\n")

    simple_hybrid_search()
    rrf_fusion()
    demo_hybrid_searcher()
    hybrid_search_best_practices()
    milvus_hybrid_search_info()

    print("\n" + "=" * 70)
    print("  下一步：05_rerank.py（Rerank 重排序）")
    print("=" * 70 + "\n")
