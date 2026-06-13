# =============================================================================
# 03_keyword_search — 关键字检索（Keyword Search / BM25）
# =============================================================================
"""
本文件教学演示基于关键词的传统检索方法。

核心概念：
  - BM25 算法原理（TF、IDF、文档长度归一化）
  - 关键词匹配 vs 语义匹配
  - 倒排索引

对比维度：关键字检索（词匹配、精确、可解释）vs 向量检索（语义匹配、理解同义词）
适用场景：精确匹配（专有名词、人名、品牌名、代码/技术术语）
"""

import math
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# 示例 1: 简单关键词匹配
# =============================================================================

def simple_keyword_match():
    """最简单的关键词匹配演示：查找包含查询词的文档"""
    print(f"\n-- 示例 1: 简单关键词匹配")

    documents = [
        "人工智能是模拟人类智能的计算机科学领域。",
        "机器学习通过训练数据让计算机自动学习规律。",
        "深度学习是机器学习的子集，使用神经网络。",
        "自然语言处理让计算机理解人类语言。",
        "计算机视觉让计算机看懂图像和视频。",
    ]

    query = "机器学习"

    print(f"  查询：{query}")
    print(f"  文档数量：{len(documents)}")

    matched = []
    for i, doc in enumerate(documents):
        if query in doc:
            matched.append((i, doc))

    print("  匹配结果（包含关键词的文档）：")
    for i, doc in matched:
        print(f"  [{i+1}] {doc}")

    print("\n  问题：")
    print("  第 3 句'深度学习是机器学习的子集'也包含'机器学习'")
    print("  但如果是'AI 学习'、'训练模型'等同义词就匹配不到了")


# =============================================================================
# 示例 2: BM25 算法实现与演示
# =============================================================================

class SimpleBM25:
    """
    简化的 BM25 实现（用于教学演示）

    BM25 核心思想：
      1. TF（词频）：词在文档中出现越多越重要，但有边际效应递减
      2. IDF（逆文档频率）：在越少文档中出现的词越重要
      3. 文档长度归一化：长文档词频天然高，需要"打折"

    公式（简化版）：Score(Q, D) = Σ [IDF(qi) × TF(qi, D)]
    """

    def __init__(self, documents, k1=1.5, b=0.75):
        """
        参数:
            documents: 文档列表
            k1: 词频饱和度参数（默认 1.5，控制 TF 饱和度）
            b: 长度归一化参数（默认 0.75，0=不考虑长度，1=完全归一化）
        """
        self.documents = documents
        self.k1 = k1
        self.b = b

        # 预处理：分词（中文用简单字符分割，实际应使用 jieba 等分词工具）
        self.tokenized_docs = []
        for doc in documents:
            tokens = list(doc.lower())
            self.tokenized_docs.append(tokens)

        # 计算文档频率（DF）
        self.df = {}
        for tokens in self.tokenized_docs:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.df[token] = self.df.get(token, 0) + 1

        self.num_docs = len(documents)
        self.avg_doc_len = sum(len(t) for t in self.tokenized_docs) / self.num_docs

    def _idf(self, term):
        """计算 IDF：log((N - df + 0.5) / (df + 0.5) + 1)"""
        df = self.df.get(term, 0)
        return math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1)

    def _tf(self, term, doc_index):
        """计算 BM25 TF：freq * (k1+1) / (freq + k1 * (1 - b + b * doc_len / avg_doc_len))"""
        tokens = self.tokenized_docs[doc_index]
        freq = tokens.count(term)
        doc_len = len(tokens)

        numerator = freq * (self.k1 + 1)
        denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
        return numerator / denominator

    def search(self, query, top_k=3):
        """
        搜索与查询最相关的文档

        参数:
            query: 查询字符串
            top_k: 返回 Top-K 结果
        返回:
            (doc_index, score, doc_content) 列表
        """
        query_terms = list(query.lower())

        scores = []
        for i in range(len(self.documents)):
            score = 0
            for term in query_terms:
                idf = self._idf(term)
                tf = self._tf(term, i)
                score += idf * tf
            scores.append((i, score, self.documents[i]))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def demo_bm25_search():
    """演示使用 SimpleBM25 进行搜索"""
    print(f"\n-- 示例 2: BM25 算法演示")

    documents = [
        "人工智能是模拟人类智能的计算机科学领域，包括机器学习。",
        "机器学习通过训练数据让计算机自动学习规律和模式。",
        "深度学习是机器学习的子集，使用多层神经网络。",
        "自然语言处理是 AI 的重要分支，研究语言理解。",
        "计算机视觉让计算机能够看懂图像和视频内容。",
        "推荐系统根据用户历史行为推荐相关内容。",
    ]

    print("  文档库：")
    for i, doc in enumerate(documents):
        print(f"  [{i+1}] {doc}")
    print()

    bm25 = SimpleBM25(documents)

    queries = ["机器学习", "人工智能", "深度学习"]

    for query in queries:
        print(f"  查询：'{query}'")
        results = bm25.search(query, top_k=3)

        for i, (doc_idx, score, doc) in enumerate(results):
            print(f"  [{i+1}] 分数：{score:.4f}")
            print(f"      内容：{doc}")
        print()


# =============================================================================
# 示例 3: 使用 rank-bm25 库
# =============================================================================

def bm25_with_library():
    """使用 rank-bm25 库 + jieba 分词进行 BM25 检索"""
    print(f"\n-- 示例 3: 使用 rank-bm25 库")

    try:
        from rank_bm25 import BM25Okapi
        import jieba

        documents = [
            "人工智能是模拟人类智能的计算机科学领域，包括机器学习、深度学习等技术。",
            "机器学习通过训练数据让计算机自动学习规律，无需显式编程。",
            "深度学习是机器学习的子集，使用多层神经网络模拟人脑。",
            "自然语言处理让计算机理解和生成人类语言，应用广泛。",
            "计算机视觉让计算机能够看懂图像和视频，用于人脸识别。",
        ]

        print("  文档库：")
        for i, doc in enumerate(documents):
            print(f"  [{i+1}] {doc}")
        print()

        print("  正在进行中文分词...")
        tokenized_docs = [list(jieba.cut(doc)) for doc in documents]
        print("  ✓ 分词完成\n")

        bm25 = BM25Okapi(tokenized_docs)

        queries = [
            "机器学习是什么",
            "深度学习和神经网络",
            "自然语言处理应用"
        ]

        for query in queries:
            print(f"  查询：'{query}'")
            tokenized_query = list(jieba.cut(query))
            scores = bm25.get_scores(tokenized_query)
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

            for i, (doc_idx, score) in enumerate(ranked[:3]):
                print(f"  [{i+1}] 分数：{score:.4f}")
                print(f"      内容：{documents[doc_idx][:50]}...")
            print()

    except ImportError as e:
        print(f"  需要安装：pip install rank-bm25 jieba")
        print(f"  当前环境未安装依赖：{e}")


# =============================================================================
# 示例 4: 关键字检索 vs 向量检索对比
# =============================================================================

def keyword_vs_vector_search():
    """对比关键字检索（字面匹配）和向量检索（语义匹配）的差异"""
    print(f"\n-- 示例 4: 关键字检索 vs 向量检索")

    documents = [
        "机器学习是人工智能的核心技术，通过数据训练模型。",
        "深度学习使用神经网络，是 ML 的重要分支。",
        "AI 模型通过训练可以自动学习规律和模式。",
        "神经网络的训练需要大量数据和计算资源。",
    ]

    print("  文档库：")
    for i, doc in enumerate(documents):
        print(f"  [{i+1}] {doc}")
    print()

    query = "AI 学习"
    print(f"  查询：'{query}'（与'机器学习'语义相近，但字面不同）")
    print()

    # 关键字检索结果
    print("  1. 关键字检索结果：")
    print("     （查找包含'A'、'I'、'学'、'习'字符的文档）")
    matched_kw = [doc for doc in documents if 'A' in doc or 'I' in doc or '学习' in doc]
    for i, doc in enumerate(matched_kw):
        print(f"     - {doc[:40]}...")

    print()

    # 向量检索结果（模拟）
    print("  2. 向量检索结果（模拟）：")
    print("     （理解'AI 学习'的语义，找到相关文档）")
    vector_results = [
        (0, 0.85),
        (2, 0.78),
        (1, 0.65),
    ]
    for doc_idx, score in vector_results:
        print(f"     [{score:.2f}] {documents[doc_idx][:40]}...")

    print("\n  结论：")
    print("     关键字检索：精确匹配字面，无法理解同义词")
    print("     向量检索：理解语义，能找到相关但字面不同的内容")
    print("     → 混合检索结合两者优势")


# =============================================================================
# 示例 5: BM25 参数调优
# =============================================================================

def bm25_parameter_tuning():
    """演示 BM25 中 k1（词频饱和度）和 b（长度归一化）两个参数的影响"""
    print(f"\n-- 示例 5: BM25 参数调优")

    documents = [
        "机器学习是人工智能的核心技术。",
        "机器学习通过数据训练模型，机器学习应用广泛。",
        "深度学习也是 ML 的一种形式。",
    ]

    query = "机器学习"

    print(f"  查询：'{query}'")
    print("  文档库：")
    for i, doc in enumerate(documents):
        print(f"  [{i+1}] {doc}（{len(doc)}字）")
    print()

    print("  不同 k1 值的影响（b=0.75 固定）：")
    for k1 in [0.5, 1.2, 2.0]:
        bm25 = SimpleBM25(documents, k1=k1, b=0.75)
        results = bm25.search(query, top_k=3)

        print(f"\n  k1={k1}:")
        for doc_idx, score, _ in results:
            print(f"  文档{doc_idx+1}: {score:.4f}")

    print("""
  参数说明：
     k1: 控制词频饱和度
         - k1 小：词频影响小，几次出现后分数不再显著增加
         - k1 大：词频影响大，多次出现会持续增加分数
         - 推荐值：1.2-2.0

     b: 控制长度归一化
         - b=0: 不考虑文档长度
         - b=1: 完全归一化
         - 推荐值：0.5-0.8
""")


# =============================================================================
# 示例 6: 关键字检索最佳实践
# =============================================================================

def keyword_search_best_practices():
    """关键字检索的最佳实践总结"""
    print(f"\n-- 示例 6: 关键字检索最佳实践")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │ 关键字检索最佳实践                                      │
  ├─────────────────────────────────────────────────────────┤
  │                                                         │
  │ 1. 适用场景                                             │
  │    ✓ 专有名词检索（人名、地名、品牌）                  │
  │    ✓ 代码/技术术语检索                                   │
  │    ✓ 精确匹配需求                                        │
  │    ✗ 语义理解需求（应用向量检索）                      │
  │                                                         │
  │ 2. 中文分词选择                                         │
  │    - jieba: 轻量级，适合通用场景                        │
  │    - HanLP: 功能丰富，支持实体识别                      │
  │    - THULAC: 清华出品，学术场景                         │
  │                                                         │
  │ 3. 优化技巧                                             │
  │    - 同义词扩展：查询"AI"时同时查"人工智能"             │
  │    - 词干提取：英文场景下还原词根                        │
  │    - 停用词过滤：去除"的"、"是"等无用词                 │
  │                                                         │
  │ 4. BM25 参数建议                                         │
  │    - k1: 1.2-2.0（默认 1.5）                            │
  │    - b: 0.5-0.8（默认 0.75）                            │
  │                                                         │
  └─────────────────────────────────────────────────────────┘

  混合检索策略（推荐）：

  ```python
  # 1. 分别检索
  keyword_results = bm25.search(query, top_k=10)
  vector_results = vector_search(query, top_k=10)

  # 2. 融合结果（RRF 倒数排名融合）
  final_results = reciprocal_rank_fusion(
      keyword_results,
      vector_results,
      k=60
  )

  # 3. 返回 Top-5
  return final_results[:5]
  ```
""")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  关键字检索（Keyword Search / BM25）")
    print("=" * 70 + "\n")

    simple_keyword_match()
    demo_bm25_search()
    bm25_with_library()
    keyword_vs_vector_search()
    bm25_parameter_tuning()
    keyword_search_best_practices()
