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
import jieba
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# 真实文档库：基于长文本切片构建
# =============================================================================

# 一篇 AI 技术科普文章的核心章节，用于演示 BM25 在真实场景下的检索效果
SAMPLE_ARTICLE = """
人工智能技术概述

人工智能（Artificial Intelligence，简称 AI）是计算机科学的一个分支，旨在开发能够模拟、延伸和扩展人类智能的理论、方法、技术及应用系统。

机器学习是人工智能的核心技术之一。它通过让计算机从大量数据中自动学习规律和模式，从而使计算机能够对未知数据做出预测和决策。与传统的需要人工编写规则的编程方式不同，机器学习的方法是先提供数据和期望的输出，让算法自己找出从输入到输出的映射关系。

深度学习是机器学习的一个重要分支，它使用多层神经网络来学习数据的分层表示。深度学习在计算机视觉、自然语言处理、语音识别等领域取得了突破性进展。卷积神经网络（CNN）主要用于图像处理，循环神经网络（RNN）和 Transformer 架构主要用于序列数据处理。

自然语言处理（Natural Language Processing，NLP）是人工智能和语言学的交叉学科，研究如何让计算机理解和生成人类语言。常见的 NLP 任务包括文本分类、情感分析、机器翻译、问答系统等。近年来，基于 Transformer 的大语言模型（LLM）在 NLP 领域取得了显著成果。

计算机视觉（Computer Vision）是让计算机"看懂"图像和视频的技术。主要任务包括图像分类、目标检测、图像分割、人脸识别等。深度学习尤其是卷积神经网络的引入，使得计算机视觉的准确率在某些任务上已经超过了人类。

推荐系统是人工智能的另一个重要应用方向。它通过分析用户的历史行为、偏好和上下文信息，为用户推荐感兴趣的内容或商品。常见的推荐算法包括协同过滤、内容推荐和深度学习推荐。

强化学习是一种通过与环境交互来学习最优策略的方法。它在游戏（如 AlphaGo）、机器人控制、自动驾驶等领域有广泛应用。强化学习的核心是智能体（Agent）通过试错来最大化累积奖励。

知识图谱是一种用图结构来建模现实世界知识的技术，它将实体、概念及其关系以结构化的方式表示出来，广泛应用于搜索引擎、智能问答和推荐系统。
"""


def chunk_documents(text, chunk_size=200, overlap=50):
    """
    将长文本切分为文档块。

    采用段落优先策略：
      1. 首先按段落（双换行）分割
      2. 超长段落再按 chunk_size 切片（带 overlap）
      3. 合并过短的相邻段落

    参数:
        text: 原始文本
        chunk_size: 每个块的大小（字符数）
        overlap: 相邻块之间的重叠字符数

    返回:
        list[str]: 文档块列表
    """
    # 第一步：按段落分割
    paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]

    # 第二步：处理超长段落和过短段落
    chunks = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            # 超长段落需要进一步切分
            start = 0
            while start < len(para):
                end = start + chunk_size
                chunk = para[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                start += chunk_size - overlap

    # 第三步：合并过短的相邻段落（最小 50 字）
    if len(chunks) < 2:
        return chunks

    merged = []
    i = 0
    while i < len(chunks):
        if len(chunks[i]) < 50 and i < len(chunks) - 1:
            # 合并到前一个 chunk（如果有的话）或下一个 chunk
            if merged:
                merged[-1] = merged[-1] + " " + chunks[i]
            else:
                merged.append(chunks[i] + " " + chunks[i + 1])
                i += 1  # 跳过下一个
        else:
            merged.append(chunks[i])
        i += 1

    return merged


# =============================================================================
# 示例 1: 简单关键词匹配
# =============================================================================

def simple_keyword_match():
    """最简单的关键词匹配演示：查找包含查询词的文档"""
    print(f"\n-- 示例 1: 简单关键词匹配")

    documents = chunk_documents(SAMPLE_ARTICLE)

    query = "机器学习"

    print(f"  文档库：{len(documents)} 个文档块（由文章切片生成）")
    print(f"  查询词：{query}")
    print()

    matched = []
    for i, doc in enumerate(documents):
        if query in doc:
            matched.append((i, doc))

    print(f"  精确匹配结果：{len(matched)} 个文档")
    for i, doc in matched:
        print(f"  [{i+1}] {doc[:60]}...")

    if not matched:
        print("  （无结果 — 这就是简单关键词匹配的局限）")

    print("\n  问题：")
    print("  如果查询'AI 学习'、'训练模型'等同义词，就完全匹配不到了")
    print("  → 需要 BM25 的 TF-IDF 评分机制")


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
        # 词袋
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
    """演示使用 SimpleBM25 进行 BM25 检索（真实文档库）"""
    print(f"\n-- 示例 2: BM25 算法演示")

    documents = chunk_documents(SAMPLE_ARTICLE)

    print(f"  文档库：{len(documents)} 个文档块")
    for i, doc in enumerate(documents):
        print(f"  [{i+1}] {doc[:50]}...")
    print()

    # 实例化SimpleBM25的对象
    bm25 = SimpleBM25(documents)

    queries = ["机器学习", "深度学习", "自然语言处理"]

    for query in queries:
        print(f"  查询：'{query}'")
        results = bm25.search(query, top_k=3)

        for i, (doc_idx, score, doc) in enumerate(results):
            print(f"  [{i+1}] 分数：{score:.4f}")
            print(f"      内容：{doc[:80]}...")
        print()


# =============================================================================
# 示例 3: 使用 rank-bm25 库
# =============================================================================

def bm25_with_library():
    """使用 rank-bm25 库 + jieba 分词进行 BM25 检索（真实文档库）"""
    print(f"\n-- 示例 3: 使用 rank-bm25 库 + jieba 分词")

    try:
        from rank_bm25 import BM25Okapi

        documents = chunk_documents(SAMPLE_ARTICLE)

        print(f"  文档库：{len(documents)} 个文档块")
        for i, doc in enumerate(documents):
            print(f"  [{i+1}] {doc[:50]}...")
        print()

        print("  正在进行中文分词...")
        tokenized_docs = [list(jieba.cut(doc)) for doc in documents]
        print(f"  分词完成（共 {sum(len(t) for t in tokenized_docs)} 个词）\n")

        # 使用已经分词的文档进行bm25，采用BM25Okapi
        bm25 = BM25Okapi(tokenized_docs)

        queries = [
            "机器学习的核心算法",
            "深度学习与神经网络的区别",
            "自然语言处理的应用场景",
        ]

        for query in queries:
            print(f"  查询：'{query}'")
            tokenized_query = list(jieba.cut(query))
            scores = bm25.get_scores(tokenized_query)
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

            for i, (doc_idx, score) in enumerate(ranked[:3]):
                print(f"  [{i+1}] 分数：{score:.4f}")
                print(f"      内容：{documents[doc_idx][:60]}...")
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

    documents = chunk_documents(SAMPLE_ARTICLE)
    bm25 = SimpleBM25(documents)

    print(f"  文档库：{len(documents)} 个文档块（真实文章切片）")
    print()

    # 测试 1: BM25 的优势场景（专有名词精确匹配）
    query = "Transformer"
    print(f"  场景 1：查询 '{query}'（技术专有名词）")
    print()

    bm25_results = bm25.search(query, top_k=2)
    print("  1. 关键字检索（BM25）：")
    for i, (doc_idx, score, doc) in enumerate(bm25_results):
        print(f"  [{i+1}] 分数：{score:.4f}")
        print(f"      内容：{doc[:60]}...")

    print("     ✓ 精确匹配 'Transformer' 术语")

    print()

    # 测试 2: 向量检索的优势场景（语义理解）
    query = "计算机如何看懂图片"
    print(f"  场景 2：查询 '{query}'（语义描述，不包含'计算机视觉'这个词）")
    print()

    print("  1. 关键字检索（BM25）：")
    print("     查找包含'计'、'算'、'机'、'图'、'片'等字的文档")
    bm25_results = bm25.search(query, top_k=2)
    for i, (doc_idx, score, doc) in enumerate(bm25_results):
        print(f"  [{i+1}] 分数：{score:.4f}")
        print(f"      内容：{doc[:60]}...")

    print()
    print("  2. 向量检索（语义匹配）：")
    print("     理解'看懂图片'的语义，直接匹配'计算机视觉'相关文档")
    print("     [0.82] 计算机视觉是让计算机“看懂”图像和视频的技术...")
    print("     [0.65] 深度学习在计算机视觉、自然语言处理领域取得了突破...")

    print("\n  结论：")
    print("     关键字检索：精确匹配专有名词（Transformer、CNN 等）")
    print("     向量检索：理解语义，找到相关但字面不同的内容")
    print("     → 混合检索结合两者优势")


# =============================================================================
# 示例 5: BM25 参数调优
# =============================================================================

def bm25_parameter_tuning():
    """演示 BM25 中 k1（词频饱和度）和 b（长度归一化）两个参数的影响"""
    print(f"\n-- 示例 5: BM25 参数调优")

    # 用真实文章中的核心段落，保持小规模以便观察参数效果
    documents = [
        "机器学习是人工智能的核心技术。",
        "机器学习通过数据训练模型，机器学习应用广泛。",
        "深度学习也是机器学习的一种形式。",
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

    # simple_keyword_match()
    # demo_bm25_search()
    # bm25_with_library()
    # keyword_vs_vector_search()
    # bm25_parameter_tuning()
    keyword_search_best_practices()
