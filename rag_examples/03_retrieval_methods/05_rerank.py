# =============================================================================
# 05_rerank — Rerank 重排序（Re-ranking）
# =============================================================================
# 用途：学习对检索结果进行精排序，提升检索精度
# 难度：⭐⭐⭐（3 星）
# =============================================================================
# 核心概念：
#   - 召回 vs 精排：两阶段检索架构
#   - Cross-Encoder vs Bi-Encoder：精度与速度的权衡
#   - Rerank 模型：BGE-Reranker、CrossEncoder、Cohere Rerank
#
# 两级架构：
#   检索（Bi-Encoder，快）→ Top-50 候选 → Rerank（CrossEncoder，准）→ Top-5 最终
# =============================================================================

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# 模拟 Rerank 流程
# =============================================================================

def mock_rerank_pipeline():
    """模拟 Rerank 流程演示"""
    print(f"\n-- 示例 1: 模拟 Rerank 流程")

    query = "机器学习需要什么基础？"

    retrieved_docs = [
        (0.85, "深度学习是机器学习的子集，使用神经网络。"),
        (0.82, "机器学习通过训练数据让计算机自动学习。"),
        (0.78, "Python 是常用的编程语言，用于 AI 开发。"),
        (0.75, "数据结构和算法是编程的基础。"),
        (0.72, "机器学习需要数学基础，包括线性代数和概率。"),
    ]

    print(f"用户查询：{query}")
    print()

    print("1. 初步检索结果（按向量相似度排序）：")
    print("-" * 50)
    for i, (score, doc) in enumerate(retrieved_docs):
        print(f"  [{i+1}] 相似度：{score:.2f}")
        print(f"      {doc}")
    print()

    # 模拟 Rerank 分数
    rerank_scores = [0.45, 0.68, 0.32, 0.28, 0.92]

    reranked = list(zip(retrieved_docs, rerank_scores))
    reranked.sort(key=lambda x: x[1], reverse=True)

    print("2. Rerank 后结果（按相关性重新排序）：")
    print("-" * 50)
    for i, ((orig_score, doc), rerank_score) in enumerate(reranked):
        orig_idx = next(j for j, (s, d) in enumerate(retrieved_docs) if s == orig_score and d == doc)
        rank_change = "↑" if i < orig_idx else "↓" if i > orig_idx else "="
        print(f"  [{i+1}] Rerank 分数：{rerank_score:.2f} {rank_change}")
        print(f"      原相似度：{orig_score:.2f}")
        print(f"      {doc}")
    print()

    print("观察：")
    print("   - 第 5 句从第 5 名上升到第 1 名（直接回答问题）")
    print("   - 第 1 句从第 1 名下降到第 3 名（讲的是子集概念）")
    print("   - 这就是 Rerank 的价值：更精准理解 query-doc 相关性")


# =============================================================================
# BGE-Reranker 演示
# =============================================================================

def bge_reranker_demo():
    """使用 BGE-Reranker 进行重排序"""
    print(f"\n-- 示例 2: 使用 BGE-Reranker")

    try:
        from FlagEmbedding import FlagReranker

        query = "机器学习需要什么基础？"
        candidates = [
            "深度学习是机器学习的子集，使用神经网络。",
            "机器学习通过训练数据让计算机自动学习。",
            "Python 是常用的编程语言，用于 AI 开发。",
            "数据结构和算法是编程的基础。",
            "机器学习需要数学基础，包括线性代数和概率统计。",
        ]

        print(f"查询：{query}")
        print(f"候选文档：{len(candidates)} 条\n")

        print("加载 BGE-Reranker 模型...")
        reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=False)
        print("模型加载完成\n")

        print("计算相关性分数...")
        scores = reranker.compute_score([query, doc] for doc in candidates)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        print("\nRerank 结果：")
        print("-" * 50)
        for rank, (idx, score) in enumerate(ranked, 1):
            print(f"  [{rank}] 分数：{score:.4f}")
            print(f"      {candidates[idx]}")
            print()

    except ImportError:
        print("需要安装：pip install FlagEmbedding")
        print("\n模拟演示（无真实模型）：")
        print("-" * 50)

        candidates = [
            "深度学习是机器学习的子集，使用神经网络。",
            "机器学习通过训练数据让计算机自动学习。",
            "Python 是常用的编程语言，用于 AI 开发。",
            "数据结构和算法是编程的基础。",
            "机器学习需要数学基础，包括线性代数和概率统计。",
        ]

        mock_scores = [0.45, 0.68, 0.32, 0.28, 0.92]
        ranked = sorted(zip(range(len(candidates)), mock_scores), key=lambda x: x[1], reverse=True)

        for rank, (idx, score) in enumerate(ranked, 1):
            print(f"  [{rank}] 分数：{score:.2f}")
            print(f"      {candidates[idx]}")


# =============================================================================
# Cross-Encoder Rerank
# =============================================================================

def cross_encoder_rerank():
    """使用 sentence-transformers 的 CrossEncoder"""
    print(f"\n-- 示例 3: CrossEncoder Rerank")

    try:
        from sentence_transformers import CrossEncoder

        query = "如何学习人工智能？"
        candidates = [
            "人工智能是计算机科学的一个分支。",
            "学习 AI 需要掌握 Python 编程和数学基础。",
            "深度学习使用神经网络处理复杂任务。",
            "推荐系统根据用户行为推荐内容。",
            "人工智能包括机器学习和知识工程。",
        ]

        print(f"查询：{query}")
        print(f"候选文档：{len(candidates)} 条\n")

        print("加载 CrossEncoder 模型...")
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
        print("模型加载完成\n")

        print("计算相关性分数...")
        pairs = [[query, doc] for doc in candidates]
        scores = model.predict(pairs)

        ranked = sorted(zip(range(len(candidates)), scores), key=lambda x: x[1], reverse=True)

        print("\nRerank 结果：")
        print("-" * 50)
        for rank, (idx, score) in enumerate(ranked, 1):
            print(f"  [{rank}] 分数：{score:.4f}")
            print(f"      {candidates[idx]}")
        print()

    except ImportError:
        print("需要安装：pip install sentence-transformers")
        print("\n模拟演示：")
        print("-" * 50)

        candidates = [
            "人工智能是计算机科学的一个分支。",
            "学习 AI 需要掌握 Python 编程和数学基础。",
            "深度学习使用神经网络处理复杂任务。",
            "推荐系统根据用户行为推荐内容。",
            "人工智能包括机器学习和知识工程。",
        ]

        mock_scores = [0.35, 0.89, 0.52, 0.28, 0.41]
        ranked = sorted(zip(range(len(candidates)), mock_scores), key=lambda x: x[1], reverse=True)

        for rank, (idx, score) in enumerate(ranked, 1):
            print(f"  [{rank}] 分数：{score:.2f}")
            print(f"      {candidates[idx]}")


# =============================================================================
# Rerank 性能对比
# =============================================================================

def rerank_performance_comparison():
    """对比 Rerank 前后的效果"""
    print(f"\n-- 示例 4: Rerank 性能对比")

    query = "RAG 系统如何工作？"

    candidates = [
        ("向量检索", 0.88, "向量检索是根据查询向量查找相似文档的过程。"),
        ("RAG 简介", 0.85, "RAG 是检索增强生成，结合检索和生成的 AI 技术。"),
        ("Embedding", 0.79, "Embedding 将文本转换为数字向量。"),
        ("生成模型", 0.76, "生成模型根据输入生成新的内容，如 GPT。"),
        ("RAG 流程", 0.73, "RAG 的工作流程：检索相关文档，将文档和问题一起交给 LLM 生成答案。"),
        ("Milvus", 0.70, "Milvus 是向量数据库，用于存储和检索向量。"),
    ]

    print(f"查询：{query}")
    print()

    print("1. 向量检索结果（原始排名）：")
    print("-" * 50)
    for i, (name, score, _) in enumerate(candidates):
        print(f"  [{i+1}] {score:.2f} - {name}")
    print()

    # 模拟 Rerank 分数
    rerank_scores = [0.42, 0.68, 0.35, 0.48, 0.91, 0.29]
    reranked = list(zip(candidates, rerank_scores))
    reranked.sort(key=lambda x: x[1], reverse=True)

    print("2. Rerank 后结果：")
    print("-" * 50)
    for i, ((name, orig_score, content), rerank_score) in enumerate(reranked):
        orig_rank = next(j + 1 for j, (n, s, _) in enumerate(candidates) if n == name)
        new_rank = i + 1
        change = new_rank - orig_rank
        arrow = "↑" if change < 0 else "↓" if change > 0 else "="
        print(f"  [{new_rank}] {rerank_score:.2f} ({orig_rank}{arrow}) - {name}")
        if i < 3:
            print(f"      {content}")
    print()

    print("3. 效果对比：")
    print("-" * 50)

    def dcg(relevances):
        return sum(rel / (i + 1) for i, rel in enumerate(relevances))

    orig_dcg = dcg([1, 2, 0, 1, 3, 0])
    reranked_dcg = dcg([3, 2, 1, 1, 0, 0])
    ideal_dcg = dcg([3, 2, 1, 1, 0, 0])

    print(f"  原始 DCG: {orig_dcg:.2f}")
    print(f"  Rerank DCG: {reranked_dcg:.2f}")
    print(f"  理想 DCG: {ideal_dcg:.2f}")
    print(f"  NDCG 提升：{(reranked_dcg - orig_dcg) / ideal_dcg * 100:.1f}%")


def complete_rag_rerank_pipeline():
    """完整的 RAG 检索 + Rerank 流程说明"""
    print(f"\n-- 示例 5: 完整 RAG + Rerank 流程")

    print("""
完整的 RAG 检索流程（带 Rerank）:

  1. 用户提问
     → "机器学习和深度学习有什么区别？"

  2. 生成查询向量
     → Embedding 模型编码

  3. 向量检索（召回）
     → 从 Milvus 检索 Top-50 候选文档

  4. Rerank 重排序（精排）
     → 用 CrossEncoder 对 50 个候选重新打分
     → 选取 Top-5 最终结果

  5. 构建上下文
     → 拼接 Top-5 文档内容

  6. 调用 LLM 生成答案
     → Prompt: 根据以下信息回答问题...
""")


def rerank_best_practices():
    """Rerank 最佳实践"""
    print(f"\n-- 示例 6: Rerank 最佳实践")

    print("""
Rerank 最佳实践：

1. 何时使用 Rerank
   ✓ 对检索精度要求高的场景
   ✓ 候选集较小（<100 条）
   ✓ 有足够的计算资源
   ✗ 实时性要求极高（Rerank 会增加延迟）
   ✗ 候选集太大（成本过高）

2. 模型选择
   中文场景：BAAI/bge-reranker-large
   英文场景：cross-encoder/ms-marco-MiniLM-L-12-v2
   多语言：BAAI/bge-reranker-v2-m3
   API 方案：Cohere Rerank API

3. 参数建议
   - 召回数量：50-100 条（给 Rerank 足够选择）
   - 返回数量：5-10 条（LLM 上下文限制）
   - 分数阈值：过滤掉 <0.3 的低相关结果

4. 性能优化
   - 使用 fp16 推理（节省显存）
   - 批量处理：一次预测多对 query-doc
   - 缓存热点查询的 Rerank 结果

典型延迟对比:
  | 阶段    | 模型           | 延迟 (单次)  |
  |---------|---------------|-------------|
  | 检索    | Bi-Encoder    | ~10ms       |
  | Rerank  | CrossEncoder  | ~100-500ms  |
  | 生成    | LLM           | ~1-5s       |

建议：Rerank 增加的延迟通常值得，检索质量提升可显著改善答案质量。
""")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  Rerank 重排序（Re-ranking）")
    print("=" * 70 + "\n")

    mock_rerank_pipeline()
    bge_reranker_demo()
    cross_encoder_rerank()
    rerank_performance_comparison()
    complete_rag_rerank_pipeline()
    rerank_best_practices()

    print("\n" + "=" * 70)
    print("  下一步：04_rag_api/（RAG API 封装）")
    print("=" * 70 + "\n")
