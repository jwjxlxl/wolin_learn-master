# =============================================================================
# 01_rag_evaluation — RAG 系统质量评估
# =============================================================================
# 用途：学习如何科学评估 RAG 系统的检索和生成质量
# 难度：⭐⭐⭐（3 星）
#
# 核心概念：
#   1. 为什么需要评估 — 从"感觉不错"到"数据证明"
#   2. RAGAS 四大指标 — 忠实度、答案相关性、上下文精确率、召回率
#   3. 构建评估数据集 — 问题 + 标准答案 + 相关文档
#   4. 自动化评估流程
#
# 注意：本模块提供评估框架的完整实现。如需使用 RAGAS 库进行更专业的评估：
#   pip install ragas datasets
# =============================================================================

import os
import json
import math
from typing import List, Dict, Tuple


# =============================================================================
# 第一部分：为什么需要评估？
# =============================================================================

def explain_why_evaluate():
    """讲解 RAG 评估的必要性"""
    print("=" * 60)
    print("第一部分：为什么需要评估 RAG？")
    print("=" * 60)

    print("""
┌─────────────────────────────────────────────────────────┐
│ 不评估 = 盲目飞行                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 场景 1: 换了 Embedding 模型                              │
│   text-embedding-v3 → text-embedding-v4                 │
│   效果变好了吗？不知道 —— 因为你没测过                    │
│                                                         │
│ 场景 2: 调整了切片参数                                   │
│   chunk_size: 500 → 800                                 │
│   检索质量有没有提升？不知道 —— 因为你没测过             │
│                                                         │
│ 场景 3: 切换了 LLM                                       │
│   qwen-plus → deepseek-chat                            │
│   回答质量有没有变化？不知道 —— 因为你没测过             │
│                                                         │
│ 场景 4: 知识库更新了 100 篇新文档                         │
│   新文档被正确检索到了吗？不知道 —— 因为你没测过         │
│                                                         │
│ 📊 评估的价值：                                          │
│   ✓ 量化改进：知道每次改动到底有没有效果                 │
│   ✓ 发现问题：定位检索/生成的短板                       │
│   ✓ 版本对比：A/B 测试不同方案                          │
│   ✓ 上线信心：有数据支撑的质量保证                      │
│                                                         │
└─────────────────────────────────────────────────────────┘
""")


# =============================================================================
# 第二部分：理解 RAGAS 四大指标
# =============================================================================

def explain_ragas_metrics():
    """详细讲解 RAGAS 框架的核心指标"""
    print("\n" + "=" * 60)
    print("第二部分：RAGAS 四大核心指标")
    print("=" * 60)

    print("""
┌─────────────────────────────────────────────────────────┐
│ RAGAS（Retrieval Augmented Generation Assessment）      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 🔵 检索质量指标：                                        │
│                                                         │
│ 1. Context Precision（上下文精确率）                      │
│    公式：相关文档数 / 检索到的文档总数                   │
│    含义：检索到的文档中有多少是真正相关的？              │
│    目标：越高越好（避免噪音干扰 LLM）                    │
│    生活化比喻：Google 搜索结果前 10 条有几条有用？       │
│                                                         │
│ 2. Context Recall（上下文召回率）                         │
│    公式：检索到的相关文档数 / 所有相关文档总数           │
│    含义：所有相关文档中，有多少被检索到了？              │
│    目标：越高越好（避免遗漏关键信息）                    │
│    生活化比喻：图书馆里的相关书你找到了几本？            │
│                                                         │
│ 🟢 生成质量指标：                                        │
│                                                         │
│ 3. Faithfulness（忠实度）                                │
│    含义：LLM 的回答是否完全基于检索到的上下文？          │
│    目标：不应该出现上下文中没有的信息（幻觉）            │
│    生活化比喻：考试时你是否只用了给出的参考资料？        │
│                                                         │
│ 4. Answer Relevancy（答案相关性）                        │
│    含义：回答是否完整、准确地回答了用户问题？            │
│    目标：回答不能答非所问，也不能遗漏关键信息            │
│    生活化比喻：你问"今天天气怎么样"，我回答"下雨"       │
│              而不是回答"今天是星期五"                   │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 📊 四个指标的关系：                                      │
│                                                         │
│   用户问题                                               │
│      ↓                                                  │
│   [检索器] → Context Precision + Recall                 │
│      ↓                                                  │
│   检索上下文 + 问题                                      │
│      ↓                                                  │
│   [LLM 生成] → Faithfulness + Answer Relevancy          │
│      ↓                                                  │
│   最终答案                                               │
│                                                         │
└─────────────────────────────────────────────────────────┘
""")


# =============================================================================
# 第三部分：构建评估数据集
# =============================================================================

def explain_eval_dataset():
    """讲解评估数据集的构建方法"""
    print("\n" + "=" * 60)
    print("第三部分：构建 RAG 评估数据集")
    print("=" * 60)

    print("""
┌─────────────────────────────────────────────────────────┐
│ 评估数据集 = 测试问题 + 标准答案 + 相关文档               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 每条评估数据的结构：                                     │
│                                                         │
│ {                                                       │
│   "question": "什么是 RAG？",                            │
│   "ground_truth": "RAG 是检索增强生成...",               │
│   "relevant_docs": ["rag_intro.txt", "rag_guide.txt"]   │
│ }                                                       │
│                                                         │
│ 构建原则：                                               │
│                                                         │
│ 1. 覆盖主要知识点                                        │
│    - 不要所有问题都问同一个概念                          │
│    - 覆盖文档的不同章节/主题                             │
│                                                         │
│ 2. 包含各种问题类型                                      │
│    - 事实性问题（"XXX 是什么？"）                        │
│    - 对比性问题（"A 和 B 有什么区别？"）                 │
│    - 应用性问题（"XXX 用在什么场景？"）                  │
│                                                         │
│ 3. 数量建议                                              │
│    - 最少：20 条（覆盖核心功能）                         │
│    - 推荐：50-100 条（有统计意义）                       │
│    - 生产：200+ 条（全面覆盖）                           │
│                                                         │
│ 4. 标注要点                                              │
│    - ground_truth 要简洁但完整                           │
│    - relevant_docs 要包含所有相关的文档 ID               │
│    - 可以加入"无关文档"测试检索器的抗噪能力             │
│                                                         │
└─────────────────────────────────────────────────────────┘
""")


# =============================================================================
# 第四部分：评估计算实现
# =============================================================================

class RAGEvaluator:
    """RAG 系统评估器

    实现了简化的 RAGAS 风格评估指标，用于教学演示。
    生产环境建议使用 ragas 库：pip install ragas

    用法：
        evaluator = RAGEvaluator()
        evaluator.add_test_case(question, ground_truth, relevant_docs)

        # 模拟一次 RAG 调用后
        scores = evaluator.evaluate(
            retrieved_docs=["doc1", "doc2"],
            llm_answer="RAG 是..."
        )
    """

    def __init__(self):
        self.test_cases: List[Dict] = []

    def add_test_case(self, question: str, ground_truth: str,
                      relevant_docs: List[str]):
        """添加一个评估用例

        Args:
            question: 测试问题
            ground_truth: 标准答案
            relevant_docs: 相关文档 ID 列表
        """
        self.test_cases.append({
            "question": question,
            "ground_truth": ground_truth,
            "relevant_docs": relevant_docs,
        })

    def load_from_json(self, file_path: str):
        """从 JSON 文件加载评估数据集

        JSON 格式：
        [
            {
                "question": "什么是 RAG？",
                "ground_truth": "RAG 是检索增强生成...",
                "relevant_docs": ["doc1", "doc2"]
            }
        ]
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            self.add_test_case(
                item["question"],
                item["ground_truth"],
                item["relevant_docs"],
            )
        print(f"✓ 从 {file_path} 加载了 {len(data)} 条评估用例")

    def context_precision(self, retrieved_docs: List[str],
                          relevant_docs: List[str]) -> float:
        """计算上下文精确率

        Precision = |检索到且相关的文档| / |检索到的文档|

        Args:
            retrieved_docs: 系统检索到的文档 ID 列表
            relevant_docs: 实际相关的文档 ID 列表

        Returns:
            精确率（0.0 ~ 1.0）
        """
        if not retrieved_docs:
            return 0.0

        relevant_set = set(relevant_docs)
        retrieved_set = set(retrieved_docs)
        relevant_retrieved = relevant_set & retrieved_set

        return len(relevant_retrieved) / len(retrieved_set)

    def context_recall(self, retrieved_docs: List[str],
                       relevant_docs: List[str]) -> float:
        """计算上下文召回率

        Recall = |检索到且相关的文档| / |所有相关文档|

        Args:
            retrieved_docs: 系统检索到的文档 ID 列表
            relevant_docs: 实际相关的文档 ID 列表

        Returns:
            召回率（0.0 ~ 1.0）
        """
        if not relevant_docs:
            return 1.0

        relevant_set = set(relevant_docs)
        retrieved_set = set(retrieved_docs)
        relevant_retrieved = relevant_set & retrieved_set

        return len(relevant_retrieved) / len(relevant_set)

    def faithfulness_score(self, llm_answer: str,
                           retrieved_docs_content: List[str]) -> float:
        """估算忠实度（简化版）

        实际 RAGAS 使用 LLM 来判断每句话是否有上下文支撑。
        这里简化为：答案中的关键词在上下文中的覆盖率。

        注意：这是教学简化版本。生产环境请使用 ragas 库的 faithfulness 指标。

        Args:
            llm_answer: LLM 生成的答案
            retrieved_docs_content: 检索到的文档内容列表

        Returns:
            忠实度（0.0 ~ 1.0）
        """
        if not llm_answer or not retrieved_docs_content:
            return 0.0

        # 简化：将上下文拼接，计算答案中有多少内容能在上下文中找到
        context_text = " ".join(retrieved_docs_content)

        # 将答案拆分为句子
        sentences = [s.strip() for s in llm_answer.replace("！", "。").replace("？", "。").split("。") if s.strip()]

        if not sentences:
            return 0.0

        # 简化：检查每个句子中的关键词是否在上下文中出现
        supported = 0
        for sentence in sentences:
            # 提取长度 >= 2 的字符对作为特征
            chars = sentence.replace(" ", "")
            if len(chars) < 4:
                supported += 1
                continue

            # 检查句子中较长的子串是否出现在上下文中
            found = False
            for i in range(len(chars) - 3):
                if chars[i:i+4] in context_text:
                    found = True
                    break
            if found:
                supported += 1

        return supported / len(sentences)

    def answer_relevancy_score(self, question: str, llm_answer: str) -> float:
        """估算答案相关性（简化版）

        实际 RAGAS 使用 LLM 生成反向问题来评估。
        这里简化为：答案中是否包含问题中的关键实体。

        注意：这是教学简化版本。生产环境请使用 ragas 库。

        Args:
            question: 用户问题
            llm_answer: LLM 生成的答案

        Returns:
            相关性（0.0 ~ 1.0）
        """
        if not llm_answer:
            return 0.0

        # 从问题中提取关键词（长度 >= 2 的中文字符组）
        q_chars = question.replace(" ", "").replace("？", "").replace("?", "")
        keywords = set()
        for i in range(len(q_chars) - 1):
            keywords.add(q_chars[i:i+2])

        if not keywords:
            return 1.0

        # 计算答案中覆盖了多少关键词
        a_text = llm_answer.replace(" ", "")
        matched = sum(1 for kw in keywords if kw in a_text)

        return matched / len(keywords)

    def evaluate_single(self, question: str, ground_truth: str,
                        relevant_docs: List[str],
                        retrieved_docs: List[str],
                        llm_answer: str,
                        retrieved_contents: List[str] = None) -> Dict:
        """评估单条用例

        Args:
            question: 测试问题
            ground_truth: 标准答案
            relevant_docs: 相关文档 ID 列表
            retrieved_docs: 系统检索到的文档 ID 列表
            llm_answer: LLM 生成的答案
            retrieved_contents: 检索到的文档内容（用于忠实度计算）

        Returns:
            各项指标得分
        """
        if retrieved_contents is None:
            retrieved_contents = []

        return {
            "question": question[:40] + "..." if len(question) > 40 else question,
            "context_precision": self.context_precision(retrieved_docs, relevant_docs),
            "context_recall": self.context_recall(retrieved_docs, relevant_docs),
            "faithfulness": self.faithfulness_score(llm_answer, retrieved_contents),
            "answer_relevancy": self.answer_relevancy_score(question, llm_answer),
        }

    def evaluate_all(self, retrieval_fn, generation_fn) -> Dict:
        """对所有测试用例运行完整评估

        Args:
            retrieval_fn: 检索函数 (question) -> (List[doc_ids], List[doc_contents])
            generation_fn: 生成函数 (question, contexts) -> answer

        Returns:
            汇总结果
        """
        results = []
        for case in self.test_cases:
            # 执行 RAG
            retrieved_ids, retrieved_contents = retrieval_fn(case["question"])
            answer = generation_fn(case["question"], retrieved_contents)

            # 评估
            scores = self.evaluate_single(
                case["question"], case["ground_truth"],
                case["relevant_docs"],
                retrieved_ids, answer, retrieved_contents,
            )
            results.append(scores)

        # 汇总
        avg = {
            "context_precision": sum(r["context_precision"] for r in results) / len(results),
            "context_recall": sum(r["context_recall"] for r in results) / len(results),
            "faithfulness": sum(r["faithfulness"] for r in results) / len(results),
            "answer_relevancy": sum(r["answer_relevancy"] for r in results) / len(results),
            "total_cases": len(results),
        }
        avg["overall"] = (
            avg["context_precision"] * 0.25 +
            avg["context_recall"] * 0.25 +
            avg["faithfulness"] * 0.25 +
            avg["answer_relevancy"] * 0.25
        )

        return {"summary": avg, "details": results}

    def print_report(self, result: Dict):
        """打印评估报告"""
        s = result["summary"]
        print("\n" + "=" * 60)
        print("  RAG 评估报告")
        print("=" * 60)
        print(f"""
  测试用例数：{s['total_cases']}

  🔵 检索质量：
     Context Precision（精确率）：{s['context_precision']:.2%}
     Context Recall（召回率）：   {s['context_recall']:.2%}

  🟢 生成质量：
     Faithfulness（忠实度）：     {s['faithfulness']:.2%}
     Answer Relevancy（相关性）： {s['answer_relevancy']:.2%}

  📊 综合得分：{s['overall']:.2%}
""")

        # 评级
        overall = s['overall']
        if overall >= 0.80:
            grade = "🌟 优秀 — RAG 系统运行良好！"
        elif overall >= 0.60:
            grade = "👍 良好 — 还有优化空间"
        elif overall >= 0.40:
            grade = "⚠️ 一般 — 需要针对性改进"
        else:
            grade = "🔴 较差 — 系统存在明显问题，建议检查配置"

        print(f"  评级：{grade}")
        print("=" * 60)

        # 打印各用例详情
        print("\n  各用例详情：")
        for i, r in enumerate(result["details"]):
            print(f"\n  [{i+1}] {r['question']}")
            print(f"      Pre={r['context_precision']:.2f} Rec={r['context_recall']:.2f} "
                  f"Faith={r['faithfulness']:.2f} Relev={r['answer_relevancy']:.2f}")


# =============================================================================
# 第五部分：评估最佳实践
# =============================================================================

def evaluation_best_practices():
    """RAG 评估的最佳实践"""
    print("\n" + "=" * 60)
    print("第五部分：评估最佳实践")
    print("=" * 60)

    print("""
┌─────────────────────────────────────────────────────────┐
│ RAG 评估最佳实践                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 1. 建立基线（Baseline）                                  │
│    - 在改任何参数之前，先跑一次完整评估                   │
│    - 记录基线分数作为对比基准                             │
│                                                         │
│ 2. 每次只改一个变量                                      │
│    - ❌ 同时改 Embedding 模型 + 切片参数 + LLM           │
│    - ✅ 只改 Embedding，其他不变，对比前后分数            │
│                                                         │
│ 3. 评估频率                                              │
│    - 开发阶段：每次重大修改后评估                         │
│    - 上线前：完整评估 + 人工抽查                          │
│    - 上线后：定期评估（检测效果衰减）                    │
│                                                         │
│ 4. 自动化                                                │
│    - 将评估脚本集成到 CI/CD 流程                          │
│    - 每次 PR 自动跑评估，避免质量回退                    │
│                                                         │
│ 5. 人工评估不可替代                                      │
│    - 自动化指标 ≠ 用户满意度                             │
│    - 定期人工抽查 10-20 条回答的质量                     │
│    - 关注：是否流畅、是否有用、是否有毒性                │
│                                                         │
│ 6. 持续更新评估集                                        │
│    - 知识库更新 → 评估集也要更新                         │
│    - 收集用户真实问题 → 加入评估集                       │
│    - 定期审查评估集的相关性和时效性                      │
│                                                         │
│ 📦 推荐工具：                                            │
│   - ragas: pip install ragas（最流行的 RAG 评估库）     │
│   - DeepEval: pip install deepeval                      │
│   - TruLens: pip install trulens-eval                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
""")


# =============================================================================
# 主程序：完整评估演示
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  RAG 系统质量评估")
    print("=" * 70 + "\n")

    # 1. 讲解评估的必要性
    explain_why_evaluate()

    # 2. 讲解 RAGAS 指标
    explain_ragas_metrics()

    # 3. 讲解数据集构建
    explain_eval_dataset()

    # 4. 演示评估计算
    print("\n" + "=" * 60)
    print("第四部分：评估演示")
    print("=" * 60)

    evaluator = RAGEvaluator()

    # 添加模拟测试用例
    evaluator.add_test_case(
        "什么是 RAG？",
        "RAG（检索增强生成）是一种结合信息检索和文本生成的技术。它先检索相关知识，再由大语言模型基于检索结果生成答案，从而减少幻觉。",
        ["rag_intro.txt", "rag_guide.txt"],
    )
    evaluator.add_test_case(
        "Milvus 支持哪些检索方式？",
        "Milvus 支持标量查询、稠密向量检索、稀疏向量（BM25）检索和混合检索。支持 RRF 和加权两种融合策略。",
        ["milvus_doc.txt", "search_guide.txt"],
    )
    evaluator.add_test_case(
        "BM25 算法有什么特点？",
        "BM25 基于 TF-IDF 改进，考虑了词频饱和度和文档长度归一化。它是经典的关键词检索算法。",
        ["bm25_paper.txt"],
    )

    # 模拟检索和生成函数
    def mock_retrieval(question):
        """模拟检索（用简单规则替代真实的 Milvus 检索）"""
        if "RAG" in question:
            return (["rag_intro.txt", "rag_guide.txt"],
                    ["RAG 是检索增强生成技术...", "RAG 结合检索和生成..."])
        elif "Milvus" in question:
            return (["milvus_doc.txt", "search_guide.txt", "other.txt"],
                    ["Milvus 支持多种检索方式...", "混合检索融合稠密和稀疏向量...", "今天天气很好..."])
        elif "BM25" in question:
            return (["bm25_paper.txt"],
                    ["BM25 是基于 TF-IDF 改进的关键词检索算法..."])
        else:
            return ([], [])

    def mock_generation(question, contexts):
        """模拟 LLM 生成（用简单规则替代真实的 LLM）"""
        context_text = " ".join(contexts)
        if "RAG" in question:
            return "RAG（检索增强生成）是一种结合信息检索和文本生成的技术。它先检索相关知识，再由大模型基于检索结果生成答案。"
        elif "Milvus" in question:
            return "Milvus 支持标量查询、稠密向量检索、BM25 稀疏向量检索和混合检索。还支持 RRF 融合策略。"
        elif "BM25" in question:
            return "BM25 是基于 TF-IDF 改进的经典关键词检索算法。它考虑了词频饱和度。"
        else:
            return "无法回答。"

    # 运行评估
    result = evaluator.evaluate_all(mock_retrieval, mock_generation)
    evaluator.print_report(result)

    # 5. 最佳实践
    evaluation_best_practices()

    print("\n" + "=" * 70)
    print("  评估学习完成！你可以：")
    print("  1. 用 RAGEvaluator 评估你的 rag_demo 项目")
    print("  2. 对比不同配置下的评估得分")
    print("  3. 安装 ragas 库进行更专业的评估")
    print("=" * 70)
