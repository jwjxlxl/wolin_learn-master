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
# 注意：本模块提供两层评估能力：
#   1. 离线原理版：纯关键词启发式，无需 API Key（用于理解指标如何计算）
#   2. LLM 判定版：忠实度 / 答案相关性改用 LLM 判定（接近 ragas），
#      并在文件末尾给出"对真实 RAG（Milvus + Qwen）做端到端评测"的完整示例
# 如需使用 ragas 库进行更专业的评估：pip install ragas datasets
# =============================================================================

import os
import re
import sys
import json
from typing import List, Dict

# Windows 控制台 UTF-8 编码（否则打印中文/emoji 会乱码）
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


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

    实现 RAGAS 风格的四大指标，支持两种计算方式：
      - 离线关键词法（默认，无需 API Key，用于理解原理）
      - LLM 判定法（传入 llm 后启用，忠实度 / 相关性接近 ragas 做法）

    用法：
        # 离线原理版
        evaluator = RAGEvaluator()

        # LLM 判定版（需 ALIYUN_API_KEY）
        evaluator = RAGEvaluator(llm=get_qwen_client())

        evaluator.add_test_case(question, ground_truth, relevant_docs)
        result = evaluator.evaluate_all(retrieval_fn, generation_fn)
        evaluator.print_report(result)
    """

    def __init__(self, llm=None):
        """初始化评估器。

        Args:
            llm: 可选。LangChain 聊天模型（如 get_qwen_client() 返回的 ChatOpenAI）。
                 传入后忠实度 / 答案相关性改用 LLM 判定；不传则用离线关键词法。
        """
        self.test_cases: List[Dict] = []
        self.llm = llm

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

    # ------------------------------------------------------------------
    # LLM 判定版指标（接近 ragas 的"LLM-as-Judge"做法）
    # ------------------------------------------------------------------

    def _ask_llm(self, system_prompt: str, user_prompt: str):
        """调用 LLM 做判断题，返回其输出文本；未配置 llm 时返回 None。"""
        if self.llm is None:
            return None
        from langchain_core.messages import SystemMessage, HumanMessage
        resp = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        return resp.content.strip()

    def faithfulness_score_llm(self, llm_answer: str,
                               retrieved_contents: List[str]) -> float:
        """忠实度（LLM 判定版）：逐句判断回答是否有参考资料支撑。

        对照 ragas 的 faithfulness：把回答拆成若干陈述句，由 LLM 判断
        每句话是否能在检索到的上下文中找到依据，得分 = 有依据句数 / 总句数。
        """
        sentences = [s.strip() for s in llm_answer.replace("！", "。").replace("？", "。").split("。") if s.strip()]
        if not sentences or not retrieved_contents:
            return 0.0

        context = "\n".join(f"[{i}] {c}" for i, c in enumerate(retrieved_contents, 1))
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, 1))

        system_prompt = (
            "你是 RAG 系统的忠实度（Faithfulness）评估员。"
            "你的任务是判断答案的每一句话是否有'参考资料'的支撑，只依据参考资料判断，不考虑外部知识。"
        )
        user_prompt = (
            f"参考资料：\n{context}\n\n"
            f"待评估答案拆成的句子：\n{numbered}\n\n"
            f"请逐句判断每句话是否有参考资料支撑，"
            f"只输出 JSON 数组，元素为 1（有支撑）或 0（无支撑），"
            f"长度必须等于句子数 {len(sentences)}，例如：[1, 0, 1]"
        )

        raw = self._ask_llm(system_prompt, user_prompt)
        if raw is None:
            return self.faithfulness_score(llm_answer, retrieved_contents)

        digits = [int(ch) for ch in re.findall(r"[01]", raw)]
        if not digits:
            print("  ⚠️ 未解析出忠实度判定结果，回退关键词法")
            return self.faithfulness_score(llm_answer, retrieved_contents)

        # 对齐长度：不足补 0，多余截断
        scores = (digits + [0] * len(sentences))[:len(sentences)]
        return sum(scores) / len(sentences)

    def answer_relevancy_llm(self, question: str, ground_truth: str,
                             llm_answer: str) -> float:
        """答案相关性（LLM 判定版）：对照问题 + 标准答案打分。

        对照 ragas 的 answer relevancy：核心是回答是否切题、是否完整覆盖
        标准答案的关键信息。这里让 LLM 直接输出 0~1 分数。
        """
        if not llm_answer:
            return 0.0

        system_prompt = (
            "你是 RAG 系统的答案相关性（Answer Relevancy）评估员。"
            "请对照'问题'和'标准答案'，评估'LLM 回答'是否切题、是否完整准确地回答了问题。"
        )
        user_prompt = (
            f"问题：{question}\n"
            f"标准答案：{ground_truth}\n"
            f"LLM 回答：{llm_answer}\n\n"
            f"请输出 0 到 1 之间的小数作为相关性得分（只输出数字，例如 0.85）。"
        )

        raw = self._ask_llm(system_prompt, user_prompt)
        if raw is None:
            return self.answer_relevancy_score(question, llm_answer)

        m = re.search(r"(\d+(?:\.\d+)?)", raw)
        if not m:
            print("  ⚠️ 未解析出相关性分数，回退关键词法")
            return self.answer_relevancy_score(question, llm_answer)

        score = float(m.group(1))
        if score > 1:
            # LLM 可能误用百分制/十分制，做一次归一化
            score = score / 100 if score > 100 else score / 10
        return max(0.0, min(1.0, score))

    def _faithfulness(self, llm_answer: str, retrieved_contents: List[str]) -> float:
        """忠实度统一入口：有 LLM 用 LLM 判定，否则回退关键词法。"""
        if self.llm is not None:
            try:
                return self.faithfulness_score_llm(llm_answer, retrieved_contents)
            except Exception as e:
                print(f"  ⚠️ LLM 忠实度评估失败，回退关键词法：{e}")
        return self.faithfulness_score(llm_answer, retrieved_contents)

    def _answer_relevancy(self, question: str, ground_truth: str,
                          llm_answer: str) -> float:
        """答案相关性统一入口：有 LLM 用 LLM 判定，否则回退关键词法。"""
        if self.llm is not None:
            try:
                return self.answer_relevancy_llm(question, ground_truth, llm_answer)
            except Exception as e:
                print(f"  ⚠️ LLM 相关性评估失败，回退关键词法：{e}")
        return self.answer_relevancy_score(question, llm_answer)

    def evaluate_single(self, question: str, ground_truth: str,
                        relevant_docs: List[str],
                        retrieved_docs: List[str],
                        llm_answer: str,
                        retrieved_contents: List[str] = None) -> Dict:
        """评估单条用例

        Args:
            question: 测试问题
            ground_truth: 标准答案（LLM 判定答案相关性时使用；离线关键词法下暂未使用）
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
            # 忠实度 / 相关性：配置了 LLM 用 LLM 判定，否则回退关键词法
            "faithfulness": self._faithfulness(llm_answer, retrieved_contents),
            "answer_relevancy": self._answer_relevancy(question, ground_truth, llm_answer),
        }

    def evaluate_all(self, retrieval_fn, generation_fn) -> Dict:
        """对所有测试用例运行完整评估

        Args:
            retrieval_fn: 检索函数 (question) -> (List[doc_ids], List[doc_contents])
            generation_fn: 生成函数 (question, contexts) -> answer

        Returns:
            汇总结果（评估集为空时返回 None）
        """
        if not self.test_cases:
            print("⚠️ 评估集为空，请先调用 add_test_case 或 load_from_json 添加用例。")
            return None

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
        if result is None:
            print("  没有可打印的评估结果（评估集为空）。")
            return
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
# 第六部分：对真实 RAG 做端到端评测（对接 05_rag_pipeline 的真实链路）
# =============================================================================

def run_real_rag_evaluation():
    """对真实 RAG 做端到端评测（需 Milvus 运行 + ALIYUN_API_KEY）。

    与前面 mock 演示的区别：
      - 检索：真实 Milvus 混合检索（HybridRetriever），与 05_rag_pipeline 一致
      - 生成：真实 Qwen LLM（get_qwen_client）
      - 指标：忠实度 / 答案相关性改用 LLM 判定（接近 ragas）
      - 文档 ID：插入时给每个切片分配稳定 doc_id，检索时读回，
        这样评估数据集的 relevant_docs 才能引用，精确率 / 召回率才算得出来

    说明：每条用例约 3 次 LLM 调用（生成 + 忠实度 + 相关性）。
    """
    print("\n" + "=" * 60)
    print("第六部分：对真实 RAG 做端到端评测")
    print("=" * 60)

    # ---- 0. 检查云端 LLM（生成答案 + LLM 判定共用）----
    try:
        from utils.model_utils import get_qwen_client
    except ImportError:
        get_qwen_client = None

    llm = get_qwen_client() if get_qwen_client else None
    if llm is None:
        print("\n✗ 未配置 ALIYUN_API_KEY，跳过真实 RAG 评测。")
        print("  请先在 .env 配置 ALIYUN_API_KEY，再运行本段。")
        return

    # ---- 1. 连接 Milvus + 准备 Embedding ----
    try:
        from pymilvus import MilvusClient, DataType
        from rag_examples.milvus_config import MILVUS_URI, MILVUS_DB_NAME
        from rag_examples.utils.hybrid_retriever import HybridRetriever
        from rag_examples.embedding_examples.aliyun_embedding import AliyunEmbedding
        from langchain_core.messages import SystemMessage, HumanMessage

        client = MilvusClient(uri=MILVUS_URI, db_name=MILVUS_DB_NAME)
        emb = AliyunEmbedding()
    except ImportError as e:
        print(f"\n✗ 缺少依赖：{e}")
        print("  请确认已安装 pymilvus / dashscope / langchain-core。")
        return
    except Exception as e:
        print(f"\n✗ 连接 Milvus 失败：{e}")
        print("  请先启动 Milvus 服务（Docker）。")
        return

    # ---- 2. 建库 + 插入（关键：给每个切片分配稳定 doc_id）----
    retriever = HybridRetriever(
        client, "rag_eval_demo", emb,
        extra_fields=[{"name": "doc_id", "datatype": DataType.VARCHAR, "max_length": 64}],
    )

    documents = [
        ("doc_0", "机器学习是人工智能的核心技术，通过数据训练模型实现预测和决策。"),
        ("doc_1", "深度学习是机器学习的一个分支，使用多层神经网络进行表征学习。"),
        ("doc_2", "自然语言处理让计算机理解、生成人类语言，应用于机器翻译和问答系统。"),
        ("doc_3", "计算机视觉让计算机理解图像和视频，应用于人脸识别和自动驾驶。"),
        ("doc_4", "推荐系统根据用户历史行为推荐内容，常用协同过滤算法。"),
        ("doc_5", "知识图谱用图结构表示实体及其关系，支撑搜索和推理。"),
    ]
    doc_ids = [d[0] for d in documents]
    texts = [d[1] for d in documents]

    try:
        dense_vectors = [emb.embed(t) for t in texts]
        retriever.create_collection(dim=len(dense_vectors[0]))
        retriever.insert(texts, dense_vectors,
                         extra_data=[{"doc_id": i} for i in doc_ids])
        print(f"  ✓ 知识库就绪（{len(texts)} 个带 doc_id 的切片）")
    except Exception as e:
        print(f"\n✗ 建库 / 插入失败：{e}")
        return

    # ---- 3. 构建评估数据集（question + ground_truth + relevant_docs）----
    # 传入 llm：忠实度 / 相关性启用 LLM 判定（不再用关键词法）
    evaluator = RAGEvaluator(llm=llm)
    evaluator.add_test_case(
        "什么是机器学习？",
        "机器学习是人工智能的核心技术，通过数据训练模型实现预测和决策。",
        ["doc_0"],
    )
    evaluator.add_test_case(
        "深度学习和机器学习是什么关系？",
        "深度学习是机器学习的一个分支，使用多层神经网络进行表征学习。",
        ["doc_1", "doc_0"],
    )
    evaluator.add_test_case(
        "自然语言处理有哪些应用？",
        "自然语言处理应用于机器翻译和问答系统。",
        ["doc_2"],
    )

    # ---- 4. 定义真实检索 / 生成函数（对接真实 RAG）----
    def retrieval_fn(question):
        """真实混合检索：返回 (doc_ids, doc_contents)。"""
        query_vec = emb.embed(question)
        hits = retriever.hybrid_search(question, query_vec, top_k=3)
        ids = [h["entity"]["doc_id"] for h in hits]
        contents = [h["entity"][retriever.text_field] for h in hits]
        return ids, contents

    def generation_fn(question, contexts):
        """真实 LLM 生成：用检索到的上下文组装 Prompt 后调用 Qwen。"""
        context_block = "\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
        user_prompt = f"参考资料：\n{context_block}\n\n问题：{question}"
        resp = llm.invoke([
            SystemMessage(content="你是严谨的知识问答助手，只能根据参考资料回答，禁止编造。"),
            HumanMessage(content=user_prompt),
        ])
        return resp.content

    # ---- 5. 运行评估 ----
    print("\n  开始逐条评估（真实检索 + 真实生成 + LLM 判定）...")
    result = evaluator.evaluate_all(retrieval_fn, generation_fn)
    evaluator.print_report(result)


# =============================================================================
# 主程序：完整评估演示
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  RAG 系统质量评估")
    print("=" * 70 + "\n")

    # 1. 讲解评估的必要性
    # explain_why_evaluate()

    # 2. 讲解 RAGAS 指标
    # explain_ragas_metrics()

    # 3. 讲解数据集构建
    # explain_eval_dataset()

    # 4. 演示评估计算（离线原理版：关键词法，无需 API Key）
    print("\n" + "=" * 60)
    print("第四部分：评估演示（离线原理版，无需 API Key）")
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
    # evaluation_best_practices()

    # 6. 对真实 RAG 做端到端评测（需 Milvus + ALIYUN_API_KEY）
    run_real_rag_evaluation()

    print("\n" + "=" * 70)
    print("  评估学习完成！你可以：")
    print("  1. 离线原理版：用关键词法理解四大指标的计算（无需 API Key）")
    print("  2. LLM 判定版：配置 ALIYUN_API_KEY 后，用 LLM 判定忠实度/相关性")
    print("  3. 真实评测：用 run_real_rag_evaluation() 评你的 Milvus+Qwen RAG")
    print("  4. 安装 ragas 库进行更专业的评估")
    print("=" * 70)
