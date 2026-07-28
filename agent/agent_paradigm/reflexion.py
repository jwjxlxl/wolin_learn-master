"""
Reflexion Agent 教学示例 (反思与迭代优化)
=========================================
背景: 《Reflexion: Language Agents with Verbal Reinforcement Learning》(2023)

核心思想:
  Agent 具备自我纠错的能力。每次尝试后，会评估自己的结果是否正确；
  如果失败了，会总结"哪里错了、为什么错"，带着反思再试一次。

流程:
  1. 尝试完成任务 (Attempt)
  2. 自我评估结果 (Self-Reflect)
  3. 如果失败，生成反思 (Reflection)
  4. 将反思加入上下文，重新尝试 (Retry with Reflection)
  5. 循环直到成功或达到最大次数

适合场景:
  代码生成 (第一次写错 → 读报错信息 → 反思 → 修正)
  逻辑推理 (答案不对 → 反思哪里想错了 → 重试)
  流程执行 (步骤遗漏 → 反思漏了什么 → 补全)
"""

import logging
import re

from dotenv import load_dotenv

from agent.agent_paradigm.qwen import call_llm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
load_dotenv()


# ──────────────────── Reflexion Agent ────────────────────

SYSTEM_PROMPT_ATTEMPT = """\
请完成以下具体任务，给出你最好的尝试。
要仔细认真，先输出你的推理过程，最后用"最终答案："明确标记你的最终结果。
"""

SYSTEM_PROMPT_REFLECT = """\
你是一位专业的评审员。给定一个任务、Agent 的尝试结果，以及（如果有的话）之前的反思，
你的工作是：
1. 指出哪里做错了或不完整。
2. 解释为什么会出错。
3. 给出具体的改进建议。

要具体且可操作。不要重写整个答案——指出问题所在，引导 Agent 改进。

请按以下格式输出你的反思：
问题：<哪里出了问题>
原因：<为什么会出现这个问题>
建议：<如何修正>
"""

SYSTEM_PROMPT_RETRY = """\
你之前已经尝试过这个任务。评审员提供了反馈意见。
请根据下面的反思修改你的答案，给出改进版本。
同样，先输出推理过程，最后用"最终答案："明确标记你的最终结果。
"""


def self_reflect(question: str, attempt: str, previous_reflections: str = "") -> str:
    """
    对一次尝试进行反思，找出问题并给出改进建议

    参数:
      question:             原始问题/任务
      attempt:              Agent 的尝试结果
      previous_reflections: 之前累积的反思 (可选)
                            传入历史反思可以避免 LLM 重复指出同一个问题

    返回:
      反思文本 (问题 / 原因 / 建议 格式)
    """
    # 组装上下文: 总是包含任务 + 本次尝试结果
    context = f"任务：{question}\n\nAgent 的尝试：\n{attempt}"
    # 如果有历史反思也一并传入，让评审员 LLM 看到完整的反思链条
    if previous_reflections:
        context += f"\n\n之前的反思：\n{previous_reflections}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_REFLECT},  # 评审员角色设定
        {"role": "user", "content": context},                   # 任务 + 尝试 + 历史反思
    ]
    return call_llm(messages)


def reflexion_agent(
    question: str,
    max_attempts: int = 3,
) -> dict[str, str]:
    """
    Reflexion Agent: 尝试 → 自我评估 → 反思 → 重试

    流程:
      1. Agent 尝试回答问题
      2. 自我评估: 回答是否正确/完整?
      3. 如果否，生成反思并重新尝试
      4. 重复直到回答满意，或达到最大次数

    参数:
      question:      问题/任务
      max_attempts:  最大尝试次数

    返回:
      {
        "attempts":     所有尝试的内容 (用分隔符连接)
        "reflections":  所有反思的内容 (用分隔符连接)
        "final_answer": 最终回答
        "succeeded":    是否在第 N 次尝试中成功
      }
    """
    # 三个变量的用途:
    #   all_attempts           — 记录每次 LLM 输出的完整答案，用于最后返回历史
    #   all_reflections        — 记录每次反思的完整文本，用于最后返回反思链
    #   accumulated_reflections — 累积的反思内容，传入下一轮 LLM 作为上下文
    #                             它与 all_reflections 的区别: 是"纯文本拼接"而非列表
    all_attempts = []
    all_reflections = []
    accumulated_reflections = ""

    for attempt_num in range(1, max_attempts + 1):
        # ── Step 1: 尝试回答 ──
        # 首次尝试用 SYSTEM_PROMPT_ATTEMPT (直接解决问题)
        # 后续重试用 SYSTEM_PROMPT_RETRY (带着反思修改答案)
        if attempt_num == 1:
            sys_prompt = SYSTEM_PROMPT_ATTEMPT
        else:
            sys_prompt = SYSTEM_PROMPT_RETRY

        # 构建 user 消息: 始终包含任务本身
        user_content = f"任务：{question}"
        # 如果不是第一次尝试，把之前的反思也传进去
        # 这样 LLM 能"记住"之前哪里做错了
        if accumulated_reflections:
            user_content += f"\n\n上次尝试的反思：\n{accumulated_reflections}"

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]

        attempt = call_llm(messages)
        all_attempts.append(attempt)
        logger.info(f"[Reflexion Attempt {attempt_num}/{max_attempts}]\n{attempt}\n")

        # ── Step 2: 自我评估 (Self-Evaluation) ──
        # 核心思想: 不依赖外部评判，让 LLM 自己当评审员来检查自己的答案
        # 这与 Reflexion 论文中的 "verbal reinforcement" 机制一致
        eval_prompt = (
            f"任务：{question}\n\n"
            f"Agent 的回答：\n{attempt}\n\n"
            f"Agent 的回答是否正确且完整地完成了任务？"
            '请只回答"是"或"否"。'  # 限制输出为二元判断，便于后续提取
        )
        eval_response = call_llm([
            {"role": "system", "content": "你是一位严格的评审员，请简洁回答。"},
            {"role": "user", "content": eval_prompt},
        ]).strip().upper()
        # 注意: 这里用了一次独立的 LLM 调用，而非复用主对话
        # 好处: 评估角色独立，不受主对话上下文影响，更客观
        # .strip().upper(): 去除首尾空白并转大写，方便后续判断

        # 提取评估结果: 检查 eval_response 中是否包含"是"
        # 示例: 若 eval_response = "是"，则 is_correct = True
        #       若 eval_response = "否"，则 is_correct = False
        is_correct = "是" in eval_response
        logger.info(f"[Self-Evaluation] {eval_response} -> {'正确' if is_correct else '不正确'}\n")

        if is_correct:
            # LLM 自评通过，提取最终答案
            # re.search(r"最终答案：\s*(.+)", attempt, re.DOTALL):
            #   "最终答案：" — 匹配字面量
            #   \s*        — 匹配 0 个或多个空白字符
            #   (.+)       — 捕获组1: 匹配 1 个或多个任意字符
            #   re.DOTALL  — 让 . 也能匹配换行符 \n (因为最终答案可能多行)
            # 示例: attempt 中包含 "最终答案：\n第 23 只鸡，第 12 只兔"
            #       则 answer_match.group(1) = "\n第 23 只鸡，第 12 只兔"
            answer_match = re.search(r"最终答案：\s*(.+)", attempt, re.DOTALL)
            final_answer = answer_match.group(1).strip() if answer_match else attempt.strip()
            return {
                # 用 "\n---\n" 分隔所有历史尝试，便于阅读和调试
                "attempts": "\n---\n".join(f"第 {i+1} 次尝试：\n{a}" for i, a in enumerate(all_attempts)),
                # 同理，分隔所有反思记录；如果没有反思说明首次尝试即成功
                "reflections": "\n---\n".join(f"第 {i+1} 次反思：\n{r}" for i, r in enumerate(all_reflections)) if all_reflections else "无（首次尝试即成功）",
                "final_answer": final_answer,
                "succeeded": True,
            }

        # ── Step 3: 生成反思 ──
        # 调用 self_reflect() 让 LLM 以评审员身份分析本次尝试的问题
        # 注意: accumulated_reflections 传入了之前所有反思的累积内容
        # 这样 LLM 能看到"之前已经反思过什么"，避免重复反思同一个问题
        reflection = self_reflect(question, attempt, accumulated_reflections)
        all_reflections.append(reflection)
        # 将本次反思追加到 accumulated_reflections 中
        # 三元表达式: 如果已有内容则在前面加换行，否则直接赋值
        accumulated_reflections += f"\n{reflection}" if accumulated_reflections else reflection
        logger.info(f"[Reflection]\n{reflection}\n")

    # ── 所有尝试均失败 ──
    # 循环走完仍没得到 LLM 自评通过的 "是"，返回最后一次尝试的结果
    # all_attempts[-1]: 取最后一次尝试的内容
    answer_match = re.search(r"最终答案：\s*(.+)", all_attempts[-1], re.DOTALL)
    final_answer = answer_match.group(1).strip() if answer_match else all_attempts[-1].strip()

    return {
        "attempts": "\n---\n".join(f"第 {i+1} 次尝试：\n{a}" for i, a in enumerate(all_attempts)),
        "reflections": "\n---\n".join(f"第 {i+1} 次反思：\n{r}" for i, r in enumerate(all_reflections)),
        "final_answer": final_answer,
        "succeeded": False,  # 标记为未成功，调用方可据此决定是否需要 fallback
    }


# ──────────────────── 教学演示 ────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Reflexion Agent 演示 (尝试 → 反思 → 重试)")
    print("=" * 60)

    # 示例 1: 代码生成 — 故意选一个容易出错的问题
    q1 = "用 Python 写一个函数 fibonacci(n)，返回前 n 个斐波那契数。要求 n=0 返回空列表，n=1 返回 [0]，n=2 返回 [0, 1]。"
    print(f"\n问题: {q1}")
    result = reflexion_agent(q1, max_attempts=3)
    print(f"\n最终答案: {result['final_answer']}")
    print(f"是否成功: {'是' if result['succeeded'] else '否'}")
    if result['reflections']:
        print(f"\n反思记录:\n{result['reflections']}")

    # 示例 2: 逻辑推理
    # print("\n" + "=" * 60)
    # q2 = "一个农场有鸡和兔子共35个头，94只脚。鸡和兔子各有多少只？请列出计算过程。"
    # print(f"\n问题: {q2}")
    # result2 = reflexion_agent(q2, max_attempts=3)
    # print(f"\n最终答案: {result2['final_answer']}")
    # print(f"是否成功: {'是' if result2['succeeded'] else '否'}")
