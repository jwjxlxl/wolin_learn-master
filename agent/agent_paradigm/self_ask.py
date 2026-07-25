"""
Self-Ask Agent 教学示例 (问题拆解)
====================================
背景: Microsoft Research 《Self-Ask with Search》(2022)

核心思想:
  让模型在回答复杂问题时学会"反问自己"，把大问题拆成多个小问题，
  逐个击破后再组合答案。

场景例子:
  问: "2016年奥斯卡最佳男主角的年龄是多少?"
  Self-Ask 会先问: "2016年奥斯卡最佳男主角是谁?" → 答: 莱昂纳多·迪卡普里奥
  再问: "莱昂纳多·迪卡普里奥的出生年份?" → 答: 1974年
  最后组合: 2016 - 1974 = 42岁

适合: 事实链路长、需要多步检索/推理的问题
"""

import json
import logging
import re

from dotenv import load_dotenv

from agent.agent_paradigm.qwen import call_llm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
load_dotenv()


# ──────────────────── Self-Ask Agent ────────────────────

SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions by breaking them down into smaller sub-questions when needed.

You MUST respond in JSON format with the following structure:

{
    "need_followup": true or false,
    "followup_question": "the sub-question to answer next (only when need_followup is true)",
    "final_answer": "the final combined answer (only when need_followup is false)"
}

Here are two examples:

---
Example 1:
Question: Who is older, Taylor Swift or Justin Bieber?

Response:
{
    "need_followup": true,
    "followup_question": "When was Taylor Swift born?",
    "final_answer": null
}

(Intermediate answer: Taylor Swift was born on December 13, 1989.)

Response:
{
    "need_followup": true,
    "followup_question": "When was Justin Bieber born?",
    "final_answer": null
}

(Intermediate answer: Justin Bieber was born on March 1, 1994.)

Response:
{
    "need_followup": false,
    "followup_question": null,
    "final_answer": "Taylor Swift is older."
}

---
Example 2:
Question: What is the population of the city where the founder of Microsoft was born?

Response:
{
    "need_followup": true,
    "followup_question": "Where was the founder of Microsoft born?",
    "final_answer": null
}

(Intermediate answer: Bill Gates, founder of Microsoft, was born in Seattle, Washington.)

Response:
{
    "need_followup": true,
    "followup_question": "What is the population of Seattle?",
    "final_answer": null
}

(Intermediate answer: The population of Seattle is approximately 740,000 (as of 2023).)

Response:
{
    "need_followup": false,
    "followup_question": null,
    "final_answer": "Approximately 740,000."
}
"""


def self_ask(question: str, max_depth: int = 5) -> str:
    """
    Self-Ask: 自动拆解问题，逐个子问题求解

    流程:
      1. 判断是否需要拆解为子问题
      2. 如果需要，提出一个 followup question 并尝试回答
      3. 将子问题的答案加入上下文，回到步骤1
      4. 直到给出最终答案

    参数:
      question:   原始问题
      max_depth:  最大拆解深度，防止无限追问

    返回:
      最终答案字符串
    """
    # 用于积累已获得的子问题答案
    context = ""

    for depth in range(1, max_depth + 1):
        # 构建当前 prompt：原始问题 + 已知上下文
        user_content = f"Question: {question}"
        if context:
            user_content += f"\n\n{context}"

        response = call_llm([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ])

        logger.info(f"[Self-Ask Step {depth}]\n{response}\n")

        # 从 LLM 回复中提取 JSON（兼容 markdown 代码块包裹）
        json_str = response
        code_block = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
        if code_block:
            json_str = code_block.group(1)

        try:
            result = json.loads(json_str)
        except json.JSONDecodeError:
            logger.error(f"JSON 解析失败，原始回复: {response}")
            return response.strip()

        need_followup = result.get("need_followup", False)
        followup_question = result.get("followup_question")
        final_answer = result.get("final_answer")

        if not need_followup and final_answer:
            return final_answer

        if need_followup and followup_question:
            logger.info(f"  -> 子问题: {followup_question}")

            # 用一次独立的 LLM 调用来回答这个子问题
            sub_answer = call_llm([
                {"role": "system", "content": "Please answer the following question concisely."},
                {"role": "user", "content": followup_question},
            ])
            logger.info(f"  -> 子问题答案: {sub_answer}\n")

            # 将子问题和答案积累到上下文中
            context += f"\nIntermediate answer: {sub_answer}"
        else:
            # 异常状态，直接返回
            return response.strip()

    return "达到最大拆解深度，未能完成推理"


# ──────────────────── 教学演示 ────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Self-Ask Agent 演示 (JSON 返回格式)")
    print("=" * 60)

    q1 = "2016年奥斯卡最佳男主角的年龄是多少？"
    print(f"问题: {q1}")
    result = self_ask(q1)
    print(f"最终答案: {result}")
