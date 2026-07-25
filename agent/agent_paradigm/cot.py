"""
Chain-of-Thought (CoT) Agent 教学示例
========================================
目标: 让学生理解 CoT 的核心思想 —— 让 LLM 先"想"再"答"

核心范式:
  1. Zero-Shot CoT: 只需在 prompt 中加入 "Let's think step by step"
  2. Few-Shot CoT: 给几个带推理步骤的示例
  3. ReAct (CoT + Tool Use): 思考 → 行动 → 观察 → 再思考
"""

import logging

from dotenv import load_dotenv

from agent.agent_paradigm.qwen import call_llm
from agent.agent_paradigm.react import react_agent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
load_dotenv()




# ──────────────────── 1. Zero-Shot CoT ────────────────────
# 最简单的方式: 不給示例，只加一句引导语

def zeroshot_cot(question: str) -> tuple[str, str]:
    """
    Zero-Shot Chain-of-Thought

    流程:
      step 1: 让模型逐步推理 (thinking)
      step 2: 基于推理给出最终答案 (answer)

    返回: (thinking, answer)  便于观察中间推理过程
    """
    # --- 第一步: 逐步推理 ---
    thinking = call_llm([
        {"role": "system", "content": "You are a helpful assistant. Think step by step."},
        {"role": "user", "content": f"Q: {question}\nA: Let's think step by step."},
    ])
    logger.info(f"[ZeroShot-Thinking]\n{thinking}\n")

    # --- 第二步: 总结答案 ---
    answer = call_llm([
        {"role": "system", "content": "Based on the reasoning above, give a concise final answer."},
        {"role": "user", "content": (
            f"Question: {question}\n\n"
            f"Reasoning:\n{thinking}\n\n"
            f"Final answer:"
        )},
    ])
    logger.info(f"[ZeroShot-Answer]\n{answer}\n")
    return thinking, answer


# ──────────────────── 2. Few-Shot CoT ────────────────────
# 提供带完整推理步骤的示例，引导模型模仿

FEWSHOT_EXAMPLES = """\
Q: 小明有5个苹果，吃了2个，又买了3个，现在有几个？
A: Let's think step by step.
   1) 小明一开始有 5 个苹果。
   2) 吃了 2 个后，剩下 5 - 2 = 3 个。
   3) 又买了 3 个，总共 3 + 3 = 6 个。
   所以答案是 6。

Q: 一个房间有4个角，每个角坐着一只猫。每只猫对面有3只猫。请问房间里一共有几只猫？
A: Let's think step by step.
   1) 房间有4个角，每个角一只猫，所以有4只猫。
   2) "每只猫对面有3只猫" 是对同一事实的另一种描述 —— 总共4只猫，任意一只猫的对面确实是其余3只。
   3) 没有额外的猫进出房间。
   所以答案是 4。
"""


def fewshot_cot(question: str) -> tuple[str, str]:
    """
    Few-Shot Chain-of-Thought

    与 Zero-Shot 的区别: 在 prompt 中给出带推理步骤的示例
    模型会模仿示例的推理格式，通常比 Zero-Shot 更可靠
    """
    prompt = (
        "请按照以下示例的方式逐步推理，然后给出最终答案。\n\n"
        f"{FEWSHOT_EXAMPLES}\n"
        f"Q: {question}\nA: Let's think step by step."
    )

    thinking = call_llm([
        {"role": "system", "content": "你是一个善于逐步推理的助手。请模仿示例的推理格式。"},
        {"role": "user", "content": prompt},
    ])
    logger.info(f"[FewShot-Thinking]\n{thinking}\n")

    answer = call_llm([
        {"role": "system", "content": "Based on the reasoning above, give a concise final answer."},
        {"role": "user", "content": (
            f"Question: {question}\n\n"
            f"Reasoning:\n{thinking}\n\n"
            f"Final answer:"
        )},
    ])
    logger.info(f"[FewShot-Answer]\n{answer}\n")
    return thinking, answer


# ──────────────────── 教学演示 ────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("1. Zero-Shot CoT 演示")
    print("=" * 60)
    q1 = "一个水池有两个水管，进水管每小时进水6吨，出水管每小时排水4吨。空池容量20吨，几小时能装满？"
    thinking, answer = zeroshot_cot(q1)
    print(f"问题: {q1}")
    print(f"推理: {thinking}")
    print(f"答案: {answer}")

    print("\n" + "=" * 60)
    print("2. Few-Shot CoT 演示")
    print("=" * 60)
    q2 = "火车从A到B时速60km，从B到A时速40km，AB距离120km，往返平均时速是多少？"
    thinking, answer = fewshot_cot(q2)
    print(f"问题: {q2}")
    print(f"推理: {thinking}")
    print(f"答案: {answer}")

    print("\n" + "=" * 60)
    print("3. ReAct Agent 演示 (CoT + 工具)")
    print("=" * 60)
    q3 = "小明有100元，买了3本书每本28.5元，又买了2支笔每支4.5元，还剩多少钱？"
    result = react_agent(q3)
    print(f"问题: {q3}")
    print(f"最终结果: {result}")