"""
Plan-and-Execute Agent 教学示例 (计划与执行)
============================================
背景: LangChain 社区实践 (2023年前后)

核心思想:
  把任务拆成两个阶段 —— 先生成计划 (Planning)，再逐步执行 (Execution)。

场景例子:
  让 Agent 写"新能源车市场调研报告"，不会直接生成，而是先拟定计划：
    1. 收集销量数据
    2. 分析政策趋势
    3. 总结消费者反馈
    4. 撰写结论
  然后逐条执行每个步骤，最后汇总结果。

适合: 多步骤、需要长时间运行的复杂任务
"""

import logging

from dotenv import load_dotenv

from agent.agent_paradigm.qwen import call_llm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
load_dotenv()


# ──────────────────── Plan-and-Execute Agent ────────────────────

def plan_and_execute(task: str, max_steps: int = 5) -> str:
    """
    Plan-and-Execute: 先计划，再逐步执行

    流程:
      1. Planning 阶段 —— 让 LLM 拟定执行计划（编号步骤列表）
      2. Execution 阶段 —— 逐条执行计划中的步骤，每步结果加入上下文
      3. Synthesis 阶段 —— 汇总所有步骤结果，生成最终答案

    参数:
      task:       用户下达的任务
      max_steps:  最大执行步骤数，防止计划过长导致无限执行

    返回:
      最终汇总结果
    """
    # ── Phase 1: Planning ──
    logger.info("=== Phase 1: Planning ===")
    plan = call_llm([
        {"role": "system", "content": (
            "你是一个任务规划助手。请将用户的任务拆解为编号的步骤列表。\n"
            "每步一行，格式为: 1. [步骤描述]\n"
            "不要执行步骤，只输出计划。\n"
            f"计划的最长步数为{max_steps},禁止超过。"
        )},
        {"role": "user", "content": f"任务: {task}"},
    ])
    logger.info(f"[计划]\n{plan}\n")

    # 解析计划：提取编号步骤
    import re
    steps = re.findall(r"^\d+\.\s*(.+)", plan, re.MULTILINE)
    if not steps:
        steps = [plan]  # 解析失败时，将整个计划视为单步
    steps = steps[:max_steps]  # 限制最大步数

    # ── Phase 2: Execution ──
    logger.info("=== Phase 2: Execution ===")
    results = []

    for i, step in enumerate(steps, 1):
        logger.info(f"正在执行步骤 {i}/{len(steps)}: {step}")

        # 构建上下文：包含任务描述和前面步骤的结果
        context = f"任务: {task}\n"
        if results:
            context += "\n已完成的步骤:\n"
            for j, (s, r) in enumerate(zip(steps, results), 1):
                context += f"  步骤{j} ({s}): {r}\n"

        context += f"\n当前需要完成的步骤: {step}\n请完成这一步。"

        step_result = call_llm([
            {"role": "system", "content": "你是一个执行助手。请专注于完成当前步骤，给出具体、有用的结果。"},
            {"role": "user", "content": context},
        ])
        logger.info(f"[步骤{i} 结果]\n{step_result}\n")
        results.append(step_result)

    # ── Phase 3: Synthesis ──
    logger.info("=== Phase 3: Synthesis ===")
    all_results = "\n\n".join(
        f"步骤{i+1} ({step}):\n{result}"
        for i, (step, result) in enumerate(zip(steps, results))
    )

    final_answer = call_llm([
        {"role": "system", "content": "你是一个总结助手。请根据以下各步骤的结果，给出完整、连贯的最终回答。"},
        {"role": "user", "content": (
            f"原始任务: {task}\n\n"
            f"各步骤执行结果:\n{all_results}\n\n"
            f"请给出最终回答:"
        )},
    ])
    logger.info(f"[最终答案]\n{final_answer}\n")

    return final_answer


if __name__ == "__main__":
    print("=" * 60)
    print("Plan-and-Execute Agent 演示")
    print("=" * 60)

    task = "请帮我写一篇关于Python编程语言2025年发展趋势的简短分析报告"
    print(f"任务: {task}")
    result = plan_and_execute(task)
    print(f"最终结果: {result}")
