# =============================================================================
# 计划与执行模式 — Plan-and-Execute
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解 Plan-and-Execute 的核心思想：先计划，再执行
#   ✅ 实现基础版：LLM 生成计划列表，逐步执行每个步骤
#   ✅ 理解与 ReAct 的差异：ReAct 是单步交替，P&E 是两阶段
#
# 运行前检查：
# 1. 已安装 Ollama 并下载模型：ollama pull qwen3.5:2b
# =============================================================================

import sys
import os
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 确保项目根目录和 agent/ 目录可导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from langchain_core.prompts import ChatPromptTemplate
from utils.model_utils import get_model


# =============================================================================
# 核心概念：Plan-and-Execute（计划与执行）
# =============================================================================
"""
提出背景：2023 年前后的 Agent 应用开发框架实践（如 LangChain 社区）。

核心思想：把任务拆成两个阶段，先生成计划(Planning)，再逐步执行(Execution)。

生活化比喻：
  ReAct = 走一步看一步：先想一下，走一步，观察一下，再想下一步
          → 灵活，但缺乏全局视野
  P&E = 先看地图再出发：先规划路线，再沿着路线走
        → 有全局计划，不容易迷路

适用场景：多步骤、需长时间的任务（写报告、做调研、对比分析等）

与 ReAct 的差异：
  ReAct:  Thought → Action → Observation → Thought → ... （在线、增量）
  P&E:    [生成完整计划] → 执行步骤1 → 执行步骤2 → ... → 汇总 （离线、规划先行）
"""


# =============================================================================
# 示例 1: 基础 Plan-and-Execute — 生成计划 + 逐步执行
# =============================================================================

def basic_plan_and_execute():
    """
    两阶段 Plan-and-Execute：
      1. Planning: 让 LLM 生成执行计划（列表）
      2. Execution: 逐步执行每个步骤，LLM 生成步骤内容
      3. Summary: 汇总所有步骤结果

    示例："比较 Python 和 JavaScript 在变量声明、循环语法、
    错误处理上的区别"
    """
    print(f"\n-- 示例 1: 基础 Plan-and-Execute — 生成计划 + 逐步执行")

    model = get_model()
    if model is None:
        print("  【跳过】请安装 Ollama 并下载模型：ollama pull qwen3.5:2b")
        return

    question = "比较 Python 和 JavaScript 在变量声明、循环语法、错误处理上的区别"

    print(f"  任务：{question}")
    print()

    # --- 阶段 1：生成计划 ---
    print("  【阶段 1：生成计划】")
    plan_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个任务规划专家。请将用户的任务拆解为具体的执行步骤。每个步骤应该是独立的、可执行的动作。请输出编号列表。"),
        ("user", "请将以下任务拆解为具体的执行步骤（3-5步）：\n\n{question}\n\n请按编号列表输出，每行一个步骤，格式如：\n1. 步骤一\n2. 步骤二\n..."),
    ])

    plan_response = (plan_prompt | model).invoke({"question": question})
    plan_text = plan_response.content
    print(f"  生成的计划：")
    for line in plan_text.split("\n"):
        print(f"    {line}")
    print()

    # 解析计划：提取编号列表行
    steps = []
    for line in plan_text.split("\n"):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith("-")):
            # 去除编号
            step_text = line.lstrip("0123456789. -）)").strip()
            if step_text:
                steps.append(step_text)

    if not steps:
        steps = ["分析 Python 的变量声明语法", "分析 JavaScript 的变量声明语法",
                 "对比循环语法差异", "对比错误处理差异"]
        print(f"  （未解析到编号列表，使用默认步骤）")

    print(f"  共拆解为 {len(steps)} 个步骤：")
    for i, s in enumerate(steps):
        print(f"    {i + 1}. {s}")
    print()

    # --- 阶段 2：逐步执行 ---
    print("  【阶段 2：逐步执行】")
    step_results = []

    for i, step in enumerate(steps):
        print(f"\n  --- 执行步骤 {i + 1}/{len(steps)}: {step} ---")

        # 构建带上下文的执行 prompt
        context_so_far = "\n".join([f"- {r}" for r in step_results]) if step_results else "（尚未有结果）"

        exec_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个执行专家。请根据已执行的结果，完成当前步骤。输出要简洁、专业。"),
            ("user", """任务：{question}

当前步骤：{step}

已执行的结果：
{context}

请完成当前步骤："""),
        ])

        result = (exec_prompt | model).invoke({
            "question": question,
            "step": step,
            "context": context_so_far,
        })

        step_text = result.content
        print(f"    结果：{step_text[:150]}...")
        step_results.append(f"步骤{i+1}({step}): {step_text}")

    # --- 阶段 3：汇总 ---
    print("\n  【阶段 3：汇总最终答案】")
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个总结专家。请根据以下所有步骤的执行结果，生成最终的完整回答。"),
        ("user", """原始任务：{question}

各步骤执行结果：
{results}

请综合以上结果，生成完整的最终回答："""),
    ])

    summary = (summary_prompt | model).invoke({
        "question": question,
        "results": "\n\n".join(step_results),
    })

    print(f"\n  ✅ 最终回答：")
    for line in summary.content.split("\n"):
        print(f"    {line}")


# =============================================================================
# 示例 2: Plan-and-Execute + 工具调用
# =============================================================================

def plan_and_execute_with_tools():
    """
    计划步骤中包含工具调用的 P&E 模式。

    示例：回答"马云的公司成立于哪年？当时他多大？公司现在多大估值？"
      计划：查公司 → 查成立年份 → 查马云出生年份 → 计算年龄 → 查估值
      执行：每一步可能调用搜索工具或计算
    """
    print(f"\n-- 示例 2: Plan-and-Execute + 工具调用")

    model = get_model()
    if model is None:
        print("  【跳过】请安装 Ollama 并下载模型：ollama pull qwen3.5:2b")
        return

    from helpers import simulate_search

    question = "马云的公司成立于哪年？当时他多大？"

    print(f"  任务：{question}")
    print()

    # --- 简化的 P&E：先计划，再带工具执行 ---
    # 阶段 1：生成计划
    plan_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个任务规划专家。请将任务拆解为 3 个具体步骤。"),
        ("user", "请将以下任务拆解为 3 个具体步骤：\n{question}\n\n每行一个步骤。"),
    ])
    plan_response = (plan_prompt | model).invoke({"question": question})
    steps = [line.strip() for line in plan_response.content.split("\n") if line.strip() and line.strip()[0].isdigit()]
    print(f"  生成的计划：")
    for s in steps:
        print(f"    {s}")
    print()

    # 阶段 2：带工具执行
    print("  【阶段 2：带工具执行】")
    tool_calls_made = []

    for i, step in enumerate(steps):
        print(f"\n  --- 步骤 {i + 1}: {step} ---")

        # 判断这一步需要搜索什么
        search_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个搜索策略专家。请根据当前步骤，提取需要搜索的关键词。只输出关键词，不要解释。"),
            ("user", """任务：{question}
当前步骤：{step}
需要搜索什么关键词？请直接输出。"""),
        ])
        search_kw = (search_prompt | model).invoke({
            "question": question,
            "step": step,
        })
        keyword = search_kw.content.strip()
        print(f"    搜索关键词: {keyword}")

        # 执行搜索
        result = simulate_search(keyword)
        print(f"    搜索结果: {result}")
        tool_calls_made.append(f"搜索「{keyword}」→ {result}")

    # 汇总
    print(f"\n  ✅ 综合以上信息：")
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个信息整合专家。"),
        ("user", """任务：{question}
搜索结果汇总：
{results}

请综合以上搜索结果，回答问题："""),
    ])
    final = (summary_prompt | model).invoke({
        "question": question,
        "results": "\n".join(tool_calls_made),
    })
    print(f"    {final.content}")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  Plan-and-Execute（计划与执行）模式")
    print("  先看地图再出发：先规划路线，再沿着路线走")
    print("=" * 70 + "\n")

    basic_plan_and_execute()
    # plan_and_execute_with_tools()

    print("\n" + "=" * 70)
    print("  行动范式学习完毕！接下来进入 03_collaboration/（协作范式）")
    print("=" * 70 + "\n")
