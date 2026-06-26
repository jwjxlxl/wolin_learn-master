# =============================================================================
# Supervisor 模式 — 主管分配任务给工人，审查后决定下一步
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解 Supervisor：一个主管 Agent 管理多个工人 Agent 的工作流
#   ✅ 用 Command(goto=...) 实现工人完成后回到主管的循环
#   ✅ 掌握 Supervisor 模式的迭代审查流程
#
# 运行前检查：
# 1. 已安装依赖：pip install langgraph langchain-core langchain-ollama pydantic
# 2. 已安装 Ollama 并下载模型：ollama pull qwen3.5:2b
# 3. 示例 1 无需 LLM（纯逻辑演示，可立即运行）
# =============================================================================

import sys
import os
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 确保项目根目录可导入（用于 utils/ 共享模块）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command


# =============================================================================
# 核心概念：Supervisor（监督者）是什么？
# =============================================================================
"""
Supervisor 模式（主管监督）

  场景：软件开发团队
  产品经理提需求后：
    项目经理（Supervisor）拆解任务 → 分配研究员调研 → 审查结果 →
    分配程序员编码 → 审查代码 → 分配评审员验收 → 最终交付 → 结束

  关键特征：
    1. Supervisor 是唯一做决策的 Agent（分配任务、审查结果、决定何时结束）
    2. 所有 Worker 完成工作后必须回到 Supervisor（不能自行结束）
    3. Supervisor 可以多次循环（研究不满意 → 再研究 / 换人研究）
    4. 用 iteration 字段防止无限循环

  与 Handoff 的区别：
    Handoff       = 接力赛，交棒后前一个人退出
    Supervisor    = 包工头派活，工人干完回来汇报，包工头再派下一个活

  与 Agent-as-Tool 的区别：
    Agent-as-Tool  = 主 Agent 同步调用子 Agent（工具式：我问你答）
    Supervisor     = 主管有完整的工作流管理（分配→审查→再分配→再审查→验收）

  生活化比喻：
    Supervisor = 装修队长，负责找工人、看进度、验收、决定下一步
"""


# =============================================================================
# 示例 1: 无 LLM 版 — 规则驱动的软件开发团队（理解控制流）
# =============================================================================

def dev_team_demo():
    """
    软件开发团队：不用 LLM，用规则展示 Supervisor 的分配→审查循环。

    START → 项目经理（分配任务）
                ├── researcher → 回到项目经理
                ├── coder → 回到项目经理
                └── reviewer → 回到项目经理 →（三次后完成）→ END

    关键点：
      - 项目经理决定"下一个是谁"
      - 每个工人干完必回项目经理（Command(goto="supervisor")）
      - 迭代计数器防止无限循环
    """
    print(f"\n-- 示例 1: 无 LLM 版 — 规则驱动的软件开发团队")

    class SupervisorState(TypedDict):
        messages: Annotated[list, add_messages]
        task: str
        next: str             # "researcher" | "coder" | "reviewer" | "FINISH"
        work_product: str     # 当前工人的产出
        all_products: list    # 所有产出汇总
        iteration: int        # 迭代计数

    def supervisor(state: SupervisorState):
        """
        项目经理：决定下一个任务分配给谁。
        顺序：researcher → coder → reviewer → FINISH
        """
        iteration = state["iteration"]
        task = state["task"]

        # 超过 5 次迭代强制结束（安全保护）
        if iteration >= 5:
            print(f"  [项目经理] 迭代次数达上限，强制完成")
            return Command(goto=END, update={
                "next": "FINISH",
                "all_products": state["all_products"] + ["（迭代上限，强制完成）"],
            })

        # 按顺序分配：研究 → 编码 → 评审
        task_order = ["researcher", "coder", "reviewer"]
        next_worker = task_order[iteration % 3]

        print(f"  [项目经理] 第 {iteration + 1} 轮 → 分配给: {next_worker}")
        print(f"    任务: {task}")

        return Command(goto=next_worker, update={
            "next": next_worker,
            "messages": [AIMessage(content=f"请 {next_worker} 处理: {task}")],
        })

    def researcher(state: SupervisorState):
        """研究员：调研技术方案。"""
        task = state["task"]
        print(f"  [研究员] 调研: {task}")
        product = f"调研报告：{task} 可采用 Python + LangGraph 实现，需 Milvus 做向量存储。"
        print(f"    产出: {product[:60]}...")
        return Command(goto="supervisor", update={
            "work_product": product,
            "all_products": state["all_products"] + [product],
            "iteration": state["iteration"] + 1,
            "next": "",  # 清空，让 supervisor 重新决定
        })

    def coder(state: SupervisorState):
        """程序员：编写代码。"""
        task = state["task"]
        prev = state.get("work_product", "")
        print(f"  [程序员] 编码（基于: {prev[:40]}...）")
        product = f"代码实现：完成 {task} 的核心模块，包含 StateGraph 和工具节点。"
        print(f"    产出: {product[:60]}...")
        return Command(goto="supervisor", update={
            "work_product": product,
            "all_products": state["all_products"] + [product],
            "iteration": state["iteration"] + 1,
            "next": "",
        })

    def reviewer(state: SupervisorState):
        """评审员：审查代码质量。"""
        task = state["task"]
        print(f"  [评审员] 审查 {task} 的产出")
        product = f"评审报告：{task} 的代码结构清晰，测试覆盖完整，可以交付。"
        print(f"    产出: {product[:60]}...")
        return Command(goto="supervisor", update={
            "work_product": product,
            "all_products": state["all_products"] + [product],
            "iteration": state["iteration"] + 1,
            "next": "",
        })

    # ===== 构建图 =====
    graph = (
        StateGraph(SupervisorState)
        .add_node("supervisor", supervisor)
        .add_node("researcher", researcher)
        .add_node("coder", coder)
        .add_node("reviewer", reviewer)
        .add_edge(START, "supervisor")
        .compile()
    )

    # ===== 测试 =====
    tasks = [
        "搭建一个智能问答 Agent",
        "实现文档检索功能",
    ]

    for task in tasks:
        print(f"\n【项目需求】{task}")
        result = graph.invoke({
            "messages": [HumanMessage(content=task)],
            "task": task,
            "next": "",
            "work_product": "",
            "all_products": [],
            "iteration": 0,
        })
        print(f"  【最终产出】（共 {len(result['all_products'])} 步）")
        for i, p in enumerate(result["all_products"], 1):
            print(f"    {i}. {p[:70]}...")
        print()


# =============================================================================
# 示例 2: LLM 版 — 真正的 Supervisor（LLM 做决策和审查）
# =============================================================================

def supervisor_with_llm():
    """
    用 LLM 做任务分配和审查的 Supervisor 模式。

    项目经理用结构化输出决定分配给谁，各工人用 LLM 执行专业工作。
    """
    print(f"\n-- 示例 2: LLM 版 — 真正的 Supervisor")

    from utils.model_utils import get_model
    from pydantic import BaseModel, Field
    from typing import Literal

    model = get_model("qwen")
    if model is None:
        print("  【跳过】请安装 Ollama 并下载模型：ollama pull qwen3.5:2b")
        return

    # ===== Supervisor 结构化输出 =====
    class SupervisorDecision(BaseModel):
        next_worker: Literal["researcher", "coder", "reviewer", "FINISH"] = Field(
            description="下一个要分配的工人，或 FINISH 表示完成"
        )
        reasoning: str = Field(description="为什么分配给这个工人")

    supervisor_model = model.with_structured_output(SupervisorDecision)

    class SupervisorState(TypedDict):
        messages: Annotated[list, add_messages]
        task: str
        next: str
        work_product: str
        all_products: list
        iteration: int

    def supervisor(state: SupervisorState):
        """项目经理：用 LLM 决定下一个工人。"""
        iteration = state["iteration"]
        task = state["task"]

        if iteration >= 5:
            print(f"  [项目经理] 迭代次数达上限，强制完成")
            return Command(goto=END, update={
                "next": "FINISH",
                "all_products": state["all_products"] + ["（迭代上限，强制完成）"],
            })

        print(f"  [项目经理] 第 {iteration + 1} 轮分配")
        print(f"    任务: {task}")
        if state.get("work_product"):
            print(f"    最新产出: {state['work_product'][:60]}...")

        decision: SupervisorDecision = supervisor_model.invoke(
            f"你是软件开发项目经理。当前任务：{task}。"
            f"已有工作进展：{state.get('all_products', [])}。"
            f"请选择下一个要分配的工人：researcher（调研方案）、coder（编写代码）、"
            f"reviewer（审查验收）、FINISH（任务完成）。说明理由。"
        )

        if decision.next_worker == "FINISH":
            print(f"    决定: 任务完成！理由: {decision.reasoning}")
            return Command(goto=END, update={
                "next": "FINISH",
            })

        print(f"    决定: → {decision.next_worker} | 理由: {decision.reasoning}")
        return Command(goto=decision.next_worker, update={
            "next": decision.next_worker,
            "messages": [AIMessage(content=f"请 {decision.next_worker} 处理: {task}。原因: {decision.reasoning}")],
        })

    def researcher(state: SupervisorState):
        """研究员：用 LLM 做技术调研。"""
        task = state["task"]
        print(f"  [研究员] 调研: {task}")

        response = model.invoke(
            f"你是技术研究员。请为以下任务做技术调研，推荐技术方案和工具：\n{task}"
        )
        product = response.content
        print(f"    产出: {product[:80]}...")
        return Command(goto="supervisor", update={
            "work_product": product,
            "all_products": state["all_products"] + [f"[调研] {product}"],
            "iteration": state["iteration"] + 1,
            "next": "",
        })

    def coder(state: SupervisorState):
        """程序员：用 LLM 编写代码。"""
        task = state["task"]
        prev = state.get("work_product", "无")
        print(f"  [程序员] 编码（参考: {prev[:50]}...）")

        response = model.invoke(
            f"你是程序员。任务：{task}。请基于已有方案编写代码实现。"
        )
        product = response.content
        print(f"    产出: {product[:80]}...")
        return Command(goto="supervisor", update={
            "work_product": product,
            "all_products": state["all_products"] + [f"[代码] {product}"],
            "iteration": state["iteration"] + 1,
            "next": "",
        })

    def reviewer(state: SupervisorState):
        """评审员：用 LLM 审查产出。"""
        task = state["task"]
        products = state.get("all_products", [])
        print(f"  [评审员] 审查 {task} 的全部产出（{len(products)} 项）")

        response = model.invoke(
            f"你是代码评审员。任务：{task}。已有产出：{products}。"
            f"请进行评审，指出优点、不足和改进建议。"
        )
        product = response.content
        print(f"    产出: {product[:80]}...")
        return Command(goto="supervisor", update={
            "work_product": product,
            "all_products": state["all_products"] + [f"[评审] {product}"],
            "iteration": state["iteration"] + 1,
            "next": "",
        })

    # ===== 构建图 =====
    graph = (
        StateGraph(SupervisorState)
        .add_node("supervisor", supervisor)
        .add_node("researcher", researcher)
        .add_node("coder", coder)
        .add_node("reviewer", reviewer)
        .add_edge(START, "supervisor")
        .compile()
    )

    # 保存图为 PNG
    images_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images')
    os.makedirs(images_dir, exist_ok=True)
    png_path = os.path.join(images_dir, 'supervisor_with_llm.png')
    with open(png_path, 'wb') as f:
        f.write(graph.get_graph().draw_mermaid_png())
    print(f"  图已保存到: {png_path}\n")

    # ===== 测试 =====
    tasks = [
        "搭建一个能搜索公司知识库的智能问答 Agent",
        "实现文档自动切片和向量化入库功能",
    ]

    for task in tasks:
        print(f"\n【项目需求】{task}")
        result = graph.invoke({
            "messages": [HumanMessage(content=task)],
            "task": task,
            "next": "",
            "work_product": "",
            "all_products": [],
            "iteration": 0,
        })
        print(f"  【项目完成】（共 {len(result['all_products'])} 步）")
        for i, p in enumerate(result["all_products"], 1):
            print(f"    {i}. {p[:70]}...")
        print()


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  Supervisor 模式 — 主管分配任务给工人，审查后决定下一步")
    print("  软件开发团队：项目经理 → 研究员 → 程序员 → 评审员 → 交付")
    print("=" * 70 + "\n")

    print("【运行前检查】")
    print("  1. 已安装依赖：pip install langgraph langchain-core langchain-ollama pydantic")
    print("  2. 已安装 Ollama 并下载模型：ollama pull qwen3.5:2b")
    print("  3. 示例 1 无需 LLM 即可运行（规则模拟）")
    print()

    # dev_team_demo()
    supervisor_with_llm()

    print("=" * 70)
    print("  回顾：Supervisor = 包工头派活，工人干完回来汇报，包工头再派下一个")
    print("  三种多 Agent 模式对比：")
    print("    Agent-as-Tool  — 主 Agent 掌控全程，子 Agent 当工具调用")
    print("    Handoff        — 控制权接力传递，交出就不管了")
    print("    Supervisor     — 主管分配+审查，工人完成后回到主管")
    print("=" * 70)
    print("  🎉 多智能体模块学习完成！")
    print("  回顾整个 LangGraph 课程：")
    print("    01_introduction    → LangGraph 是什么")
    print("    02_state/branching → 状态管理 + 条件分支")
    print("    03_agent_loop      → ReAct Agent 循环（核心）")
    print("    04_workflows       → 工作流模式（链/路由/并行/改进）")
    print("    05_practical       → 综合实战（智能问答 Agent）")
    print("    06_multi_agent     → 多智能体（Agent-as-Tool/Handoff/Supervisor）")
    print("=" * 70 + "\n")
