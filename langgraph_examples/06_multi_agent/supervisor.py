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
# 示例 : LLM 版 — 真正的 Supervisor（LLM 做决策和审查）
# =============================================================================

def supervisor_with_llm():
    """
    用 LLM 做任务分配和审查的 Supervisor 模式。

    START → 项目经理（分配任务）
                ├── researcher → 回到项目经理
                ├── coder → 回到项目经理
                └── reviewer → 回到项目经理 →（三次后完成）→ END

    关键点：
      - 项目经理决定"下一个是谁"
      - 每个工人干完必回项目经理（Command(goto="supervisor")）
      - 迭代计数器防止无限循环
    """
    print(f"\n-- 示例 2: LLM 版 — 真正的 Supervisor")

    from utils.model_utils import get_model
    from pydantic import BaseModel, Field
    from typing import Literal

    model = get_model("qwen")
    if model is None:
        return

    # ===== Supervisor 结构化输出 =====
    class SupervisorDecision(BaseModel):
        next_worker: Literal["researcher", "coder", "reviewer", "FINISH"] = Field(
            description="下一个要分配的工人，或 FINISH 表示完成"
        )
        reasoning: str = Field(description="为什么分配给这个工人")

    supervisor_model = model.with_structured_output(SupervisorDecision)

    # ===== Supervisor 状态 =====
    # 在Agent之间进行数据传输和存储的管理
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
        "搭建一个能搜索公司知识库的智能问答 Agent"
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

    supervisor_with_llm()

    print("=" * 70)
    print("  回顾：Supervisor = 包工头派活，工人干完回来汇报，包工头再派下一个")
    print("  三种多 Agent 模式对比：")
    print("    Agent-as-Tool  — 主 Agent 掌控全程，子 Agent 当工具调用")
    print("    Handoff        — 控制权接力传递，交出就不管了")
    print("    Supervisor     — 主管分配+审查，工人完成后回到主管")
    print("=" * 70)
