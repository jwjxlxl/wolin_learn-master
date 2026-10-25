# =============================================================================
# Agent 循环 — ReAct 模式（LangGraph 最核心的应用）
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解 ReAct 循环：思考 → 行动 → 观察 → 再思考
#   ✅ 用 @tool 装饰器定义 AI 可调用的工具
#   ✅ 用 bind_tools() 把工具绑定到模型
#   ✅ 构建带循环的图（tool_node → llm_node 形成闭环）
#   ✅ 理解 AIMessage.tool_calls 和 ToolMessage 的消息流转
#
# 运行前检查：
# 1. 已安装依赖：pip install langgraph langchain-core langchain-openai
# 2. 配置 .env 中的 ALIYUN_API_KEY（使用云端 Qwen 模型）
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
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from utils.model_utils import get_qwen_client


# =============================================================================
# 核心概念：ReAct Agent 循环
# =============================================================================
"""
什么是 ReAct Agent？

  ReAct = Reasoning（推理） + Acting（行动）

  流程：
    START → LLM 思考 → (需要工具?) → 执行工具 → 观察结果 → 回到 LLM 再思考
                           ↘ (不需要) ──────────────────────────→ END

  消息流转（这是理解 Agent 的关键）：
    用户输入 → LLM → AIMessage(tool_calls=[...]) → 执行工具 → ToolMessage(...) →
    LLM 看到 ToolMessage → 再次推理 → 最终 AIMessage(content="...") → END

  三大组件：
    1. bind_tools() — 把工具"告诉"模型，模型会在 tool_calls 字段列出想调用的工具
    2. llm_call 节点 — 调用带工具的模型，返回 AIMessage（可能含 tool_calls）
    3. tool_node 节点 — 执行 AIMessage 中的 tool_calls，返回 ToolMessage

  生活化比喻：
    LLM  = "老板"（做决策，但不亲自干活）
    Tool = "员工"（听老板指挥，执行具体任务）
    ReAct 循环 = "老板想→指派任务→员工干活→汇报结果→老板再想..."
"""


# =============================================================================
# 核心概念：手动构建 vs 预构建（create_react_agent）
# =============================================================================
"""
两种构建 ReAct Agent 的方式

方式 A：预构建 create_react_agent()（推荐入门）
  from langgraph.prebuilt import create_react_agent
  agent = create_react_agent(model, tools)
  result = agent.invoke({"messages": [HumanMessage("...")]})
  一行搞定！

方式 B：手动 StateGraph（理解原理，即下方示例）
  graph = StateGraph(...)
    .add_node("llm_call", llm_call)
    .add_node("tool_node", tool_node)
    .add_edge(START, "llm_call")
    .add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    .add_edge("tool_node", "llm_call")
    .compile()

create_react_agent() 底层做的事（就是下方示例手动写的代码）：
  1. 用 MessagesState 管理消息（Annotated[list, add_messages]）
  2. 创建 llm_call 节点（调用带工具的模型）
  3. 创建 tool_node 节点（执行工具调用）
  4. 创建条件路由（有 tool_calls → tool_node，没有 → END）
  5. 编译成 graph

何时用哪种？
  用 create_react_agent()：
    - 快速原型开发
    - 标准的 ReAct Agent（不需要自定义逻辑）
    - 只想关注 Tool 定义，不想管图结构

  手动构建 StateGraph：
    - 需要自定义状态（不只是 messages）
    - 需要多个 LLM 节点
    - 需要在工具执行前后加自定义逻辑
    - 需要混合工作流模式（如：先路由、再 Agent、再汇总）
"""


# =============================================================================
# 示例: Agent 多工具组合 — 单次对话使用多个工具
# =============================================================================

def multi_tool_agent():
    """
    多工具组合：一个复杂问题触发多次工具调用。

    演示点：
      - 单次问题触发多次工具调用（先加、再乘）
      - Agent 自动决定工具调用的顺序
      - 消息流如何在循环中传递
    """
    print(f"\n-- 示例: 多工具组合 — Agent 自动编排")

    # 1. 定义工具
    @tool
    def add(a: int, b: int) -> int:
        """计算两个数的和"""
        return a + b

    @tool
    def multiply(a: int, b: int) -> int:
        """计算两个数的乘积"""
        return a * b

    @tool
    def divide(a: int, b: int) -> float:
        """计算 a 除以 b 的商"""
        if b == 0:
            return "错误：除数不能为 0"
        result = a / b
        print(f"  [工具: divide] {a} ÷ {b} = {result}")
        return result

    tools = [add, multiply, divide]
    tools_by_name = {t.name: t for t in tools}

    model = get_qwen_client()
    if model is None:
        print("  【跳过】请先在 .env 配置 ALIYUN_API_KEY")
        return
    model_with_tools = model.bind_tools(tools)

    # 2. 定义图和节点
    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]

    def llm_call(state: AgentState):
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    def tool_node(state: AgentState):
        last_msg = state["messages"][-1]
        results = []
        for tc in last_msg.tool_calls:
            func = tools_by_name[tc["name"]]
            results.append(ToolMessage(
                content=str(func.invoke(tc["args"])),
                tool_call_id=tc["id"]
            ))
        return {"messages": results}

    def should_continue(state: AgentState):
        last_msg = state["messages"][-1]
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            return "tool_node"
        return END

    agent = (
        StateGraph(AgentState)
        .add_node("llm_call", llm_call)
        .add_node("tool_node", tool_node)
        .add_edge(START, "llm_call")
        .add_conditional_edges("llm_call", should_continue, ["tool_node", END])
        .add_edge("tool_node", "llm_call")
        .compile()
    )

    # 3. 测试复杂问题（需要多次工具调用）
    print("【用户提问】(3 + 4) × 5 等于多少？")
    result = agent.invoke({"messages": [HumanMessage(content="(3 + 4) × 5 等于多少？")]})
    final_msg = result["messages"][-1]
    print(f"【Agent 回答】{final_msg.content}")
    print(f"【消息总数】{len(result['messages'])} 条（包含中间工具调用）")
    print()


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  Agent 循环 — ReAct 模式")
    print("  理解思考→行动→观察→再思考的核心循环")
    print("=" * 70 + "\n")

    multi_tool_agent()

    print("=" * 70)
    print("  总结：")
    print("    本文件手动构建 ReAct Agent，帮助你理解底层原理")
    print("    生产环境可直接用 create_react_agent(model, tools) 一行搞定")
    print("  接下来学习：04_workflows/（工作流模式）")
    print("=" * 70 + "\n")
