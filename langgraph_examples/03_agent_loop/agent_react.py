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
# 1. 已安装依赖：pip install langgraph langchain-core langchain-ollama
# 2. 已安装 Ollama 并下载模型：ollama pull qwen3.5:2b
# 3. （可选）配置 .env 中的 ALIYUN_API_KEY 使用云端模型
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
from utils.model_utils import get_model


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
# 示例 1: 最简 Agent — 数学计算
# =============================================================================

def simple_math_agent():
    """
    最简 ReAct Agent：能用 add/multiply/divide 工具的 AI。

    START → llm_call → (有 tool_calls?) → tool_node → 回到 llm_call
                          ↘ (无) ──────────────────────→ END

    关键点：
      - @tool 装饰器把普通函数变成 AI 可调用的工具
      - bind_tools 让模型知道有哪些工具可用
      - tool_node → llm_call 这条边形成了循环！
      - 直到模型不再有 tool_calls，循环才终止
    """
    print(f"\n-- 示例 1: 最简 ReAct Agent — 数学计算")

    # 1. 定义工具
    @tool
    def add(a: int, b: int) -> int:
        """计算两个数的和"""
        result = a + b
        print(f"  [工具: add] {a} + {b} = {result}")
        return result

    @tool
    def multiply(a: int, b: int) -> int:
        """计算两个数的乘积"""
        result = a * b
        print(f"  [工具: multiply] {a} × {b} = {result}")
        return result

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

    # 2. 获取模型并绑定工具（默认使用 Ollama 本地模型，免费无需 API Key）
    model = get_model()
    if model is None:
        print("  【跳过】请安装 Ollama 并下载模型：ollama pull qwen3.5:2b")
        return
    model_with_tools = model.bind_tools(tools)

    # 3. 定义状态
    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]

    # 4. 定义节点
    def llm_call(state: AgentState):
        """LLM 节点：调用带工具的模型"""
        messages = state["messages"]
        print(f"  [节点: llm_call] 调用模型，消息数: {len(messages)}")
        response = model_with_tools.invoke(messages)
        if response.content:
            print(f"    模型回复: {response.content[:80]}...")
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"    模型要调用工具: {[tc['name'] for tc in response.tool_calls]}")
        return {"messages": [response]}

    def tool_node(state: AgentState):
        """工具节点：执行模型请求的工具调用"""
        messages = state["messages"]
        last_message = messages[-1]
        results = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            tool_func = tools_by_name[tool_name]
            result = tool_func.invoke(tool_args)
            results.append(ToolMessage(content=str(result), tool_call_id=tool_id))

        return {"messages": results}

    def should_continue(state: AgentState):
        """路由函数：检查模型是否要调用工具"""
        last_message = state["messages"][-1]
        has_tool_calls = hasattr(last_message, 'tool_calls') and last_message.tool_calls
        if has_tool_calls:
            print(f"  [路由] 需要调用工具: {[tc['name'] for tc in last_message.tool_calls]} → 走 tool_node")
            return "tool_node"
        else:
            print(f"  [路由] 无需工具 → 结束")
            return END

    # 5. 构建图 — 经典的 ReAct Agent 循环
    agent = (
        StateGraph(AgentState)
        .add_node("llm_call", llm_call)
        .add_node("tool_node", tool_node)
        .add_edge(START, "llm_call")
        .add_conditional_edges("llm_call", should_continue, ["tool_node", END])
        .add_edge("tool_node", "llm_call")  # ★ 循环的关键：工具执行后回到 LLM
        .compile()
    )

    # 6. 测试
    questions = [
        "3 加 4 等于多少？",
        "5 乘以 6 等于多少？",
        "20 除以 4 等于多少？",
    ]

    for q in questions:
        print(f"【用户提问】{q}")
        result = agent.invoke({"messages": [HumanMessage(content=q)]})
        final_msg = result["messages"][-1]
        if hasattr(final_msg, 'content') and final_msg.content:
            print(f"【Agent 回答】{final_msg.content}")
        print()


# =============================================================================
# 示例 2: Agent 多工具组合 — 单次对话使用多个工具
# =============================================================================

def multi_tool_agent():
    """
    多工具组合：一个复杂问题触发多次工具调用。

    演示点：
      - 单次问题触发多次工具调用（先加、再乘）
      - Agent 自动决定工具调用的顺序
      - 消息流如何在循环中传递
    """
    print(f"\n-- 示例 2: 多工具组合 — Agent 自动编排")

    # 1. 定义工具
    @tool
    def add(a: int, b: int) -> int:
        """计算两个数的和"""
        return a + b

    @tool
    def multiply(a: int, b: int) -> int:
        """计算两个数的乘积"""
        return a * b

    tools = [add, multiply]
    tools_by_name = {t.name: t for t in tools}

    model = get_model()
    if model is None:
        print("  【跳过】请安装 Ollama 并下载模型：ollama pull qwen3.5:2b")
        return
    model_with_tools = model.bind_tools(tools)

    # 2. 定义图和节点（与示例 1 同构）
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

    # simple_math_agent()
    multi_tool_agent()

    print("=" * 70)
    print("  接下来学习：04_workflows/（工作流模式）")
    print("=" * 70 + "\n")
