# =============================================================================
# create_agent() vs 手动构建 — 两种 Agent 构建方式对比
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 使用 langgraph 内置的 create_agent() 一行创建 Agent
#   ✅ 理解 create_agent() 底层做的事（就是 agent_react.py 的内容！）
#   ✅ 知道何时用 create_agent()，何时手动构建 StateGraph
#
# 运行前检查：
# 1. 已安装依赖：pip install langgraph langchain-core langchain-ollama
# 2. 已安装 Ollama 并下载模型：ollama pull qwen3.5:2b
# =============================================================================

import sys
import os
import io

from langchain.agents import create_agent

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 确保项目根目录可导入（用于 utils/ 共享模块）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import add_messages
from utils.model_utils import get_model


# =============================================================================
# 核心概念：create_react_agent() 是什么？
# =============================================================================
"""
两种构建 Agent 的方式

方式 A：create_react_agent()（推荐入门）
  agent = create_react_agent(model, tools)
  result = agent.invoke({"messages": [HumanMessage("...")]})
  一行搞定！

方式 B：手动 StateGraph（理解原理）
  graph = StateGraph(...)
    .add_node("llm", llm_call)
    .add_node("tools", tool_node)
    .add_edge(START, "llm")
    .add_conditional_edges("llm", router, ["tools", END])
    .add_edge("tools", "llm")
    .compile()
  → 已经在 agent_react.py 中学过

create_react_agent() 底层做的事：
  1. 用 MessagesState 管理消息（Annotated[list, add_messages]）
  2. 创建 llm_call 节点（调用带工具的模型）
  3. 创建 tool_node 节点（执行工具调用）
  4. 创建条件路由（有 tool_calls → tool_node，没有 → END）
  5. 编译成 graph

  就是 agent_react.py 中我们手动写的所有代码！

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
# 示例 1: 用 create_react_agent() 创建 Agent（推荐方式）
# =============================================================================

def agent_with_prebuilt():
    """
    使用 langgraph 内置的 create_react_agent() 一行创建 Agent。

    对比 agent_react.py 中 50+ 行的手动构建，这种方式简洁得多。
    """
    print(f"\n-- 示例 1: 用 create_react_agent() 创建 Agent")

    # 1. 定义工具（和 agent_react.py 完全一样）
    @tool
    def add(a: int, b: int) -> int:
        """计算两个数的和"""
        return a + b

    @tool
    def multiply(a: int, b: int) -> int:
        """计算两个数的乘积"""
        return a * b

    @tool
    def devile(a: int, b: int) -> int:
        """计算两个数的相除的结果"""
        return a / b

    @tool
    def decus(a: int, b: int) -> int:
        """计算两个数的减法"""
        return a - b

    tools = [add, multiply, devile, decus]

    # 2. 获取模型
    model = get_model("qwen")
    if model is None:
        print("  【跳过】请安装 Ollama 并下载模型：ollama pull qwen3.5:2b")
        return

    # 3. ★ 一行创建 Agent！底层就是 agent_react.py 中手写的所有逻辑
    agent = create_agent(model, tools)

    # 4. 测试
    print("  【测试】(3 + 4) × 5 = ?")
    result = agent.invoke({"messages": [HumanMessage(content="(3 + 4) × 5 + (12.5 - 2.3) / 3 等于多少？")]})

    # 提取最终回答
    for msg in result["messages"]:
        if hasattr(msg, 'content') and msg.content and msg.type == "ai":
            print(f"  【回答】{msg.content}")
            break

    # 展示消息流
    print(f"\n  【消息流追踪】（共 {len(result['messages'])} 条消息）")
    for i, msg in enumerate(result["messages"]):
        msg_type = msg.__class__.__name__
        if msg_type == "HumanMessage":
            preview = msg.content[:40]
        elif msg_type == "AIMessage":
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                preview = f"调用工具: {[tc['name'] for tc in msg.tool_calls]}"
            else:
                preview = msg.content[:40] if msg.content else "(空)"
        elif msg_type == "ToolMessage":
            preview = f"结果: {msg.content}"
        else:
            preview = str(msg)[:40]
        print(f"    [{i}] {msg_type}: {preview}")
    print()


# =============================================================================
# 示例 2: 手动构建相同 Agent（对比理解）
# =============================================================================

def agent_manual():
    """
    手动构建和 create_react_agent() 完全相同的 Agent。

    目的：展示 create_react_agent() 底层做了什么，
    帮助理解它不是"魔法"，只是把常见模式封装了。

    如果已经学过 agent_react.py，这个结构和那里完全一样。
    """
    print(f"\n-- 示例 2: 手动构建相同 Agent（对比）")

    @tool
    def add(a: int, b: int) -> int:
        """计算两个数的和"""
        return a + b

    @tool
    def multiply(a: int, b: int) -> int:
        """计算两个数的乘积"""
        return a * b

    tools = [add, multiply]

    model = get_model("qwen")
    if model is None:
        print("  【跳过】请安装 Ollama 并下载模型：ollama pull qwen3.5:2b")
        return
    # 模型绑定工具列表
    model_with_tools = model.bind_tools(tools)

    # 手动构建 — 这就是 create_react_agent() 内部做的事
    from typing import Annotated
    from typing_extensions import TypedDict
    from langchain_core.messages import ToolMessage

    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]

    # 大模型的节点
    def llm_call(state: AgentState):
        response = model_with_tools.invoke(state["messages"])
        print(f"LLM response: {response}")
        return {"messages": [response]}

    # 工具执行节点 tools_by_name = {"add":add, "multiply":multiply}
    tools_by_name = {t.name: t for t in tools}
    def tool_executor(state: AgentState):
        last_msg = state["messages"][-1]
        results = []
        for tc in last_msg.tool_calls:
            func = tools_by_name[tc["name"]]
            print(f"Tool call: {tc["name"]}")
            results.append(ToolMessage(
                content=str(func.invoke(tc["args"])),
                tool_call_id=tc["id"]
            ))
        return {"messages": results}


    # 条件路由, 作用是
    def router(state: AgentState):
        last_msg = state["messages"][-1]
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            return "tool_node"
        return END

    graph = (
        StateGraph(AgentState)
        .add_node("llm_call", llm_call)
        .add_node("tool_node", tool_executor)
        .add_edge(START, "llm_call")
        .add_conditional_edges("llm_call", router, ["tool_node", END])
        .add_edge("tool_node", "llm_call")
        .compile()
    )

    print("  【测试】同样的输入 → 同样的结果")
    result = graph.invoke({"messages": [HumanMessage(content="(3 + 4) × 5 + (12.5 - 2.3) / 3 等于多少？")]})

    final_msg = result["messages"][-1]
    print(f"  【回答】{final_msg.content if hasattr(final_msg, 'content') and final_msg.content else '（查看工具调用结果）'}")

    print(f"\n  【对比总结】")
    print(f"    create_react_agent(): 1 行代码（隐藏了图结构细节）")
    print(f"    手动 StateGraph:     ~20 行代码（完全控制每个节点和边）")
    print(f"    效果：完全相同！create_react_agent() 只是帮我们写了那 ~20 行")
    print()


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  create_react_agent() vs 手动构建 — 对比学习")
    print("  理解内置函数的底层原理")
    print("=" * 70 + "\n")

    '''
        agent_with_prebuilt():
        我们直接调用Agent创建的过程，底层运行的逻辑和原理，不关注
        它能帮忙我们完成ReAct模式的Agent
    '''
    # agent_with_prebuilt()
    '''
        agent_manual():
        手动创建一个ReAct模式的Agent，用Langgraph来实现它的每一个步骤，节点，边，router判断都是自己手动实现
        因为Langgraph是最底层的实现
    '''
    agent_manual()

    print("=" * 70)
    print("  总结：")
    print("    - 入门/快速原型 → create_react_agent()")
    print("    - 自定义复杂流程 → 手动 StateGraph")
    print("    - create_react_agent() 底层 = agent_react.py 的手动代码")
    print("=" * 70 + "\n")
