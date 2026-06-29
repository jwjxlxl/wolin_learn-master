# =============================================================================
# ReAct 模式 — 推理 + 行动
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解 ReAct 的核心思想：推理(Reasoning)和行动(Acting)交替进行
#   ✅ 手动实现 ReAct 循环：Thought → Action → Observation → 再思考
#   ✅ 用 LangGraph StateGraph 构建 ReAct Agent
#   ✅ 理解手动循环 vs 框架自动化的差异
#
# 运行前检查：
# 1. 已安装依赖：pip install langgraph langchain-core langchain-ollama
# 2. 已安装 Ollama 并下载模型：ollama pull qwen3.5:2b
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
from langchain_core.tools import tool
from utils.model_utils import get_model
from helpers import simulate_search, parse_react_response


# =============================================================================
# 核心概念：ReAct（Reasoning + Acting）
# =============================================================================
"""
提出背景：Princeton 与 Google Research 2022 年论文《ReAct:
Synergizing Reasoning and Acting in Language Models》。

核心思想：在推理(Reasoning)和外部行动(Acting,比如调用搜索引擎或API)
之间交替进行。ReAct 比 CoT、Self-Ask 更全能，因为它不仅是推理模式，
还内建了与外部世界交互的闭环。

生活化比喻：
  CoT = 学生在纸上一步步推导公式 → 只思考，不动手
  Self-Ask = 学生不断反问自己 → 只思考，不动手
  ReAct = 学生先思考要查什么 → 动手去查 → 根据结果再思考 → 再动手
          = "既动脑又动手"

ReAct 循环格式：
  Thought: 我需要知道xxx才能回答 → 我想知道...
  Action:  调用工具名称（如 search、calculate）
  Observation: 工具返回结果
  → 回到 Thought，直到得出最终答案

适用场景：需要与外部世界交互的复杂问题（查天气、查数据、计算等）
"""


# =============================================================================
# 示例 1: Prompt 版 ReAct — 手动实现 Thought→Action→Observation 循环
# =============================================================================

def prompt_react():
    """
    纯 Prompt 版本 ReAct：不依赖 LangGraph，手动实现循环。

    通过系统 prompt 定义 Thought/Action/Observation 格式，
    代码负责：调用 LLM → 解析响应 → 执行工具 → 把结果喂回去 → 继续

    示例：回答"杭州昨天的天气温度比水的沸点低多少度？"
      需要：查天气 → 获取水的沸点 → 计算差值
    """
    print(f"\n-- 示例 1: Prompt 版 ReAct — 手动循环")

    model = get_model()
    if model is None:
        print("  【跳过】请安装 Ollama 并下载模型：ollama pull qwen3.5:2b")
        return

    # --- 定义工具 ---
    def get_weather(city: str) -> str:
        return simulate_search(f"{city}天气")

    def get_fact(entity: str) -> str:
        return simulate_search(entity)

    def calculate(expression: str) -> str:
        """简单计算器（实际应更复杂）"""
        print(f"  [计算器] 计算: {expression}")
        try:
            # 仅支持简单算术，实际应用中应使用更安全的计算
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception:
            return f"无法计算: {expression}"

    tools = {
        "get_weather": get_weather,
        "get_fact": get_fact,
        "calculate": calculate,
    }

    # --- ReAct 系统提示 ---
    react_system = """你是一个使用工具来回答问题的助手。你必须严格按照以下格式工作：

在每一步中，你必须按顺序输出：
  Thought: 我现在需要做什么？（解释你的思考过程）
  Action: 工具名称（从以下选择：get_weather, get_fact, calculate）
  Action Input: 工具的输入参数

当工具返回 Observation 后，你会继续输出 Thought...

如果你有足够信息直接回答问题，输出：
  Thought: 我已经有了答案
  Final Answer: 最终回答

可用工具：
  - get_weather(city): 查询某个城市的天气
  - get_fact(entity): 查询某个事实（如"水的沸点"）
  - calculate(expression): 计算数学表达式（如"100 - 15"）"""

    question = "杭州昨天的天气温度比水的沸点低多少度？"

    print(f"  问题：{question}")
    print()

    # --- 手动 ReAct 循环 ---
    conversation = [
        {"role": "system", "content": react_system},
        {"role": "user", "content": question},
    ]

    max_steps = 8
    for step in range(max_steps):
        # 调用 LLM
        response = model.invoke(
            [{"role": m["role"], "content": m["content"]} for m in conversation]
        )
        response_text = response.content

        print(f"  --- 第 {step + 1} 步 ---")
        # 解析响应
        parsed = parse_react_response(response_text)
        print(f"    思考: {parsed['thought'][:100]}...")

        # 检查是否已有最终答案
        if parsed.get("final_answer"):
            print(f"\n  ✅ 最终答案: {parsed['final_answer']}")
            break

        # 需要执行工具
        if parsed.get("action"):
            action_name = parsed["action"]
            action_input = parsed.get("action_input", "")
            print(f"    行动: {action_name}({action_input})")

            tool_func = tools.get(action_name)
            if tool_func:
                observation = tool_func(action_input) if action_input else tool_func()
            else:
                observation = f"未知工具: {action_name}"

            print(f"    观察: {observation}")

            # 把完整的 Thought→Action→Observation 加入对话
            conversation.append({
                "role": "assistant",
                "content": response_text,
            })
            conversation.append({
                "role": "user",
                "content": f"Observation: {observation}",
            })
        else:
            # 模型没有给出结构化响应
            print(f"    模型直接回答: {response_text[:200]}")
            break
        print()

    print(f"  【循环完成】共 {step + 1} 步")


# =============================================================================
# 示例 2: LangGraph 版 ReAct — 用 StateGraph 构建
# =============================================================================

def langgraph_react():
    """
    LangGraph 版 ReAct：用 StateGraph 构建自动循环。

    与示例 1 相同的问题，对比代码量差异。
    核心区别：手动循环 → 框架自动处理循环。
    """
    print(f"\n-- 示例 2: LangGraph 版 ReAct — 框架自动循环")

    try:
        from langgraph.graph import StateGraph, START, END
        from langgraph.graph.message import add_messages
        from langchain_core.messages import HumanMessage, ToolMessage
        from typing import Annotated
        from typing_extensions import TypedDict
    except ImportError:
        print("  【跳过】请安装 langgraph：pip install langgraph")
        return

    model = get_model()
    if model is None:
        print("  【跳过】请安装 Ollama 并下载模型：ollama pull qwen3.5:2b")
        return

    # --- 定义工具 ---
    @tool
    def get_weather(city: str) -> str:
        """查询某个城市的天气"""
        return simulate_search(f"{city}天气")

    @tool
    def get_fact(entity: str) -> str:
        """查询某个事实"""
        return simulate_search(entity)

    @tool
    def calculate(expression: str) -> str:
        """计算数学表达式"""
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception:
            return f"无法计算: {expression}"

    tools = [get_weather, get_fact, calculate]
    tools_by_name = {t.name: t for t in tools}
    model_with_tools = model.bind_tools(tools)

    # --- 构建 StateGraph ---
    class ReactState(TypedDict):
        messages: Annotated[list, add_messages]

    def llm_node(state: ReactState):
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        if response.content:
            print(f"  [LLM] {response.content[:80]}...")
        return {"messages": [response]}

    def tool_node(state: ReactState):
        last_msg = state["messages"][-1]
        results = []
        for tc in last_msg.tool_calls:
            func = tools_by_name[tc["name"]]
            result = func.invoke(tc["args"])
            print(f"  [工具] {tc['name']}({tc['args']}) → {result}")
            results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        return {"messages": results}

    def should_continue(state: ReactState):
        last_msg = state["messages"][-1]
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            return "tool_node"
        return END

    graph = (
        StateGraph(ReactState)
        .add_node("llm", llm_node)
        .add_node("tool_node", tool_node)
        .add_edge(START, "llm")
        .add_conditional_edges("llm", should_continue, ["tool_node", END])
        .add_edge("tool_node", "llm")
        .compile()
    )

    # --- 运行 ---
    question = "杭州昨天的天气温度比水的沸点低多少度？"
    print(f"  问题：{question}")
    print()

    result = graph.invoke({"messages": [HumanMessage(content=question)]})
    final_msg = result["messages"][-1]
    print(f"\n  ✅ 最终答案: {final_msg.content}")
    print(f"  【消息总数】{len(result['messages'])} 条")

    print(f"\n  【对比】")
    print(f"    Prompt 版: 手动写 for 循环 + 解析 + 工具调用 → ~50 行")
    print(f"    LangGraph 版: 定义节点 + 边 + compile() → ~40 行 + 自动循环")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  ReAct（Reasoning + Acting）模式")
    print("  既思考又动手：Thought → Action → Observation → 再思考")
    print("=" * 70 + "\n")

    prompt_react()
    # langgraph_react()

    print("\n" + "=" * 70)
    print("  接下来学习：plan_and_execute.py（Plan-and-Execute 计划与执行）")
    print("=" * 70 + "\n")
