# =============================================================================
# Agent 记忆 — InMemorySaver：让 Agent 记住多轮对话
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 用 InMemorySaver + thread_id 让 Agent 记住对话历史
#   ✅ 理解 thread_id = "对话分组标识"，不同线程互不干扰
#   ✅ 用 @before_model 中间件修剪过长消息
# =============================================================================

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)


"""
Agent 记忆 vs 手动记忆：

  前面学的（05_memory）: 手动用 list 保存历史
  Agent 记忆:            InMemorySaver 自动持久化，通过 thread_id 分组

  thread_id 就像聊天软件的"会话 ID"——同一个会话的消息自动关联，
  不同会话互不干扰。
"""

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langchain_ollama import ChatOllama


# =============================================================================
# 示例 1: InMemorySaver — 同一线程 Agent 自动记住历史
# =============================================================================

def simple_memory_demo():
    """
    checkpointer=InMemorySaver() + thread_id — Agent 自动记住对话。

    关键参数:
    - checkpointer: 记忆的存储后端（InMemorySaver = 内存，开发测试用）
    - config={"configurable": {"thread_id": "1"}}: 线程标识
    - 同一 thread_id 的多次 invoke() 共享历史
    """
    print(f"\n-- 示例 1: InMemorySaver — Agent 自动记忆")

    @tool
    def get_weather(city: str) -> str:
        """查询指定城市的天气。"""
        return {"北京": "晴，25°C", "上海": "多云，28°C"}.get(city, f"暂无 {city} 数据")

    model = ChatOllama(model="qwen3.5:2b")
    agent = create_agent(model=model, tools=[get_weather],
                         system_prompt="你是一个有用的助手，请简洁回答。",
                         checkpointer=InMemorySaver())

    config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    # 第 1 轮: 告诉 Agent 用户信息
    r = agent.invoke({"messages": [HumanMessage("你好，我叫小明，我在北京。")]}, config)
    print(f"[第1轮] {r['messages'][-1].content}")

    # 第 2 轮: Agent 应该记住名字和城市
    r = agent.invoke({"messages": [HumanMessage("我所在城市天气怎么样？")]}, config)
    print(f"[第2轮] {r['messages'][-1].content}")
    print("  ↑ Agent 记住了'北京'，自动查了天气！")

    # 换个线程 —— Agent 不记得小明
    other_config: RunnableConfig = {"configurable": {"thread_id": "2"}}
    r = agent.invoke({"messages": [HumanMessage("还记得我的名字吗？")]}, other_config)
    print(f"[线程2] {r['messages'][-1].content}")
    print("  ↑ 不同线程，记忆隔离")


# =============================================================================
# 示例 2: 修剪消息 — @before_model + RemoveMessage
# =============================================================================

def trim_messages_demo():
    """
    对话太长时用 @before_model 中间件自动修剪旧消息。

    策略: 保留第一条（系统提示）+ 最近 K 条，丢弃中间的。
    这能防止上下文超限，同时保留关键信息。
    """
    from langchain.agents.middleware import before_model, AgentState
    from langgraph.runtime import Runtime
    from langchain_core.messages import RemoveMessage
    from langgraph.graph.message import REMOVE_ALL_MESSAGES

    print(f"\n-- 示例 2: 消息修剪 — 自动丢弃旧消息")

    KEEP = 3  # 保留最近 3 条

    @before_model
    def trim(state: AgentState, runtime: Runtime) -> dict | None:
        messages = state["messages"]
        if len(messages) <= KEEP + 1:
            return None
        first = messages[0]  # 系统提示
        recent = messages[-KEEP:]
        removed = len(messages) - len([first] + recent)
        print(f"  [trim] 消息 {len(messages)} → {len([first] + recent)}（修剪 {removed} 条）")
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), first] + recent}

    @tool
    def get_weather(city: str) -> str:
        """查询天气。"""
        return {"北京": "晴，25°C"}.get(city, "暂无数据")

    model = ChatOllama(model="qwen3.5:2b")
    agent = create_agent(model=model, tools=[get_weather],
                         system_prompt="你是一个有用的助手。",
                         middleware=[trim], checkpointer=InMemorySaver())

    config: RunnableConfig = {"configurable": {"thread_id": "trim-1"}}
    for i, msg in enumerate(["你好，我叫小明", "我在北京", "北京天气？", "上海呢？",
                              "还记得我的名字吗？"], 1):
        r = agent.invoke({"messages": [HumanMessage(content=msg)]}, config)
        print(f"  [第{i}轮 消息数={len(r['messages'])}] {r['messages'][-1].content}")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n>>> 09_agent/agent_memory — Agent 记忆管理\n")

    simple_memory_demo()
    trim_messages_demo()

    # 接下来学习: middleware.py（中间件拦截器）
