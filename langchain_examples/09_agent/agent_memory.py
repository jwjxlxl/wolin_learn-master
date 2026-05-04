import sys
import os
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import model_untils

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, after_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from typing import Any


# =============================================================================
# 短期记忆 Short-term Memory
# =============================================================================
#
# 用途：教学演示 - 使用 LangChain 短期记忆管理对话历史
#
# 核心概念：
#   - 短期记忆 = 让 Agent 记住单个线程/对话中的先前交互
#   - 线程(thread) = 在一个会话中组织多次交互，类似邮件中的对话分组
#   - checkpointer = 短期记忆的持久化后端（内存或数据库）
#
# 为什么需要记忆管理？
#   - LLM 上下文窗口有限，长对话可能超出限制
#   - 即使模型支持长上下文，过多消息也会导致"分心"、变慢、成本增加
#   - 需要用技术来移除或"遗忘"陈旧信息
#
# 常见策略：
#   - 修剪消息(Trim)：保留最近 N 条消息，丢弃旧的
#   - 删除消息(Delete)：从状态中永久删除特定消息
#   - 总结消息(Summarize)：将旧消息压缩为摘要，保留关键信息
# =============================================================================

# -----------------------------------------------------------------------------
# 运行前检查
# -----------------------------------------------------------------------------
# 1. 已配置阿里云 API Key（.env 文件）
# -----------------------------------------------------------------------------


# =============================================================================
# 第一部分：理解短期记忆
# =============================================================================
"""
什么是短期记忆（Short-term Memory）？

🧠 定义
   短期记忆 = 让 Agent 记住当前对话中的先前交互
   类似人类的"工作记忆"，在对话过程中保持上下文

🔄 工作原理
   1. 用户发送消息 → Agent 处理 → 响应
   2. 所有消息存储在 Agent 状态(state)中
   3. 下次调用时，Agent 能看到之前的完整对话
   4. 通过 checkpointer 持久化，线程可随时恢复

⚠️ 挑战
   - 上下文窗口有限 → 长对话超出限制
   - 过多消息导致"分心" → 回答质量下降
   - Token 消耗增加 → 响应变慢、成本上升

💡 解决方案
   修剪(Trim)    → 快速丢弃旧消息，简单高效
   删除(Delete)  → 精确删除特定消息
   总结(Summarize)→ 压缩旧消息为摘要，保留关键信息（推荐）
"""


# =============================================================================
# 示例 1: 简单用法 - 使用 InMemorySaver 添加短期记忆
# =============================================================================

def simple_memory_demo():
    """
    短期记忆的简单用法

    通过指定 checkpointer 让 Agent 拥有短期记忆：
      - InMemorySaver：内存存储，适合开发测试
      - PostgresSaver：数据库存储，适合生产环境

    关键参数：
      - checkpointer：记忆持久化后端
      - thread_id：线程标识，同一 thread_id 共享对话历史
    """

    @tool
    def get_weather(city: str) -> str:
        """查询指定城市的天气信息。"""
        weather_db = {
            "北京": "晴，25°C",
            "上海": "多云，28°C",
            "深圳": "晴，29°C",
        }
        return weather_db.get(city, f"暂无 {city} 的天气数据")

    model = model_untils.get_qwen_client()
    if model is None:
        print("【跳过】未配置阿里云 API Key，无法运行此示例")
        return

    # -------------------------------------------------------------------------
    # 创建带记忆的 Agent
    # -------------------------------------------------------------------------
    # checkpointer = 记忆持久化后端
    # InMemorySaver = 内存存储（开发测试用）
    # 生产环境请使用 PostgresSaver 等数据库后端

    agent = create_agent(
        model,
        tools=[get_weather],
        system_prompt="你是一个有用的助手，请简洁回答。",
        checkpointer=InMemorySaver(),
    )

    print("【Agent 创建成功 - 带短期记忆】")
    print(f"  checkpointer: InMemorySaver (内存存储)")
    print()

    # -------------------------------------------------------------------------
    # 同一线程的多轮对话：Agent 能记住之前的内容
    # -------------------------------------------------------------------------

    config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    # 第 1 轮：告诉 Agent 用户名字
    print(f"{'─' * 50}")
    print("【第 1 轮】告诉 Agent 用户名字")
    result_1 = agent.invoke(
        {"messages": [HumanMessage(content="你好，我叫小明，我在北京。")]},
        config,
    )
    print(f"  用户: 你好，我叫小明，我在北京。")
    print(f"  Agent: {result_1['messages'][-1].content}")
    print()

    # 第 2 轮：Agent 应该"记住"用户名字和城市
    print(f"{'─' * 50}")
    print("【第 2 轮】Agent 应该记住用户信息")
    result_2 = agent.invoke(
        {"messages": [HumanMessage(content="我所在的城市今天天气怎么样？")]},
        config,
    )
    print(f"  用户: 我所在的城市今天天气怎么样？")
    print(f"  Agent: {result_2['messages'][-1].content}")
    print(f"  [验证] Agent 是否记住了用户在北京？")
    print()

    # 第 3 轮：验证记忆延续
    print(f"{'─' * 50}")
    print("【第 3 轮】验证记忆延续")
    result_3 = agent.invoke(
        {"messages": [HumanMessage(content="你还记得我的名字吗？")]},
        config,
    )
    print(f"  用户: 你还记得我的名字吗？")
    print(f"  Agent: {result_3['messages'][-1].content}")
    print()

    # -------------------------------------------------------------------------
    # 不同线程 = 不同的对话，互不干扰
    # -------------------------------------------------------------------------
    print(f"{'─' * 50}")
    print("【不同线程】切换 thread_id，对话互不干扰")
    config_other: RunnableConfig = {"configurable": {"thread_id": "2"}}

    result_4 = agent.invoke(
        {"messages": [HumanMessage(content="你还记得我的名字吗？")]},
        config_other,
    )
    print(f"  用户(线程2): 你还记得我的名字吗？")
    print(f"  Agent(线程2): {result_4['messages'][-1].content}")
    print(f"  [验证] 线程2 的 Agent 不知道小明是谁（记忆隔离）")
    print()


# =============================================================================
# 示例 2: 自定义 Agent 记忆 - 扩展 AgentState
# =============================================================================

def customizing_memory_demo():
    """
    自定义 Agent 记忆：扩展 AgentState 添加额外字段

    默认 Agent 使用 AgentState 管理短期记忆，主要通过 messages 键管理会话历史。
    可以扩展 AgentState 添加自定义字段（如 user_id、preferences 等）。

    自定义状态模式通过 state_schema 参数传递给 create_agent。
    """

    # -------------------------------------------------------------------------
    # 定义自定义状态：扩展 AgentState，添加额外字段
    # -------------------------------------------------------------------------
    class CustomAgentState(AgentState):
        """扩展 Agent 状态，添加用户 ID 和偏好。

        这些字段会在整个对话期间保持，随 checkpointer 持久化。
        """
        user_id: str           # 用户标识
        preferences: dict      # 用户偏好

    @tool
    def get_user_info(runtime) -> str:
        """查询用户信息。"""
        # 通过 runtime 访问自定义状态
        return f"查询到用户信息"

    @tool
    def get_weather(city: str) -> str:
        """查询指定城市的天气信息。"""
        weather_db = {
            "北京": "晴，25°C",
            "上海": "多云，28°C",
            "深圳": "晴，29°C",
        }
        return weather_db.get(city, f"暂无 {city} 的天气数据")

    model = model_untils.get_qwen_client()
    if model is None:
        print("【跳过】未配置阿里云 API Key，无法运行此示例")
        return

    # -------------------------------------------------------------------------
    # 创建带自定义状态的 Agent
    # -------------------------------------------------------------------------

    agent = create_agent(
        model,
        tools=[get_weather],
        state_schema=CustomAgentState,  # 注册自定义状态模式
        system_prompt="你是一个有用的助手。如果知道用户偏好，请据此回答。",
        checkpointer=InMemorySaver(),
    )

    print("【Agent 创建成功 - 带自定义状态】")
    print(f"  自定义状态字段:")
    print(f"    - user_id: 用户标识")
    print(f"    - preferences: 用户偏好")
    print()

    # -------------------------------------------------------------------------
    # 调用时传入自定义状态
    # -------------------------------------------------------------------------

    config: RunnableConfig = {"configurable": {"thread_id": "custom-1"}}

    # 第 1 轮：传入自定义状态
    print(f"{'─' * 50}")
    print("【第 1 轮】传入自定义状态（user_id + preferences）")
    result = agent.invoke(
        {
            "messages": [HumanMessage(content="帮我查一下天气")],
            "user_id": "user_123",
            "preferences": {"city": "深圳", "language": "中文"},
        },
        config,
    )
    print(f"  用户: 帮我查一下天气")
    print(f"  Agent: {result['messages'][-1].content}")
    print(f"  [状态] user_id: {result.get('user_id', 'N/A')}")
    print(f"  [状态] preferences: {result.get('preferences', 'N/A')}")
    print()

    # 第 2 轮：自定义状态随 checkpointer 保持
    print(f"{'─' * 50}")
    print("【第 2 轮】自定义状态持久化（无需再次传入）")
    result_2 = agent.invoke(
        {"messages": [HumanMessage(content="你还记得我的偏好吗？")]},
        config,
    )
    print(f"  用户: 你还记得我的偏好吗？")
    print(f"  Agent: {result_2['messages'][-1].content}")
    print(f"  [状态] preferences: {result_2.get('preferences', 'N/A')}")
    print()


# =============================================================================
# 示例 3: 修剪消息 - 使用 @before_model 保留最近消息
# =============================================================================

def trim_messages_demo():
    """
    修剪消息(Trim Messages)：保留最近 N 条消息，丢弃旧的

    大多数 LLM 有最大上下文窗口限制。当对话变长时，需要截断消息历史。
    修剪策略：保留第一条消息（系统提示）+ 最近几条消息，丢弃中间的。

    使用 @before_model 中间件在每次模型调用前执行修剪。

    关键 API：
      - RemoveMessage(id=msg.id)：删除特定消息
      - REMOVE_ALL_MESSAGES：删除所有消息的标记
    """

    @tool
    def get_weather(city: str) -> str:
        """查询指定城市的天气信息。"""
        weather_db = {
            "北京": "晴，25°C",
            "上海": "多云，28°C",
            "深圳": "晴，29°C",
        }
        return weather_db.get(city, f"暂无 {city} 的天气数据")

    model = model_untils.get_qwen_client()
    if model is None:
        print("【跳过】未配置阿里云 API Key，无法运行此示例")
        return

    # -------------------------------------------------------------------------
    # 定义修剪中间件
    # -------------------------------------------------------------------------
    # 策略：当消息数 > keep_count 时，保留第一条消息 + 最近 keep_count 条消息
    # 使用 RemoveMessage + REMOVE_ALL_MESSAGES 实现消息替换

    KEEP_COUNT = 3  # 保留最近 3 条消息

    @before_model
    def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """修剪消息历史，保留第一条消息和最近的几条消息。

        为什么要保留第一条消息？
          - 第一条消息通常是系统提示，包含重要指令
          - 丢弃它会导致 Agent 失去基本行为指引

        修剪流程：
          1. 检查消息数量是否超过阈值
          2. 如果超过，先用 REMOVE_ALL_MESSAGES 清空所有消息
          3. 再放回第一条消息 + 最近的消息
        """
        messages = state["messages"]

        if len(messages) <= KEEP_COUNT + 1:
            return None  # 消息不多，无需修剪

        first_msg = messages[0]  # 保留第一条消息（系统提示）
        recent_messages = messages[-KEEP_COUNT:]  # 保留最近几条
        new_messages = [first_msg] + recent_messages

        removed_count = len(messages) - len(new_messages)
        print(f"  [trim_messages] 消息数 {len(messages)} → {len(new_messages)} (修剪了 {removed_count} 条)")

        # 先删除所有消息，再添加需要保留的
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages
            ]
        }

    # -------------------------------------------------------------------------
    # 创建带修剪中间件的 Agent
    # -------------------------------------------------------------------------

    agent = create_agent(
        model,
        tools=[get_weather],
        system_prompt="你是一个有用的助手，请简洁回答。",
        middleware=[trim_messages],
        checkpointer=InMemorySaver(),
    )

    print("【Agent 创建成功 - 带消息修剪】")
    print(f"  修剪策略: 保留第1条 + 最近 {KEEP_COUNT} 条")
    print()

    # -------------------------------------------------------------------------
    # 多轮对话：观察修剪效果
    # -------------------------------------------------------------------------

    config: RunnableConfig = {"configurable": {"thread_id": "trim-1"}}

    conversations = [
        "你好，我叫小明，我在北京。",
        "帮我查一下北京的天气。",
        "上海呢？",
        "今天星期几？",
        "你还记得我的名字吗？",  # 修剪后可能不记得了
    ]

    for i, msg in enumerate(conversations, 1):
        print(f"{'─' * 50}")
        print(f"【第 {i} 轮】")
        result = agent.invoke(
            {"messages": [HumanMessage(content=msg)]},
            config,
        )
        print(f"  用户: {msg}")
        print(f"  Agent: {result['messages'][-1].content}")
        print(f"  [当前消息数] {len(result['messages'])}")
        print()

    print("  [验证] 最后一个问题：修剪后 Agent 可能不记得你的名字")
    print("  [原因] 早期消息被修剪掉了，名字信息丢失")
    print("  [解决] 使用总结消息(Summarize)策略可以保留关键信息")
    print()


# =============================================================================
# 示例 4: 总结消息 - 使用 SummarizationMiddleware 压缩历史
# =============================================================================

def summarize_messages_demo():
    """
    总结消息(Summarize Messages)：将旧消息压缩为摘要

    修剪消息的问题：可能丢失重要信息（如用户名字、关键偏好）。
    总结消息的优势：将旧消息压缩为摘要，保留关键信息的同时减少 token 消耗。

    使用内置的 SummarizationMiddleware：
      - 当消息接近 token 限制时，自动对历史进行摘要
      - 摘要后保留最近的消息，用摘要替代旧消息
      - 摘要由另一个 LLM 生成，确保关键信息不丢失
    """

    from langchain.agents.middleware import SummarizationMiddleware

    @tool
    def get_weather(city: str) -> str:
        """查询指定城市的天气信息。"""
        weather_db = {
            "北京": "晴，25°C",
            "上海": "多云，28°C",
            "深圳": "晴，29°C",
        }
        return weather_db.get(city, f"暂无 {city} 的天气数据")

    model = model_untils.get_qwen_client()
    if model is None:
        print("【跳过】未配置阿里云 API Key，无法运行此示例")
        return

    # -------------------------------------------------------------------------
    # 创建带摘要中间件的 Agent
    # -------------------------------------------------------------------------
    # SummarizationMiddleware 配置：
    #   model: 用于生成摘要的模型（可以用轻量模型节省成本）
    #   max_tokens_before_summary: 超过此 token 数时触发摘要
    #   messages_to_keep: 摘要后保留最近的消息数量



    agent = create_agent(
        model,
        tools=[get_weather],
        system_prompt="你是一个有用的助手，请简洁回答。",
        middleware=[
            SummarizationMiddleware(
                model="qwen-plus",           # 用于生成摘要的模型
                max_tokens_before_summary=4000,        # 超过 4000 token 时触发摘要
                messages_to_keep=20,                   # 摘要后保留最近 20 条消息
            ),
        ],
        checkpointer=InMemorySaver(),
    )

    # -------------------------------------------------------------------------
    # 多轮对话：观察摘要效果
    # -------------------------------------------------------------------------
    # 在短对话中，摘要不会触发（消息太少）
    # 但即使如此，Agent 的记忆也是完整的

    config: RunnableConfig = {"configurable": {"thread_id": "summary-1"}}

    conversations = [
        "你好，我叫小明，我在北京。",
        "帮我查一下北京的天气。",
        "你还记得我的名字吗？",
    ]

    for i, msg in enumerate(conversations, 1):
        print(f"{'─' * 50}")
        print(f"【第 {i} 轮】")
        result = agent.invoke(
            {"messages": [HumanMessage(content=msg)]},
            config,
        )
        print(f"  用户: {msg}")
        print(f"  Agent: {result['messages'][-1].content}")
        print()

    print("  [说明] 短对话不会触发摘要，但记忆是完整的")
    print("  [说明] 当对话超过 4000 token 时，SummarizationMiddleware 会自动将旧消息压缩为摘要")
    print("  [优势] 相比修剪(Trim)，摘要(Summarize)能保留关键信息")
    print()


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  Agent 短期记忆 - Short-term Memory")
    print("  说明：让 Agent 记住当前对话中的先前交互")
    print("=" * 70 + "\n")

    # 运行示例
    # simple_memory_demo()
    # customizing_memory_demo()
    # trim_messages_demo()
    summarize_messages_demo()
