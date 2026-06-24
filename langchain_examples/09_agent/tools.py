# =============================================================================
# Agent 工具（Tool）— 让 AI 有了"手和脚"
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 使用 @tool 装饰器定义 AI 可调用的工具函数
#   ✅ 用 Pydantic 定义复杂工具参数
#   ✅ 使用 ToolRuntime 在工具中访问运行状态
# =============================================================================

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)


"""
什么是 Tool？

  纯 LLM 只能"说"——生成文字
  有了 Tool 就能"做"——调用外部函数（查天气、算数学、搜数据库...）

  Tool = 给 AI 配的工具箱
    想查天气？→ get_weather 工具
    想发邮件？→ send_email 工具
    想算数学？→ calculator 工具

  生活化比喻: Tool = 万能工具箱
    AI 是聪明的助手，但没有手——Tool 就是它的手
"""


# =============================================================================
# 示例 1: @tool 装饰器基础
# =============================================================================

def basic_tool():
    """用 @tool 装饰器定义最简单的工具函数。"""
    from langchain_core.tools import tool

    print(f"\n-- 示例 1: @tool 装饰器基础")

    @tool(name_or_callable="my_weather_tool")
    def get_weather(city: str) -> str:
        """查询指定城市的天气信息。

        Args:
            city: 城市名称，如"北京"、"上海"
        """
        weather_db = {"北京": "晴，25°C", "上海": "多云，28°C", "广州": "小雨，30°C"}
        return weather_db.get(city, f"暂无 {city} 的天气数据")

    # 查看工具元信息
    print(f"  工具名: {get_weather.name}")
    print(f"  描述:   {get_weather.description}")
    print(f"  参数:   {get_weather.args}")

    # 直接调用
    # get_weather可以调用invoke方法，chain可以调用invoke方法, model.invoke
    result = get_weather.invoke({"city": "北京"})
    print(f"  调用结果: {result}")


# =============================================================================
# 示例 2: Pydantic Schema — 复杂参数 + 参数验证
# =============================================================================

def pydantic_schema_tool():
    """用 Pydantic 模型定义复杂工具参数，自带类型验证。"""
    from pydantic import BaseModel, Field
    from typing import Literal
    from langchain_core.tools import tool

    print(f"\n-- 示例 2: Pydantic Schema 工具")

    class WeatherInput(BaseModel):
        """天气查询参数。"""
        city: str = Field(description="城市名称")
        units: Literal["celsius", "fahrenheit"] = Field(default="celsius", description="温度单位")

    @tool(args_schema=WeatherInput)
    def get_weather(city: str, units: str = "celsius") -> str:
        """查询指定城市的天气。"""
        temp = 22 if units == "celsius" else 72
        unit = "°C" if units == "celsius" else "°F"
        return f"{city} 当前温度: {temp}{unit}"

    print(f"  工具名: {get_weather.name}")
    print(f"  Schema: {get_weather.args}")
    print(f"  调用(celsius): {get_weather.invoke({'city': '北京', 'units': 'celsius'})}")
    print(f"  调用(fahrenheit): {get_weather.invoke({'city': '北京', 'units': 'fahrenheit'})}")


# =============================================================================
# 示例 3: ToolRuntime — 在工具中访问运行状态
# =============================================================================

def tool_runtime_demo():
    """ToolRuntime 让工具能读写 Agent 的运行状态。

    ToolRuntime 只能在 LangGraph 的 create_react_agent 中使用——
    Agent 运行工具时，框架自动注入 runtime（含 state、config 等），
    这个参数对 LLM 是隐藏的，不会出现在工具描述中。
    """
    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage
    from langgraph.prebuilt import ToolRuntime, create_react_agent
    from langchain_ollama import ChatOllama
    from langchain.agents import create_agent

    print(f"\n-- 示例 3: ToolRuntime — 在 Agent 中访问运行状态")

    # 定义带 ToolRuntime 的工具
    @tool
    def conversation_stats(runtime: ToolRuntime) -> str:
        """统计当前对话的消息数。

        runtime 参数对 LLM 是隐藏的——它不会出现在工具描述中，
        但运行时框架会自动注入，包含 state、config 等。"""
        messages = runtime.state.get("messages", [])
        human_count = sum(1 for m in messages if m.__class__.__name__ == "HumanMessage")
        ai_count = sum(1 for m in messages if m.__class__.__name__ == "AIMessage")
        return f"对话统计: {human_count} 条用户消息, {ai_count} 条 AI 回复"

    # 先看工具元信息——runtime 参数对 LLM 隐藏
    print(f"  工具名: {conversation_stats.name}")
    print(f"  描述: {conversation_stats.description}")
    print(f"  参数: {conversation_stats.args}  ← 注意: 没有 runtime 参数")
    print()

    # 将工具放入 Agent 运行——此时 runtime 才会被自动注入
    model = ChatOllama(model="qwen3.5:2b", format="json")
    agent = create_agent(model=model, tools=[conversation_stats])

    result = agent.invoke({"messages": [HumanMessage(content="请统计一下当前对话的消息数")]})
    print(f"  Agent 回复: {result['messages'][-1].content}")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n>>> 09_agent/tools — Agent 工具定义\n")

    # basic_tool()
    pydantic_schema_tool()
    # tool_runtime_demo()

    # 接下来学习: agent.py（create_agent 创建智能体）
