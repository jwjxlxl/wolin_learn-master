from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama

from utils import model_untils
from langchain.tools import tool, ToolRuntime


# =============================================================================
# 工具组件
# =============================================================================
#  
# 用途：教学演示 - 使用 LangChain Tool 组件扩展 AI 能力
#
# 核心概念：
#   - Tool = "AI 的工具箱"
#   - 让 AI 能够调用外部函数与世界交互
#   - ToolRuntime = 访问运行时上下文
# =============================================================================

# -----------------------------------------------------------------------------
# 运行前检查
# -----------------------------------------------------------------------------
# 1. 已安装 Ollama 服务
# 2. 已下载模型：ollama pull qwen3.5:2b
# -----------------------------------------------------------------------------


# =============================================================================
# 第一部分：理解 Tool 组件
# =============================================================================
"""
什么是 Tool？

🔧 定义
   Tool = "工具" = 让 AI 调用外部函数执行操作
   Tool 封装了一个可调用的函数及其输入架构

🎯 为什么需要 Tool？

   纯粹的语言模型只能生成文字：
   ❌ 无法访问实时信息（天气、股票、新闻）
   ❌ 无法执行操作（发邮件、查数据库、调用 API）
   ❌ 无法访问本地文件或系统

   有了 Tool，AI 就能：
   ✅ 搜索网络获取实时信息
   ✅ 查询数据库获取用户数据
   ✅ 调用各种 API 执行操作
   ✅ 读写文件、控制设备

💡 生活化比喻
   Tool = "万能工具箱"
   AI 是聪明的助手，但需要工具才能完成任务
   - 想查天气？需要天气预报工具
   - 想发邮件？需要邮件发送工具
   - 想算数学？需要计算器工具
"""


# =============================================================================
# 示例 1: 基础 Tool 定义
# =============================================================================

def basic_tool_definition():
    """
    使用 @tool 装饰器定义基础工具

    最简单的工具创建方式
    """
    print("=" * 60)
    print("示例 1: 基础 Tool 定义")
    print("=" * 60)

    from langchain_core.tools import tool

    # 使用 @tool 装饰器创建工具
    # 默认情况下，函数的文档字符串会成为工具的描述
    @tool
    def search_database(query: str, limit: int = 10) -> str:
        """Search the customer database for records matching the query.

        Args:
            query: Search terms to look for
            limit: Maximum number of results to return

        Returns:
            A string containing the results
        """
        return f"Found {limit} results for '{query}'"

    # 查看工具信息
    print("【工具信息】")
    print(f"  名称: {search_database.name}")
    print(f"  描述: {search_database.description}")
    print(f"  参数: {search_database.args}")
    print()

    # 调用工具
    result = search_database.invoke({"query": "张三", "limit": 5})
    print(f"【调用结果】{result}")
    print()

    # 不带参数调用（使用默认值）
    result2 = search_database.invoke({"query": "李四"})
    print(f"【调用结果2】{result2}")
    print()



# =============================================================================
# 示例 2: 自定义工具属性
# =============================================================================

def custom_tool_properties():
    """
    自定义工具的名称和描述

    让工具名称和描述更清晰
    """

    from langchain_core.tools import tool

    # 自定义工具名称
    @tool("web_search")
    def search(query: str) -> str:
        """Search the web for information."""
        return f"Web search results for: {query}"

    print("【自定义名称】")
    print(f"  工具名称: {search.name}")
    print(f"  原函数名被覆盖")
    print()

    # 自定义工具描述
    @tool("calculator", description="Performs arithmetic calculations. Use this for any math problems.")
    def calc(runtime: ToolRuntime) -> str:
        """Evaluate mathematical expressions."""
        print(f"runtime: {runtime.state.get("messages")}")
        print(f"runtime: {runtime.state.get("user_id")}")
        # 注意：eval 在实际使用中有安全风险，这里仅用于演示
        try:
            # result = eval(expression)
            return str(999)
        except Exception as e:
            return f"Error: {e}"

    print("【自定义描述】")
    print(f"  工具名称: {calc.name}")
    print(f"  工具描述: {calc.description}")
    print()

    # 测试计算器工具
    # test_expr = "2 + 3 * 4"
    # result = calc.invoke({"expression": test_expr})
    # print(f"【计算结果】{test_expr} = {result}")
    # print()

    model = ChatOllama(model="qwen3.5:2b")
    # bind_tools 返回一个绑定工具的新模型实例，原 model 不受影响
    model_with_tools = model.bind_tools([search, calc])


    # -------------------------------------------------------------------------
    # 场景 1: 触发搜索工具调用
    # -------------------------------------------------------------------------
    print("场景 1: 触发 web_search 工具")
    # messages = [HumanMessage(content="帮我搜索一下人工智能的最新进展")]
    messages = [
        AIMessage(content="你好，你是一个AI助手，不需要web搜索的功能可以不调用工具"),
        HumanMessage(content="简单介绍一下你自己")
    ]

    # 第 1 步: 模型决定是否调用工具
    # response = model_with_tools.invoke(messages)
    # print(f"  用户: {messages[0].content}")
    #
    # if response.tool_calls:
    #     for tc in response.tool_calls:
    #         print(f"  模型决定调用工具: {tc['name']}, 参数: {tc['args']}")
    #
    #         # 第 2 步: 执行工具
    #         tool_result = search.invoke(tc['args'])
    #         print(f"  工具返回: {tool_result}")
    #
    #         # 第 3 步: 将工具结果追加到消息列表
    #         messages.append(response)  # 追加模型的 AI 消息（含 tool_call）
    #         from langchain_core.messages import ToolMessage
    #         messages.append(ToolMessage(content=tool_result, tool_call_id=tc["id"]))
    #
    #         # 第 4 步: 让模型基于工具结果生成最终回答
    #         final_response = model_with_tools.invoke(messages)
    #         print(f"  最终回答: {final_response.content}")
    # else:
    #     print(f"  直接回复，不需要调用工具: {response.content}")
    # print()

    #-------------------------------------------------------------------------
    # 场景 2: 触发计算器工具调用
    #-------------------------------------------------------------------------
    print("场景 2: 触发 calculator 工具")
    messages = [HumanMessage(content="23 乘以 456 等于多少？")]

    response = model_with_tools.invoke(messages)
    print(f"  用户: {messages[0].content}")

    if response.tool_calls:
        for tc in response.tool_calls:
            print(f"  模型决定调用工具: {tc['name']}, 参数: {tc['args']}")

            tool_result = calc.invoke(tc['args'])
            print(f"  工具返回: {tool_result}")

            messages.append(response)
            from langchain_core.messages import ToolMessage
            messages.append(ToolMessage(content=tool_result, tool_call_id=tc["id"]))

            final_response = model_with_tools.invoke(messages)
            print(f"  最终回答: {final_response.content}")
    else:
        print(f"  Agent: {response.content}")
    print()


# =============================================================================
# 示例 3: 使用 Pydantic 定义复杂参数   做为了解
# =============================================================================

def pydantic_schema_tools():
    """
    使用 Pydantic 模型定义复杂的工具参数

    支持复杂的数据结构和参数验证
    """

    from pydantic import BaseModel, Field
    from typing import Literal, Optional
    from langchain_core.tools import tool

    # 定义输入 schema
    class WeatherInput(BaseModel):
        """Input for weather queries."""
        location: str = Field(description="City name or coordinates")
        units: Literal["celsius", "fahrenheit"] = Field(
            default="celsius",
            description="Temperature unit preference"
        )
        include_forecast: bool = Field(
            default=False,
            description="Include 5-day forecast"
        )

    # 使用 Pydantic schema 创建工具
    @tool(args_schema=WeatherInput)
    def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
        """Get current weather and optional forecast."""
        # 模拟天气数据
        temp = 22 if units == "celsius" else 72
        unit_symbol = "°C" if units == "celsius" else "°F"
        result = f"Current weather in {location}: {temp}{unit_symbol}"
        if include_forecast:
            result += "\nNext 5 days: Sunny, Cloudy, Rainy, Sunny, Sunny"
        return result

    print("【Pydantic Schema 工具】")
    print(f"  工具名称: {get_weather.name}")
    print(f"  描述: {get_weather.description}")
    print(f"  参数结构: {get_weather.args}")
    print()

    # 调用工具 - 完整参数
    result1 = get_weather.invoke({
        "location": "北京",
        "units": "celsius",
        "include_forecast": True
    })
    print(f"【完整参数调用】{result1}")
    print()

    # 调用工具 - 部分参数（使用默认值）
    result2 = get_weather.invoke({"location": "上海"})
    print(f"【部分参数调用】{result2}")
    print()



# =============================================================================
# 示例 4: ToolRuntime 访问状态
# =============================================================================

def tool_runtime_state():
    """
    使用 ToolRuntime 访问运行时状态

    state：messages，context上下文
    把运行时的信息注入到runtime中

    作用是：
        在工具中，可以访问和修改运行时的状态

    ToolRuntime 可以在工具执行时访问和修改状态
    """
    print("=" * 60)
    print("示例 5: ToolRuntime 访问状态")
    print("=" * 60)

    from langchain_core.messages import HumanMessage, AIMessage


    # 访问对话状态
    @tool
    def conversation_stats(runtime: ToolRuntime) -> str:
        """Summarize the conversation so far."""
        messages = runtime.state.get("messages", [])

        human_msgs = sum(1 for m in messages if m.__class__.__name__ == "HumanMessage")
        ai_msgs = sum(1 for m in messages if m.__class__.__name__ == "AIMessage")

        return f"对话统计: {human_msgs} 条用户消息, {ai_msgs} 条AI回复"

    # 访问自定义状态
    @tool
    def get_user_preference(pref_name: str, runtime: ToolRuntime) -> str:
        """Get a user preference value."""
        # runtime 参数对模型是隐藏的，不会暴露给 LLM
        preferences = runtime.state.get("user_preferences", {})
        return preferences.get(pref_name, "Not set")

    print("【ToolRuntime 工具】")
    print(f"  conversation_stats 名称: {conversation_stats.name}")
    print(f"  get_user_preference 名称: {get_user_preference.name}")
    print()
    print("  说明: runtime 参数对模型是隐藏的")
    print()



# =============================================================================
# 示例 5: 模拟工具调用流程
# =============================================================================

def tool_call_simulation():
    """
    模拟 AI 调用工具的完整流程

    展示模型如何决定调用哪个工具
    """
    print("=" * 60)
    print("示例 11: 模拟工具调用流程")
    print("=" * 60)

    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    from langchain_core.tools import tool
    from langchain_ollama import ChatOllama

    # 创建工具
    @tool
    def search_news(topic: str) -> str:
        """Search for news about a topic.
        
        Args:
            topic: The topic to search news for
        """
        news = {
            "AI": "最新AI技术突破：GPT-5发布",
            "科技": "苹果发布新一代iPhone",
            "体育": "世界杯即将开始",
        }
        return news.get(topic, f"没有找到关于'{topic}'的新闻")

    @tool
    def translate(text: str, target_lang: str = "English") -> str:
        """Translate text to another language.
        
        Args:
            text: The text to translate
            target_lang: The target language (default: English)
        """
        translations = {
            "你好": "Hello",
            "谢谢": "Thank you",
            "再见": "Goodbye",
        }
        return translations.get(text, f"[Translated to {target_lang}]: {text}")

    # 创建模型并绑定工具
    model = model_untils.get_qwen_client()
    model_with_tools = model.bind_tools([search_news, translate])

    # 创建提示词
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个新闻助手，可以搜索新闻和翻译文本。"),
        MessagesPlaceholder("messages"),
    ])

    print("【工具调用流程模拟】")
    print()
    print("步骤 1: 用户提问")
    messages = [HumanMessage(content="帮我搜索一下AI相关的新闻")]
    print(f"  用户: {messages[0].content}")
    print()

    print("步骤 2: 模型决定是否调用工具")
    # 第一次调用
    ai_response = model_with_tools.invoke(messages)
    print(f"  模型响应类型: {type(ai_response).__name__}")
    print(f"  是否有工具调用: {hasattr(ai_response, 'tool_calls') and len(ai_response.tool_calls) > 0}")
    
    if hasattr(ai_response, 'tool_calls') and ai_response.tool_calls:
        for tool_call in ai_response.tool_calls:
            print(f"  工具调用: {tool_call}")
    print()

    print("步骤 3: 执行工具并返回结果")
    # 执行工具
    tool_results = []
    if hasattr(ai_response, 'tool_calls') and ai_response.tool_calls:
        for tool_call in ai_response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            print(f"  执行工具: {tool_name}")
            print(f"  参数: {tool_args}")
            
            if tool_name == "search_news":
                result = search_news.invoke(tool_args)
            elif tool_name == "translate":
                result = translate.invoke(tool_args)
            else:
                result = "Unknown tool"
            
            print(f"  结果: {result}")
            tool_results.append(result)
    print()


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':

    # 运行示例
    # basic_tool_definition()
    # custom_tool_properties()
    # pydantic_schema_tools()
    tool_runtime_state()
    # tool_call_simulation()

