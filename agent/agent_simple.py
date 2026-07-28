import os
import sys
import io
import json

'''
    不使用任何Agent框架，如何创建一个Agent
'''

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# 实例化LLM的客户端
client = OpenAI(
    api_key=os.getenv("ALIYUN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ── 工具实现 没有使用tool装饰器，就是一个普通的函数──────────────
def search_weather(city: str) -> str:
    """查询指定城市的天气"""
    return f"{city}明天晴，15-25°C"


def book_flight(origin: str, destination: str, date: str) -> str:
    """预订从 origin 到 destination 的机票，日期为 date"""
    return f"已订票：{origin}→{destination}，{date}"


# ── 工具定义（告诉 LLM 有哪些工具可用） ─────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_weather",
            "description": "查询指定城市未来的天气情况",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称，如 '北京', '上海'"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_flight",
            "description": "预订两个城市之间的航班",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "出发城市"},
                    "destination": {"type": "string", "description": "目的城市"},
                    "date": {"type": "string", "description": "出发日期，如 '明天', '2024-03-15'"},
                },
                "required": ["origin", "destination", "date"],
            },
        },
    },
]

# 工具名称 → 实际函数的映射
AVAILABLE_FUNCTIONS = {
    "search_weather": search_weather,
    "book_flight": book_flight,
}


# ── Agent 主循环（基于 function calling） ──────────────────
def agent_loop(user_input: str):
    system_prompt = (
        "你是一个差旅助手，帮助用户规划出差行程。\n"
        "你可以使用提供的工具查询天气和订票。\n"
        "在给出最终建议之前，请先调用工具获取真实信息。\n"
        "获取所有必要信息后，给出完整的行程安排。"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    # 之前不是Agent的写法：client.调用大模型回答：大模型的回答只有一次，也不会调用工具

    while True:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            tools=TOOLS,
        )

        choice = response.choices[0]
        message = choice.message

        # ── 情况 1：LLM 选择调用工具 ──────────────────────
        if message.tool_calls:
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                func = AVAILABLE_FUNCTIONS[func_name]

                print(f"[调用工具] {func_name}({json.dumps(func_args, ensure_ascii=False)})")
                result = func(**func_args)
                print(f"[工具返回] {result}")

                # 将工具调用和结果追加到上下文
                messages.append(message)  # assistant 的 tool_calls 消息
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                })

        # ── 情况 2：LLM 回复文本（不再需要工具） ──────────
        elif message.content:
            print(f"\n{message.content}")
            break


# ── 运行 ───────────────────────────────────────────────────
agent_loop("帮我安排明天去北京出差，我从上海出发")
