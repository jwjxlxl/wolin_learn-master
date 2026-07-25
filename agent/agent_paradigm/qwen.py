import os
from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ALIYUN_API_KEY"),
    base_url=os.getenv(
        "DASH_SCOPE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
)


# ──────────────────── 基础 LLM 调用 ────────────────────

def call_llm(messages: list[dict]) -> str:
    """发送消息列表给 LLM，返回助手的回复文本。"""
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
        # extra_body={"enable_search": True},
    )
    return completion.choices[0].message.content

if __name__ == "__main__":
    print(call_llm([{"role": "user", "content": "你好"}]))