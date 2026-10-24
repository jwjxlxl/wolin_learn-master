from utils.model_utils import get_model

model = get_model("qwen")


"""
千问大模型客户端示例

使用 OpenAI 兼容接口实例化千问客户端，并发起提问。
参考：https://help.aliyun.com/zh/model-studio/getting-started/first-api-call-to-qwen
"""

import os
from dotenv import load_dotenv
from openai import OpenAI


def create_qwen_client() -> OpenAI:
    """创建千问大模型客户端"""
    load_dotenv()

    api_key = os.getenv("ALIYUN_API_KEY")
    if not api_key or api_key == "your_aliyun_api_key_here":
        print("错误：未配置有效的 ALIYUN_API_KEY")
        print("请在 .env 文件中配置你的阿里云 DashScope API Key")
        print("获取地址：https://bailian.console.aliyun.com/")
        return None

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    return client


def chat_simple(client: OpenAI, user_message: str):
    """单轮对话"""
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "你是一个有帮助的AI助手。"},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content


def chats(client: OpenAI):
    messages = [{
        "role": "system",
        "content": "你是一个有帮助的AI助手。"
    }]
    q1 = {"role":"user", "content": "我是Luke，很高兴认识你!"}
    messages.append(q1)
    r1 = client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
    )
    print("r1--------", r1.choices[0].message.content)

    q2 = [{"role":"user", "content": "你知道我的名字吗？"}]
    messages.extend(q2)
    messages.append({"role":"assistant", "content": r1.choices[0].message.content})
    r2 = client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
    )
    print("r2---------", r2.choices[0].message.content)


def chat_multi_turn(client: OpenAI):
    """多轮对话示例"""
    messages = [
        {"role": "system", "content": "你是一个精通中国历史的AI助手，回答简洁准确。"},
    ]

    questions = [
        "三国演义中桃园三结义是哪三个人？",
        "其中谁的武力最强？",
        "请用一个词总结他们的关系。",
    ]

    for q in questions:
        messages.append({"role": "user", "content": q})

        response = client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
        )

        answer = response.choices[0].message.content
        messages.append({"role": "assistant", "content": answer})

        print(f"问：{q}")
        print(f"答：{answer}\n")


if __name__ == "__main__":
    client = create_qwen_client()

    print(f"\n-- 示例 2: 多轮对话")
    chats(client)


# if __name__ == '__main__':
#     messages = []
#     user_message = "你好！我的名字是Luke，很高兴认识你！"
#     r1 = model.invoke(user_message)
#     print(r1)
#     messages.append(user_message)
#     messages.append(r1.content)
#
#     q2 = "50个字介绍一下机器学习"
#     r2 = model.invoke(q2)
#     messages.append(q2)
#     messages.append(r2.content)
#
#     q3 = "你知道我的名字吗？"
#     messages.append(q3)
#     r3 = model.invoke(messages)
#     print(r3.content)
