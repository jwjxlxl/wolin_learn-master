# =============================================================================
# Chat Model 消息类型 — SystemMessage / HumanMessage / AIMessage
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解三种消息类型的作用和区别
#   ✅ 用 SystemMessage 设定 AI 的角色人设
#   ✅ 手动构建多轮对话历史，让 AI "记住"上下文
# =============================================================================

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


"""
三种消息类型 — 像戏剧中的三个角色：

  SystemMessage（导演）— 设定 AI 的人设和行为准则
    例子: "你是一位友好的助手"、"用小学生的语言解释"

  HumanMessage（观众）— 用户的问题和输入
    例子: "你好"、"什么是人工智能"

  AIMessage（演员）— AI 的回复，手动添加到历史中用于多轮对话
    例子: "你好！很高兴见到你"
"""


# =============================================================================
# 示例 1: SystemMessage — 同一问题，不同人设，不同回答
# =============================================================================

def system_message_demo():
    """
    演示 SystemMessage 如何改变 AI 的回答风格。

    同一个问题 "请介绍一下你自己"，配上不同的 SystemMessage，
    AI 会呈现出完全不同的性格——这就像给演员不同的剧本。
    """
    print(f"\n-- 示例 1: SystemMessage — 设定 AI 人设")

    model = ChatOllama(model="qwen3.5:2b")

    personas = [
        ("严肃的科学家", "你是一位严肃的科学家，说话严谨、准确。"),
        ("热情的朋友",   "你是一位热情的朋友，说话活泼、有趣。"),
    ]

    for role, system_prompt in personas:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="请介绍一下你自己。"),
        ]
        response = model.invoke(messages)
        print(f"  [{role}]: {response.content}\n")


# =============================================================================
# 示例 2: 多轮对话 — 手动把 AI 的回复加回历史
# =============================================================================

def multiturn_demo():
    """
    演示如何手动实现多轮对话记忆。

    关键步骤：
    1. 把用户消息加到列表 → messages.append(HumanMessage(...))
    2. 调用模型拿到回复
    3. 把 AI 回复也加到列表 → messages.append(AIMessage(...))
    4. 下次调用时传完整列表

    这就是 Memory 模块的底层原理——后面 05_memory 会教更优雅的方式。
    """
    print(f"\n-- 示例 2: 多轮对话 — 让 AI 记住上下文")

    model = ChatOllama(model="qwen3.5:2b")
    messages = [SystemMessage(content="你是一位友好的助手。")]

    # 第 1 轮
    messages.append(HumanMessage(content="我叫小明，今年 10 岁，喜欢打篮球。"))
    response = model.invoke(messages)
    print(f"[第1轮] {response.content}")

    messages.append(AIMessage(content=response.content))

    # 第 2 轮
    messages.append(HumanMessage(content="我是谁？我几岁？我喜欢什么？"))
    response = model.invoke(messages)
    print(f"[第2轮] {response.content}")


# =============================================================================
# 示例 3: 角色扮演 — 实用场景
# =============================================================================

def roleplay_demo():
    """
    通过 SystemMessage 让 AI 扮演特定角色——这是日常最实用的场景。

    比如：英语老师、代码审查员、产品经理面试官...
    写好 SystemMessage = 设定好角色，AI 就会按角色行事。
    """
    print(f"\n-- 示例 3: 角色扮演 — 英语老师")

    model = ChatOllama(model="qwen3.5:2b")

    messages = [
        SystemMessage(content="""你是一位英语老师。
要求：
1. 用简单易懂的英语和学生对话
2. 适当解释生词
3. 鼓励学生多说"""),
        HumanMessage(content="老师好，我想练习英语口语。"),
    ]

    response = model.invoke(messages)
    print(f"英语老师: {response.content}")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n>>> 02_llm_call/chat_model — 消息类型详解\n")

    system_message_demo()
    multiturn_demo()
    roleplay_demo()

    # 接下来学习: streaming_output.py（流式输出）
