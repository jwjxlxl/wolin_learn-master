# =============================================================================
# 对话记忆
# =============================================================================
#  
# 用途：教学演示 - 使用 Memory 让 AI 记住历史对话
#
# 核心概念：
#   - 为什么 LLM 记不住之前的对话？（无状态）
#   - Memory = "外部记事本"
#   - Memory 帮它记住历史
# =============================================================================

# -----------------------------------------------------------------------------
# 运行前检查
# -----------------------------------------------------------------------------
# 1. 已安装 Ollama 服务
# 2. 已下载模型：ollama pull qwen3:4b
# -----------------------------------------------------------------------------

# 设置 UTF-8 编码（Windows 专用）
import sys
import io
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer,
    encoding='utf-8',
    errors='replace',
    line_buffering=True
)


# =============================================================================
# 第一部分：理解 Memory
# =============================================================================
"""
为什么需要 Memory？

🤔 问题
   LLM 本身是"无状态"的 - 每次调用都是全新的对话

   第一次：
   你："我叫小明"
   AI："你好小明！"

   第二次（紧接着）：
   你："我叫什么名字？"
   AI："我不知道您叫什么。" ← 失忆了！

💡 解决方案
   手动保存历史对话，每次调用时把历史记录一起传给 AI

   Memory = "外部记事本"
   帮 AI 记住之前说过的话

📊 Memory 的工作原理
   用户：我叫小明
   → Memory 保存：[{"role": "user", "content": "我叫小明"}]

   用户：我叫什么？
   → Memory 提供历史：[
       {"role": "user", "content": "我叫小明"},
       {"role": "assistant", "content": "你好小明！"},
       {"role": "user", "content": "我叫什么？"}
     ]
   → AI 根据历史回答："你叫小明"
"""


# =============================================================================
# 示例 1: 没有 Memory 的情况
# =============================================================================

def without_memory():
    """
    演示没有 Memory 时 AI 会"失忆"
    """
    print("=" * 60)
    print("示例 1: 没有 Memory（失忆的 AI）")
    print("=" * 60)

    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage

    model = ChatOllama(model="qwen3:4b")

    # 第一轮
    print("第一轮：")
    response = model.invoke([HumanMessage(content="我叫小明，请记住这个名字。")])
    print(f"AI: {response.content}")
    print()

    # 第二轮（新的调用，AI 不记得之前的事）
    print("第二轮：")
    response = model.invoke([HumanMessage(content="我叫什么名字？")])
    print(f"AI: {response.content}")
    print("↑ 看到没？AI 不记得了！")
    print()


# =============================================================================
# 示例 2: 手动实现记忆
# =============================================================================

def manual_memory():
    """
    手动保存和传递历史对话

    这是最基础的记忆实现方式
    """
    print("=" * 60)
    print("示例 2: 手动实现记忆")
    print("=" * 60)

    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage, AIMessage

    model = ChatOllama(model="qwen3:4b")

    # 用一个列表保存历史对话
    history = []

    # 第一轮
    print("第一轮：")
    user_input = "我叫小明，请记住这个名字。"
    print(f"你：{user_input}")

    # 构建完整消息历史
    messages = [HumanMessage(content=user_input)]
    response = model.invoke(messages)

    # 保存到历史
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response.content))

    print(f"AI: {response.content}")
    print()

    # 第二轮
    print("第二轮：")
    user_input = "我叫什么名字？"
    print(f"你：{user_input}")

    # 构建完整消息历史（包含之前的对话）
    messages = history + [HumanMessage(content=user_input)]
    response = model.invoke(messages)

    # 保存到历史
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response.content))

    print(f"AI: {response.content}")
    print("↑ 这次 AI 记住了！")
    print()


# =============================================================================
# 示例 3: 使用 ConversationBufferMemory
# =============================================================================

def using_buffer_memory():
    """
    使用 LangChain 的 ConversationBufferMemory

    更优雅的记忆管理方式
    """
    print("=" * 60)
    print("示例 3: ConversationBufferMemory")
    print("=" * 60)

    from langchain_ollama import ChatOllama
    from langchain_classic.memory import ConversationBufferMemory
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    # 1. 创建 Memory
    # conversation_prefix 用于存储历史对话
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # 变量名
        return_messages=True        # 返回消息列表
    )

    # 2. 创建 Prompt（包含历史对话的位置）
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位友好的助手。"),
        MessagesPlaceholder(variable_name="chat_history"),  # 历史对话放这里
        ("human", "{input}"),
    ])

    # 3. 创建链
    model = ChatOllama(model="qwen3:4b")
    chain = prompt | model

    # 4. 对话循环
    print("开始对话（输入 'quit' 退出）\n")

    while True:
        user_input = input("你：").strip()
        if user_input.lower() == 'quit':
            break

        # 从 memory 获取历史对话
        chat_history = memory.load_memory_variables({})["chat_history"]

        # 构建消息
        messages = prompt.format_messages(
            chat_history=chat_history,
            input=user_input
        )

        # 调用模型
        response = chain.invoke({"input": user_input, "chat_history": chat_history})

        print(f"AI: {response.content}")
        print()

        # 保存对话到 memory
        memory.save_context(
            {"input": user_input},
            {"output": response.content}
        )


# =============================================================================
# 示例 4: 实用的对话助手（带记忆）
# =============================================================================

def practical_chat_assistant():
    """
    实用的带记忆对话助手

    简化版本，适合日常使用
    """
    print("=" * 60)
    print("示例 4: 实用对话助手（带记忆）")
    print("=" * 60)
    print("（输入 'quit' 退出，输入 'clear' 清空记忆）\n")

    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

    model = ChatOllama(model="qwen3:4b")

    # 用列表保存历史
    messages = [
        SystemMessage(content="你是一位友好、有帮助的助手。")
    ]

    while True:
        user_input = input("你：").strip()
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'clear':
            messages = messages[:1]  # 保留 system 消息
            print("记忆已清空\n")
            continue

        # 添加用户消息
        messages.append(HumanMessage(content=user_input))

        # 调用模型
        response = model.invoke(messages)

        print(f"AI: {response.content}")
        print()

        # 添加 AI 消息到历史
        messages.append(AIMessage(content=response.content))


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  对话记忆 - Conversation Memory")
    print("  说明：让 AI 记住历史对话")
    print("=" * 70 + "\n")

    print("【运行前检查】")
    print("  1. 已安装 Ollama 服务")
    print("  2. 已下载模型：ollama pull qwen3:4b")
    print()

    # 演示失忆问题
    without_memory()

    # 演示手动记忆
    manual_memory()

    # 交互式示例（取消注释运行）
    # using_buffer_memory()
    # practical_chat_assistant()

    print("=" * 70)
    print("  接下来学习：buffer_memory.py（更多记忆类型）")
    print("=" * 70 + "\n")
