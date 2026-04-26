# =============================================================================
# 缓冲区记忆
# =============================================================================
#  
# 用途：教学演示 - 使用不同类型的 Memory 管理对话历史
#
# 核心概念：
#   - ConversationBufferMemory = "完整聊天记录"
#   - 限制长度的原因（成本、上下文限制）
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
# 第一部分：Memory 的类型
# =============================================================================
"""
LangChain 提供的 Memory 类型：

📋 ConversationBufferMemory
   - 保存完整的聊天记录
   - 适合：短对话、需要完整上下文的场景
   - 缺点：对话长了会消耗大量 token

✂️ ConversationBufferWindowMemory
   - 只保存最近的 K 轮对话
   - 适合：长对话、节省 token
   - 类似：滑动窗口

📝 ConversationSummaryMemory
   - 用 AI 总结历史对话
   - 适合：超长对话
   - 类似：把厚书读薄

💬 ConversationSummaryBufferMemory
   - 结合窗口和总结
   - 最近对话保留原文，更早的总结
   - 最佳平衡
"""


# =============================================================================
# 示例 1: ConversationBufferMemory 详解
# =============================================================================

def buffer_memory_details():
    """
    详细了解 ConversationBufferMemory
    """
    print("=" * 60)
    print("示例 1: ConversationBufferMemory 详解")
    print("=" * 60)

    from langchain_classic.memory import ConversationBufferMemory
    from langchain_core.messages import HumanMessage, AIMessage

    # 创建 Memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True  # 返回消息对象列表
    )

    # 保存对话
    print("保存对话...")
    memory.save_context(
        {"input": "你好，我叫小明。"},
        {"output": "你好小明！很高兴认识你。"}
    )
    memory.save_context(
        {"input": "我今年 10 岁。"},
        {"output": "10 岁正是学习的好年纪！"}
    )
    memory.save_context(
        {"input": "我喜欢打篮球。"},
        {"output": "篮球是一项很好的运动！"}
    )

    # 查看历史
    history = memory.load_memory_variables({})["chat_history"]
    print(f"\n历史对话数量：{len(history)}")
    print(f"历史对话类型：{type(history)}")

    for i, msg in enumerate(history, 1):
        print(f"  {i}. {type(msg).__name__}: {msg.content[:30]}...")

    print()

    # 获取对话字符串
    buffer = memory.load_memory_variables({})["chat_history"]
    print("完整对话字符串:")
    for msg in buffer:
        if isinstance(msg, HumanMessage):
            print(f"  用户：{msg.content}")
        else:
            print(f"  AI: {msg.content}")
    print()


# =============================================================================
# 示例 2: ConversationBufferWindowMemory（窗口记忆）
# =============================================================================

def window_memory_example():
    """
    使用窗口记忆，只保留最近的对话

    适合长对话场景，节省 token
    """
    print("=" * 60)
    print("示例 2: ConversationBufferWindowMemory（窗口记忆）")
    print("=" * 60)

    from langchain_classic.memory import ConversationBufferWindowMemory
    from langchain_core.messages import HumanMessage, AIMessage

    # k=2 表示只保留最近 2 轮对话
    memory = ConversationBufferWindowMemory(
        k=2,
        return_messages=True
    )

    print("保存 5 轮对话（但只保留最近 2 轮）...\n")

    # 保存多轮对话
    conversations = [
        ("第 1 轮：你好", "你好！有什么可以帮你？"),
        ("第 2 轮：今天天气不错", "是啊，适合出去玩！"),
        ("第 3 轮：我想去公园", "公园是个好选择！"),
        ("第 4 轮：有什么推荐的活动吗", "可以散步、野餐！"),
        ("第 5 轮：好的谢谢", "不客气，玩得开心！"),
    ]

    for user_input, ai_response in conversations:
        memory.save_context(
            {"input": user_input},
            {"output": ai_response}
        )
        print(f"保存：{user_input} → {ai_response}")

    # 查看历史（只应该有最近 2 轮）
    memory_vars = memory.load_memory_variables({})
    # 新版本可能使用不同的键名
    history = memory_vars.get("chat_history", memory_vars.get("history", []))
    print(f"\n实际保留的历史数量：{len(history)}")
    print("(因为 k=2，所以只保留最近 2 轮 = 4 条消息)")

    for msg in history:
        print(f"  {type(msg).__name__}: {msg.content}")
    print()


# =============================================================================
# 示例 3: 带记忆的对话链
# =============================================================================

def conversation_chain():
    """
    使用 ConversationChain 创建带记忆的对话

    ConversationChain 是 LangChain 封装好的对话链
    """
    print("=" * 60)
    print("示例 3: ConversationChain（封装好的对话链）")
    print("=" * 60)

    from langchain_ollama import ChatOllama
    from langchain_classic.memory import ConversationBufferMemory
    from langchain_classic.chains import ConversationChain
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    # 创建模型
    model = ChatOllama(model="qwen3:4b")

    # 创建记忆
    memory = ConversationBufferMemory(return_messages=True)

    # 创建对话链
    # ConversationChain = Prompt + Model + Memory 的封装
    chain = ConversationChain(
        llm=model,
        memory=memory,
        prompt=ChatPromptTemplate.from_messages([
            ("system", "你是一位友好、有帮助的助手。"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
    )

    print("开始对话（输入 'quit' 退出）\n")

    while True:
        user_input = input("你：").strip()
        if user_input.lower() == 'quit':
            break

        response = chain.invoke({"input": user_input})
        print(f"AI: {response['response']}")
        print()


# =============================================================================
# 示例 4: 实际应用场景 - 客服机器人
# =============================================================================

def customer_service_bot():
    """
    模拟客服机器人

    需要记住用户的问题和之前的对话
    """
    print("=" * 60)
    print("示例 4: 客服机器人（实际应用）")
    print("=" * 60)
    print("(输入 'quit' 退出)\n")

    from langchain_ollama import ChatOllama
    from langchain_classic.memory import ConversationBufferMemory
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    # 系统提示词
    system_prompt = """你是一位电商客服助手，负责回答用户关于订单、物流、售后等问题。
要求：
1. 态度友好、专业
2. 回答简洁明了
3. 如果不知道具体信息，引导用户提供订单号"""

    # 创建记忆
    memory = ConversationBufferMemory(return_messages=True)

    # 创建 Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    model = ChatOllama(model="qwen3:4b")
    chain = prompt | model

    print("客服机器人已启动！")
    print("可以问：我的订单到哪了？怎么退换货？etc.\n")

    while True:
        user_input = input("你：").strip()
        if user_input.lower() == 'quit':
            break

        # 获取历史
        chat_history = memory.load_memory_variables({})["chat_history"]

        # 调用
        messages = prompt.format_messages(
            chat_history=chat_history,
            input=user_input
        )
        response = chain.invoke(messages)

        print(f"客服：{response.content}")
        print()

        # 保存对话
        memory.save_context(
            {"input": user_input},
            {"output": response.content}
        )


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  缓冲区记忆 - Buffer Memory")
    print("  说明：不同类型的记忆管理")
    print("=" * 70 + "\n")

    print("【运行前检查】")
    print("  1. 已安装 Ollama 服务")
    print("  2. 已下载模型：ollama pull qwen3:4b")
    print()

    # 运行示例
    buffer_memory_details()
    window_memory_example()

    # 交互式示例（按需运行）
    # conversation_chain()
    # customer_service_bot()

    print("=" * 70)
    print("  接下来学习：06_chains/simple_chain.py")
    print("=" * 70 + "\n")
