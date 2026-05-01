# =============================================================================
# Chat Model 对话
# =============================================================================
#  
# 用途：教学演示 - 理解 Chat Model 的消息类型
#
# 核心概念：
#   - Message 类型：SystemMessage, HumanMessage, AIMessage
#   - 为什么 Chat Model 更适合对话？
# =============================================================================

# -----------------------------------------------------------------------------
# 运行前检查
# -----------------------------------------------------------------------------
# 1. 已安装 Ollama 服务
# 2. 已下载模型：ollama pull qwen3.5:2b
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

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# =============================================================================
# 第一部分：理解三种消息类型
# =============================================================================
"""
三种消息类型，就像戏剧中的三个角色：

🎭 SystemMessage（系统提示词）
   角色：导演/编剧
   作用：设定 AI 的人设和行为准则
   例子："你是一位友好的助手"、"用小学生能懂的语言解释"

👤 HumanMessage（用户消息）
   角色：观众/提问者
   作用：用户的输入和问题
   例子："你好"、"什么是人工智能"

🤖 AIMessage（AI 消息）
   角色：演员/回答者
   作用：AI 的回复（用于构建多轮对话历史）
   例子："你好！很高兴见到你"
"""


# =============================================================================
# 示例 1: SystemMessage - 设定 AI 人设
# =============================================================================

def system_message_example():
    """
    SystemMessage 的作用：设定 AI 的人设和行为

    就像给演员剧本，告诉 TA 应该扮演什么角色
    """
    print("=" * 60)
    print("示例 1: SystemMessage - 设定 AI 人设")
    print("=" * 60)

    model = ChatOllama(model="qwen3.5:2b")

    # 不同的系统提示词，得到不同的回复风格
    print("【人设 1】严肃的科学家：")
    messages = [
        SystemMessage(content="你是一位严肃的科学家，说话严谨、准确。"),
        HumanMessage(content="你好，请介绍一下你自己。"),
    ]
    response = model.invoke(messages)
    print(f"AI: {response.content}")
    print()

    print("【人设 2】热情的朋友：")
    messages = [
        SystemMessage(content="你是一位热情的朋友，说话活泼、有趣。"),
        HumanMessage(content="你好，请介绍一下你自己。"),
    ]
    response = model.invoke(messages)
    print(f"AI: {response.content}")
    print()

    print("【人设 3】贴吧老哥：")
    messages = [
        SystemMessage(content="你是一个贴吧老哥，喜欢阴阳怪气，说话带梗。"),
        HumanMessage(content="你好，请介绍一下你自己。"),
    ]
    response = model.invoke(messages)
    print(f"AI: {response.content}")
    print()


# =============================================================================
# 示例 2: HumanMessage - 用户提问
# =============================================================================

def human_message_example():
    """
    HumanMessage 是最常用的消息类型

    就是用户的输入内容
    """
    print("=" * 60)
    print("示例 2: HumanMessage - 用户提问")
    print("=" * 60)

    model = ChatOllama(model="qwen3.5:2b")

    # 简单用法
    messages = [
        HumanMessage(content="Python 中如何反转字符串？")
    ]
    response = model.invoke(messages)
    print(f"AI: {response.content}")
    print()


# =============================================================================
# 示例 3: AIMessage - 构建多轮对话
# =============================================================================

def multi_turn_conversation():
    """
    使用 AIMessage 构建多轮对话历史

    AI 本身没有记忆，需要手动把历史对话传给 TA
    """
    print("=" * 60)
    print("示例 3: 多轮对话（使用 AIMessage）")
    print("=" * 60)

    model = ChatOllama(model="qwen3.5:2b")

    # 第一轮对话
    print("第一轮：")
    messages = [
        SystemMessage(content="你是一位友好的助手。"),
        HumanMessage(content="我叫小明，今年 10 岁，喜欢打篮球。"),
    ]
    response = model.invoke(messages)
    print(f"AI: {response.content}")

    # 把 AI 的回复添加到历史记录
    messages.append(AIMessage(content=response.content))

    # 第二轮对话
    print("\n第二轮：")
    messages.append(HumanMessage(content="你喜欢什么运动？"))
    response = model.invoke(messages)
    print(f"AI: {response.content}")

    # 把 AI 的回复添加到历史记录
    messages.append(AIMessage(content=response.content))

    # 第三轮对话 - 测试 AI 是否记住之前的信息
    print("\n第三轮：")
    messages.append(HumanMessage(content="我是谁？我几岁？我喜欢什么？"))
    response = model.invoke(messages)
    print(f"AI: {response.content}")
    print()


# =============================================================================
# 示例 4: 消息类型的简化写法
# =============================================================================

def simplified_message_example():
    """
    LangChain 支持简化的消息写法

    可以直接用元组 (role, content) 表示消息
    """
    print("=" * 60)
    print("示例 4: 简化写法（元组形式）")
    print("=" * 60)

    model = ChatOllama(model="qwen3.5:2b")

    # 简化写法：使用元组 ("角色", "内容")
    # "system" = SystemMessage
    # "human"  = HumanMessage
    # "ai"     = AIMessage
    messages = [
        ("system", "你是一位友好的助手。"),
        ("human", "你好！"),
        ("ai", "你好！很高兴见到你，有什么可以帮你的吗？"),
        ("human", "我想学习 Python，应该怎么开始？"),
    ]

    response = model.invoke(messages)
    print(f"AI: {response.content}")
    print()

    # 也可以用列表推导式批量转换
    # messages = [(role, content) for role, content in [...]]


# =============================================================================
# 示例 5: 实用场景 - 角色扮演对话
# =============================================================================

def role_play_conversation():
    """
    实际应用场景：角色扮演对话

    通过 SystemMessage 设定特定角色，让 AI 扮演
    """
    print("=" * 60)
    print("示例 5: 角色扮演 - 英语老师")
    print("=" * 60)

    model = ChatOllama(model="qwen3.5:2b")

    # 设定 AI 为英语老师
    messages = [
        SystemMessage(content="""你是一位英语老师，请用英语和学生对话。
        要求：
        1. 用简单易懂的英语
        2. 适当解释生词
        3. 鼓励学生多说
        """),
        HumanMessage(content="老师好，我想练习英语口语。"),
    ]

    response = model.invoke(messages)
    print(f"AI: {response.content}")
    print()


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  Chat Model 消息类型")
    print("  说明：SystemMessage, HumanMessage, AIMessage")
    print("=" * 70 + "\n")


    # 运行示例
    # system_message_example()
    # human_message_example()
    multi_turn_conversation()
    # simplified_message_example()
    # role_play_conversation()

    print("=" * 70)
    print("  接下来学习：streaming_output.py（流式输出）")
    print("=" * 70 + "\n")
