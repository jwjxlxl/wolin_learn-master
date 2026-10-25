# =============================================================================
# 记忆策略 — 完整记忆 vs 窗口记忆
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解不同记忆策略的适用场景
#   ✅ 使用 ConversationBufferMemory（完整记忆）和 WindowMemory（滑动窗口）
# =============================================================================

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)


from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.memory import ConversationBufferMemory, ConversationBufferWindowMemory

# =============================================================================
# 示例 1: 完整记忆 — 保存所有对话
# =============================================================================

def buffer_memory_demo():
    """
    ConversationBufferMemory 保存完整的聊天记录。

    save_context({"input": ..., "output": ...})  — 每次存一对问答
    load_memory_variables({})["chat_history"]     — 取出所有历史
    """
    print(f"\n-- 示例 1: 完整记忆（Buffer）")

    memory = ConversationBufferMemory(return_messages=True)

    memory.save_context(
        {"input": "你好，我叫小明。"},
        {"output": "你好小明！很高兴认识你。"},
    )
    memory.save_context(
        {"input": "我今年 10 岁。"},
        {"output": "10 岁正是学习的好年纪！"},
    )
    memory.save_context(
        {"input": "我喜欢打篮球。"},
        {"output": "篮球是一项很好的运动！"},
    )

    history = memory.load_memory_variables({})["history"]
    print(f"  历史消息数: {len(history)} 条")
    for msg in history:
        role = "用户" if isinstance(msg, HumanMessage) else "AI"
        print(f"    {role}: {msg.content}")


# =============================================================================
# 示例 2: 窗口记忆 — 只保留最近 K 轮
# =============================================================================

def window_memory_demo():
    """
    ConversationBufferWindowMemory 只保留最近 K 轮对话。

    k=3 表示只保留最近 3 轮。旧消息自动丢弃，节省 Token。
    """
    print(f"\n-- 示例 2: 窗口记忆（Window, k=3）")

    memory = ConversationBufferWindowMemory(k=3, return_messages=True)

    conversations = [
        ("第1轮：你好", "你好！有什么可以帮你？"),
        ("第2轮：今天天气不错", "是啊，适合出去玩！"),
        ("第3轮：我想去公园", "公园是个好选择！"),
        ("第4轮：有什么推荐的活动", "可以散步、野餐！"),
        ("第5轮：好的谢谢", "不客气，玩得开心！"),
    ]

    for user_msg, ai_msg in conversations:
        memory.save_context({"input": user_msg}, {"output": ai_msg})

    history = memory.load_memory_variables({}).get("history", [])
    print(f"  保存了 5 轮，但只保留 {len(history)} 条消息（最近 3 轮）")
    for msg in history:
        role = "用户" if isinstance(msg, HumanMessage) else "AI"
        print(f"    {role}: {msg.content}")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n>>> 05_memory/buffer_memory — 记忆策略\n")

    window_memory_demo()
