# =============================================================================
# 记忆策略 — 完整记忆 vs 窗口记忆
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解不同记忆策略的适用场景
#   ✅ 完整记忆 = 保留全部历史；窗口记忆 = trim_messages 只保留最近 K 条
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from utils.model_utils import setup_encoding
setup_encoding()


from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages.utils import trim_messages

# =============================================================================
# 示例 1: 完整记忆 — 保存所有对话
# =============================================================================

def buffer_memory_demo():
    """
    完整记忆 = 保留全部对话历史，不做任何修剪。

    直接用一个 list 保存所有消息，需要时全部传给模型。
    """
    print(f"\n-- 示例 1: 完整记忆（保留全部历史）")

    history = [
        HumanMessage(content="你好，我叫小明。"),
        AIMessage(content="你好小明！很高兴认识你。"),
        HumanMessage(content="我今年 10 岁。"),
        AIMessage(content="10 岁正是学习的好年纪！"),
        HumanMessage(content="我喜欢打篮球。"),
        AIMessage(content="篮球是一项很好的运动！"),
    ]

    print(f"  历史消息数: {len(history)} 条（全部保留）")
    for msg in history:
        role = "用户" if isinstance(msg, HumanMessage) else "AI"
        print(f"    {role}: {msg.content}")


# =============================================================================
# 示例 2: 窗口记忆 — 只保留最近 K 条
# =============================================================================

def window_memory_demo():
    """
    窗口记忆 = 用 trim_messages 只保留最近 K 条消息，旧消息自动丢弃。

    token_counter=len 是教学简化：按"消息条数"计数。
    实际项目可换成真实的 token 计数器（如模型的 tokenizer）精确控制 Token。
    """
    print(f"\n-- 示例 2: 窗口记忆（trim_messages，保留最近 4 条）")

    history = [
        HumanMessage(content="第1轮：你好"),
        AIMessage(content="你好！有什么可以帮你？"),
        HumanMessage(content="第2轮：今天天气不错"),
        AIMessage(content="是啊，适合出去玩！"),
        HumanMessage(content="第3轮：我想去公园"),
        AIMessage(content="公园是个好选择！"),
        HumanMessage(content="第4轮：有什么推荐的活动"),
        AIMessage(content="可以散步、野餐！"),
        HumanMessage(content="第5轮：好的谢谢"),
        AIMessage(content="不客气，玩得开心！"),
    ]

    trimmed = trim_messages(
        history,
        max_tokens=4,
        strategy="last",
        token_counter=len,
    )

    print(f"  保存了 {len(history)} 条，但只保留 {len(trimmed)} 条（最近 4 条）")
    for msg in trimmed:
        role = "用户" if isinstance(msg, HumanMessage) else "AI"
        print(f"    {role}: {msg.content}")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n>>> 05_memory/buffer_memory — 记忆策略\n")

    window_memory_demo()
