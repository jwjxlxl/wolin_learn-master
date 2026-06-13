# =============================================================================
# 对话记忆 — 让 AI "记住"之前说过的话
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解 LLM 为什么"失忆"（每次调用都是全新的）
#   ✅ 手动用消息列表实现对话记忆
# =============================================================================

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)


"""
为什么 LLM 记不住之前的对话？

  LLM 本身是"无状态"的——每次 model.invoke() 都是独立的。
  就像每次都拨通一个新电话，对方不知道你上一通说了什么。

  解决方案: 手动保存历史消息，每次调用时把历史一起传过去。
  这就像每次打电话前先翻看聊天记录，然后把记录一起给 AI 看。
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage


# =============================================================================
# 示例 1: 没有记忆 — AI 会"失忆"
# =============================================================================

def without_memory():
    """
    两次独立的 model.invoke()，AI 完全不记得第一轮说了什么。

    这是 LLM 的默认行为——每次调用都是全新的对话。
    """
    print(f"\n-- 示例 1: 没有记忆（失忆的 AI）")

    model = ChatOllama(model="qwen3.5:2b")

    # 第一轮
    r = model.invoke([HumanMessage(content="我叫小明，请记住这个名字。")])
    print(f"[第1轮] {r.content}")

    # 第二轮 — 新的调用，不传历史
    r = model.invoke([HumanMessage(content="我叫什么名字？")])
    print(f"[第2轮] {r.content}")
    print("  ↑ AI 不记得了！")


# =============================================================================
# 示例 2: 手动实现记忆 — 保存历史 + 每次传过去
# =============================================================================

def manual_memory():
    """
    用 Python 列表保存对话历史，每次调用时把完整历史传给模型。

    步骤:
    1. history = []                             — 初始化空历史
    2. history.append(HumanMessage(...))        — 加用户消息
    3. response = model.invoke(history)          — 带历史调模型
    4. history.append(AIMessage(...))           — 加 AI 回复

    这是所有 Memory 组件的底层原理——后面 Agent 模块会用 InMemorySaver 更优雅地做这件事。
    """
    print(f"\n-- 示例 2: 手动实现记忆")

    model = ChatOllama(model="qwen3.5:2b")
    history = []

    # 第一轮
    history.append(HumanMessage(content="我叫小明，今年10岁，喜欢打篮球。"))
    r = model.invoke(history)
    print(f"[第1轮] {r.content}")
    history.append(AIMessage(content=r.content))

    # 第二轮 — 带完整历史
    history.append(HumanMessage(content="我叫什么名字？我几岁？我喜欢什么？"))
    r = model.invoke(history)
    print(f"[第2轮] {r.content}")
    print("  ↑ 这次 AI 记住了！因为历史被一起传了进去")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n>>> 05_memory/conversation_memory — 对话记忆\n")

    without_memory()
    manual_memory()

    # 接下来学习: buffer_memory.py（不同记忆策略）
