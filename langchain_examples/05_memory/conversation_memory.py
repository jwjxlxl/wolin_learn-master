# =============================================================================
# 对话记忆 — 让 AI "记住"之前说过的话
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解 LLM 为什么"失忆"（每次调用都是全新的）
#   ✅ 手动用消息列表实现对话记忆
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from utils.model_utils import setup_encoding
setup_encoding()


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
