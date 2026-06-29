# =============================================================================
# 自问自答模式 — Self-Ask
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解 Self-Ask 的核心思想：把大问题拆成多个小问题，逐个回答
#   ✅ 实现基础版 Self-Ask：通过结构化 prompt 引导模型自问自答
#   ✅ 实现工具版 Self-Ask：模型自主决定追问什么问题
#
# 运行前检查：
# 1. 已安装 Ollama 并下载模型：ollama pull qwen3.5:2b
# =============================================================================

import sys
import os
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 确保项目根目录和 agent/ 目录可导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from utils.model_utils import get_model
from helpers import simulate_search


# =============================================================================
# 核心概念：Self-Ask（自问自答）
# =============================================================================
"""
提出背景：Microsoft Research 2022 年研究工作《Self-Ask with Search》。

核心思想：让模型在回答时学会"反问自己"，把大问题拆成多个小问题，
然后逐个回答。

生活化比喻：
  直接回答 = 被问到"马云的公司总部在哪" → 想不起来就瞎猜
  Self-Ask = "马云的公司总部在哪？"
    → 先问自己：马云创办的公司叫什么？ → 阿里巴巴
    → 再问自己：阿里巴巴总部在哪？ → 杭州
    → 最终答：杭州

适用场景：事实链路长、需要多步检索的问题
"""


# =============================================================================
# 示例 1: 基础 Self-Ask — 结构化 prompt 驱动
# =============================================================================

def basic_self_ask():
    """
    Self-Ask 核心循环：
      1. 模型判断是否需要追问（生成 Follow-up 问题）
      2. 模拟搜索回答 Follow-up 问题
      3. 组合所有中间答案，生成最终回答

    示例："2016年奥斯卡最佳男主角的年龄是多少？"
      → 追问1：2016年奥斯卡最佳男主角是谁？ → 莱昂纳多·迪卡普里奥
      → 追问2：莱昂纳多·迪卡普里奥当时多大？ → 41 岁
      → 最终：41 岁
    """
    print(f"\n-- 示例 1: 基础 Self-Ask — 结构化 prompt 驱动")

    model = get_model()
    if model is None:
        print("  【跳过】请安装 Ollama 并下载模型：ollama pull qwen3.5:2b")
        return

    # Self-Ask 系统提示：强制模型按格式输出
    self_ask_system = """你是一个善于提问的助手。当你被问到一个需要多步信息才能回答的问题时，请按以下格式工作：

1. 先判断：要回答这个问题，我需要知道什么？
2. 生成一个 Follow-up 问题（以 "Follow-up:" 开头）
3. 等待回答
4. 继续生成下一个 Follow-up 问题，直到你有足够信息
5. 最后给出最终答案（以 "Final answer:" 开头）

格式示例：
  问：2016年奥斯卡最佳男主角的年龄是多少？
  你的回答：
    Follow-up: 2016年奥斯卡最佳男主角是谁？
    [等待回答]
    Intermediate answer: 莱昂纳多·迪卡普里奥
    Follow-up: 莱昂纳多·迪卡普里奥在2016年时多大？
    [等待回答]
    Intermediate answer: 41 岁
    Final answer: 2016年奥斯卡最佳男主角莱昂纳多·迪卡普里奥当时 41 岁。

如果你已经有足够信息直接回答，请直接输出 "Final answer: ..."。"""

    question = "2016年奥斯卡最佳男主角的年龄是多少？"

    print(f"  问题：{question}")
    print()

    # 手动实现 Self-Ask 循环
    follow_up_count = 0
    intermediate_answers = []
    context = ""

    for round_num in range(5):  # 最多 5 轮追问
        # 构建当前轮次的 prompt
        current_prompt = question
        if context:
            current_prompt = f"{question}\n\n已知信息：\n{context}"

        messages = [
            ("system", self_ask_system),
            ("user", current_prompt),
        ]

        prompt = ChatPromptTemplate.from_messages(messages)
        response = (prompt | model).invoke({})
        response_text = response.content

        print(f"  【第 {round_num + 1} 轮】模型输出：")
        for line in response_text.split("\n"):
            print(f"    {line}")
        print()

        # 解析响应
        if "Final answer:" in response_text:
            # 提取最终答案
            final_idx = response_text.index("Final answer:")
            final_answer = response_text[final_idx + len("Final answer:"):].strip()
            print(f"  ✅ 最终答案：{final_answer}")
            break

        # 提取 Follow-up 问题
        lines = response_text.split("\n")
        follow_up = None
        for line in lines:
            if line.strip().startswith("Follow-up:"):
                follow_up = line.replace("Follow-up:", "").strip()
                break

        if follow_up:
            follow_up_count += 1
            print(f"  🔍 追问 {follow_up_count}: {follow_up}")

            # 模拟搜索获取答案
            intermediate = simulate_search(follow_up)
            print(f"  📋 中间答案: {intermediate}")
            intermediate_answers.append(intermediate)

            context += f"- {follow_up} → {intermediate}\n"
        else:
            # 模型没有生成 Follow-up，可能是已经回答了
            print(f"  模型未生成追问，尝试提取答案...")
            print(f"  ✅ 最终答案：{response_text[:200]}")
            break

    print(f"\n  【统计】共追问 {follow_up_count} 次")


# =============================================================================
# 示例 2: Self-Ask + bind_tools — 模型自主决定追问
# =============================================================================

def self_ask_with_tools():
    """
    使用 LangChain bind_tools 实现 Self-Ask：
    定义一个 "ask_follow_up" 工具，模型可以自主决定何时追问。

    示例："Python之父是谁？他出生在哪座城市？"
    """
    print(f"\n-- 示例 2: Self-Ask + bind_tools — 模型自主决定追问")

    model = get_model()
    if model is None:
        print("  【跳过】请安装 Ollama 并下载模型：ollama pull qwen3.5:2b")
        return

    # 定义追问工具
    @tool
    def search_info(query: str) -> str:
        """当需要查找某个事实信息时调用此工具。query 是你要搜索的关键词。"""
        result = simulate_search(query)
        print(f"  [搜索] {query} → {result}")
        return result

    tools = [search_info]
    model_with_tools = model.bind_tools(tools)

    # 系统提示
    system_prompt = """你是一个信息查找助手。当被问到需要搜索才能回答的问题时，
请使用 search_info 工具来获取信息。可以多次调用工具逐步查找，
最后给出完整答案。"""

    question = "Python之父是谁？他出生在哪座城市？"

    print(f"  问题：{question}")
    print()

    from langchain_core.messages import HumanMessage, ToolMessage

    messages = [
        ("system", system_prompt),
        ("human", question),
    ]

    for i in range(5):
        response = model_with_tools.invoke(
            [{"role": m["role"] if isinstance(m, dict) else m.type,
              "content": m.content if hasattr(m, 'content') else m}
             if not isinstance(m, str) else m for m in messages]
            if not all(isinstance(m, (dict, str)) for m in messages)
            else messages
        )

        # 检查是否有工具调用
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                query = tool_args.get("query", "")
                result = search_info.invoke({"query": query})
                messages.append(response)
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        else:
            print(f"  ✅ 最终答案：{response.content}")
            break

    print(f"\n  【消息总数】{len(messages)} 条")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  Self-Ask（自问自答）模式")
    print("  把大问题拆成小问题，逐步查找、逐步回答")
    print("=" * 70 + "\n")

    basic_self_ask()
    # self_ask_with_tools()

    print("\n" + "=" * 70)
    print("  接下来学习：tree_of_thoughts.py（Tree of Thoughts 树状思维）")
    print("=" * 70 + "\n")
