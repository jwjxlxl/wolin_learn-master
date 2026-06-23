# =============================================================================
# 路由链 — 根据输入内容自动选择不同的处理路径
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解 Router Chain = "智能分线器"
#   ✅ 用 if-else 实现简单路由
#   ✅ 用 AI 自动判断输入类型并路由
# =============================================================================

import sys
import io

from langchain_community.llms.aviary import get_models

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)


"""
什么是 Router Chain？

  普通 Chain: 无论什么输入，都走同一条路
  路由 Chain: 根据输入类型，自动选择不同的处理路径

  例如客服系统:
    用户问题 → 路由判断 → 技术问题 → 技术客服
                         售后问题 → 售后客服
                         投诉     → 人工处理

  生活化比喻: Router = 医院分诊台
    病人 → 分诊护士判断 → 内科/外科/儿科...
"""

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from utils.model_utils import get_model


# =============================================================================
# 示例 1: if-else 路由 — 手动指定任务类型
# =============================================================================

def simple_if_else_router():
    """
    最基础的路由方式：根据外部指定的任务类型，选择不同的 Chain。

    代码清晰易懂，适合任务类型由用户手动选择的场景。
    """
    print(f"\n-- 示例 1: if-else 路由")

    # model = ChatOllama(model="qwen3.5:2b")
    model = get_model()
    parser = StrOutputParser()

    # 准备三个不同功能的 Chain
    translate_c = (PromptTemplate.from_template("翻译成英语: {text}") | model | parser)
    summarize_c = (PromptTemplate.from_template("用一句话总结: {text}") | model | parser)
    explain_c = (PromptTemplate.from_template("解释这个概念: {text}") | model | parser)

    def router(text: str, task: str) -> str:
        """根据 task 选择 Chain。"""
        chains = {"translate": translate_c, "summarize": summarize_c, "explain": explain_c}
        return chains.get(task, explain_c).invoke({"text": text})

    print(f"翻译: {router('你好，世界', 'translate')}")
    print(f"总结: {router('人工智能是研究如何让计算机具有人类智能的学科', 'summarize')}")
    print(f"解释: {router('机器学习', 'explain')}")


# =============================================================================
# 示例 2: 智能路由 — 让 AI 自己判断类型
# =============================================================================

def intelligent_router():
    """
    不让用户手动指定类型，而是先用 AI 判断输入属于哪种类型。

    两步走:
    1. 分类 Chain: AI 判断输入类型（translate/summarize/explain）
    2. 路由: 根据判断结果选择对应的 Chain

    这就是 Agent 的雏形——先思考再行动。
    """
    print(f"\n-- 示例 2: 智能路由（AI 自动判断类型）")

    # model = ChatOllama(model="qwen3.5:2b")
    model = get_model("qwen")
    # 分类 Chain — 判断输入类型
    classifier = (
        PromptTemplate.from_template("判断以下输入的类型，只输出一个词（translate/summarize/explain/other）:\n\n{input}\n\n类型:")
        | model | StrOutputParser()
    )

    # 功能 Chain
    translate_c = (PromptTemplate.from_template("翻译成英语: {text}") | model | StrOutputParser())
    summarize_c = (PromptTemplate.from_template("用一句话总结: {text}") | model | StrOutputParser())

    def smart_router(user_input: str) -> str:
        task = classifier.invoke({"input": user_input}).strip().lower()
        print(f"  AI 判断类型: {task}")

        if "translate" in task or "翻译" in task:
            return translate_c.invoke({"text": user_input})
        elif "summarize" in task or "总结" in task:
            return summarize_c.invoke({"text": user_input})
        return "无法判断该输入的类型"

    inputs = [
        "把'你好'翻译成英语",
        "总结一下: 人工智能是计算机科学的一个分支",
        "今天天气不错",
    ]
    for text in inputs:
        print(f"\n输入: \"{text}\"")
        print(f"结果: {smart_router(text)}")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n>>> 06_chains/router_chain — 路由链\n")

    # simple_if_else_router()
    intelligent_router()

    # 接下来学习: 07_retrieval/document_loader.py（文档加载）
