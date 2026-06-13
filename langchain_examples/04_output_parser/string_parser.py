# =============================================================================
# 输出解析（一）— StrOutputParser：把 AI 回复转成纯字符串
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解 OutputParser 的作用：AI 的"翻译官"
#   ✅ 对比有无 Parser 的区别（Message 对象 vs 纯字符串）
#   ✅ 使用 batch() 批量处理多个输入
# =============================================================================

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)


"""
什么是 OutputParser？

  AI 的原始输出是 Message 对象（带着元数据），
  程序需要的是纯字符串。Parser = 翻译官，负责这个转换。

  不用 Parser: response.content   → 每次都手动取
  用 Parser:   prompt | model | parser → 自动返回字符串
"""

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser


# =============================================================================
# 示例 1: 不用 Parser vs 用 Parser
# =============================================================================

def compare_with_without_parser():
    """
    对比有无 Parser 的代码差异。

    不用 Parser: invoke() 返回 Message 对象，需要 .content 取文本
    用 Parser:    invoke() 直接返回纯字符串
    """
    model = ChatOllama(model="qwen3.5:2b")
    question = "请用一句话介绍 Python。"

    # 不用 Parser
    print(f"\n{'─' * 50}")
    print("[不用 Parser]")
    response = model.invoke(question)
    print(f"  返回类型: {type(response).__name__}")
    print(f"  取文本: response.content → {response.content}")

    # 用 Parser
    print(f"\n[用 StrOutputParser]")
    chain = model | StrOutputParser()
    result = chain.invoke(question)
    print(f"  返回类型: {type(result).__name__}")
    print(f"  直接就是: {result}")


# =============================================================================
# 示例 2: 完整的 Pipeline + 批量处理
# =============================================================================

def pipeline_with_batch():
    """
    prompt | model | parser 三段式 + batch() 批量处理。

    batch() 一次处理多个输入，比 for 循环调 invoke() 更方便。
    """
    print(f"\n-- 示例 2: 三段式 Pipeline + 批量处理")

    prompt = PromptTemplate.from_template("用{num}个字介绍{topic}。")
    model = ChatOllama(model="qwen3.5:2b")
    chain = prompt | model | StrOutputParser()

    # 批量输入
    inputs = [
        {"topic": "Python", "num": 10},
        {"topic": "Java",   "num": 10},
        {"topic": "C++",    "num": 10},
    ]

    for inp, result in zip(inputs, chain.batch(inputs)):
        print(f"  {inp['topic']}: {result}")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n>>> 04_output_parser/string_parser — StrOutputParser\n")

    compare_with_without_parser()
    pipeline_with_batch()

    # 接下来学习: json_parser.py（JSON 结构化输出）
