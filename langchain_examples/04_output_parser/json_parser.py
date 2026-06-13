# =============================================================================
# 输出解析（二）— JsonOutputParser：让 AI 输出结构化 JSON
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 让 AI 输出 JSON 格式而非自由文本
#   ✅ 使用 JsonOutputParser 自动解析为 Python dict
#   ✅ 在实际场景（产品评论分析）中应用 JSON 输出
# =============================================================================

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)


"""
为什么需要 JSON 输出？

  文本输出: "小明今年25岁，是一名软件工程师"
  → 程序需要自己从文字中提取 name/age/job，容易出错

  JSON 输出: {"name": "小明", "age": 25, "job": "软件工程师"}
  → 程序直接 json.loads() 就能用，准确可靠

  生活化比喻: JSON 输出 = 让 AI 填表格而不是写作文
"""

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser


# =============================================================================
# 示例 1: 用 JsonOutputParser 提取信息
# =============================================================================

def using_json_parser():
    """
    三步使用 JsonOutputParser：
    1. 创建 JsonOutputParser 实例
    2. 在 Prompt 中注入 parser.get_format_instructions()（告诉 AI 输出格式）
    3. Pipeline 中串联 parser，自动解析为 Python dict

    parser.get_format_instructions() 会自动生成类似：
    "请输出 JSON 格式，字段包括: name, age, job"
    """
    print(f"\n-- 示例 1: JsonOutputParser 提取人物信息")

    parser = JsonOutputParser()

    prompt = PromptTemplate.from_template("""
请从文本中提取姓名、年龄、职业，输出为 JSON。

文本: 小明今年25岁，是一名软件工程师，在一家科技公司工作。

{format_instructions}

JSON:""")

    # 注入格式指令
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    model = ChatOllama(model="qwen3.5:2b")
    chain = prompt | model | parser

    result = chain.invoke({})
    print(f"  解析结果: {result}")
    print(f"  name: {result.get('name')}, age: {result.get('age')}, job: {result.get('job')}")


# =============================================================================
# 示例 2: 实用场景 — 产品评论分析
# =============================================================================

def product_review_analysis():
    """
    用 JsonOutputParser 做产品评论的情感分析。

    一个 Chain 可以批量分析多条评论，输出统一结构。
    这在实际产品中非常有用——比如自动分析用户反馈。
    """
    print(f"\n-- 示例 2: 产品评论分析（实用场景）")

    parser = JsonOutputParser()

    prompt = PromptTemplate.from_template("""
分析以下产品评论，输出 JSON。

评论: {review}

请按此格式:
{{
    "sentiment": "情感（positive/negative/neutral）",
    "rating": 评分（1-5 的数字）,
    "summary": "一句话总结"
}}

{format_instructions}

JSON:""")

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    model = ChatOllama(model="qwen3.5:2b")
    chain = prompt | model | parser

    reviews = [
        "这个产品很好用，质量不错，就是价格有点贵。",
        "完全不值这个价，用了一周就坏了，客服态度也很差。",
    ]

    for review in reviews:
        result = chain.invoke({"review": review})
        print(f"  评论: {review}")
        print(f"  分析: {result}\n")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n>>> 04_output_parser/json_parser — JsonOutputParser\n")

    using_json_parser()
    product_review_analysis()

    # 接下来学习: pydantic_parser.py（强类型解析）
