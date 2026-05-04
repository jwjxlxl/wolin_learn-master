# =============================================================================
# JSON 结构化输出
# =============================================================================
#  
# 用途：教学演示 - 让 LLM 输出 JSON 格式并用 JsonOutputParser 解析
#
# 核心概念：
#   - 让 LLM 输出 JSON 格式
#   - 结构化数据处理
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


# =============================================================================
# 第一部分：理解 JSON 输出
# =============================================================================
"""
为什么需要 JSON 输出？

📦 场景
   AI 回复了一段文本："小明今年 25 岁，是一名软件工程师。"

   程序需要提取：
   - name: "小明"
   - age: 25
   - job: "软件工程师"

💡 解决方案
   让 AI 直接输出 JSON 格式：
   {
     "name": "小明",
     "age": 25,
     "job": "软件工程师"
   }

   程序可以直接解析使用！

🔧 JsonOutputParser 的作用
   1. 生成格式指令，告诉 AI 输出 JSON
   2. 解析 AI 的 JSON 输出为 Python 字典
"""


# =============================================================================
# 示例 1: 最简单的 JSON 输出
# =============================================================================

def simplest_json_output():
    """
    最简单的 JSON 输出示例

    让 AI 输出 JSON 格式
    """
    print("=" * 60)
    print("示例 1: 最简单的 JSON 输出")
    print("=" * 60)

    from langchain_ollama import ChatOllama
    from langchain_core.prompts import PromptTemplate

    # 设计 Prompt 让 AI 输出 JSON
    prompt = PromptTemplate.from_template("""
请用 JSON 格式输出以下信息：
- 姓名：小明
- 年龄：25
- 职业：软件工程师

JSON 格式：""")

    model = ChatOllama(model="qwen3.5:2b")
    response = model.invoke(prompt.format())

    print(f"AI 输出:\n{response.content}")
    print()


# =============================================================================
# 示例 2: 使用 JsonOutputParser（推荐方式）
# =============================================================================

def using_json_output_parser():
    """
    使用 LangChain 的 JsonOutputParser

    更规范、更可靠
    """
    print("=" * 60)
    print("示例 2: 使用 JsonOutputParser")
    print("=" * 60)

    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import PromptTemplate

    # 1. 创建 JsonOutputParser
    # 告诉它你需要的字段
    parser = JsonOutputParser()

    # 2. 创建 Prompt，包含格式指令
    prompt = PromptTemplate.from_template("""
请提取以下文本中的信息，输出为 JSON 格式。

文本：小明今年 25 岁，是一名软件工程师，在一家科技公司工作。

{format_instructions}

JSON 输出：""")

    # 3. 注入格式指令
    # parser.get_format_instructions() 会返回格式说明
    prompt_with_format = prompt.partial(
        format_instructions=parser.get_format_instructions()
    )

    # 4. 创建 Pipeline
    model = ChatOllama(model="qwen3.5:2b")
    chain = prompt_with_format | model | parser

    # 5. 调用
    result = chain.invoke({})

    print(f"解析结果：{result}")
    print(f"类型：{type(result)}")
    print(f"name: {result.get('name')}")
    print(f"age: {result.get('age')}")
    print()


# =============================================================================
# 示例 3: 指定 JSON Schema
# =============================================================================

def with_json_schema():
    """
    使用 JsonOutputParser 指定 JSON Schema

    更严格地控制输出格式
    """
    print("=" * 60)
    print("示例 3: 指定 JSON Schema")
    print("=" * 60)

    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import PromptTemplate

    # 定义期望的 JSON 结构（Pydantic 风格）
    from typing import List

    # 创建 Parser，指定字段说明
    parser = JsonOutputParser()

    prompt = PromptTemplate.from_template("""
请分析以下文本，提取人员信息。

文本：{text}

请按以下 JSON 格式输出：
{{
    "name": "姓名（字符串）",
    "age": 年龄（数字）,
    "hobbies": ["爱好 1", "爱好 2", ...]
}}

{format_instructions}

JSON 输出：""")

    prompt_with_format = prompt.partial(
        format_instructions=parser.get_format_instructions()
    )

    model = ChatOllama(model="qwen3.5:2b")
    chain = prompt_with_format | model | parser

    text = "小红今年 20 岁，她喜欢画画、游泳和听音乐。"
    result = chain.invoke({"text": text})

    print(f"输入：{text}")
    print(f"提取结果：{result}")
    print()


# =============================================================================
# 示例 4: 实用场景 - 产品评论分析
# =============================================================================

def product_review_analysis():
    """
    使用 JSON 输出分析产品评论
    """
    print("=" * 60)
    print("示例 4: 产品评论分析（实用场景）")
    print("=" * 60)

    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import PromptTemplate

    parser = JsonOutputParser()

    prompt = PromptTemplate.from_template("""
请分析以下产品评论，提取情感信息。

评论：{review}

请按以下 JSON 格式输出：
{{
    "sentiment": "情感倾向（positive/negative/neutral）",
    "rating": 评分（1-5 的数字）,
    "pros": ["优点 1", "优点 2", ...],
    "cons": ["缺点 1", "缺点 2", ...]
}}

{format_instructions}

JSON 输出：""")

    prompt_with_format = prompt.partial(
        format_instructions=parser.get_format_instructions()
    )

    model = ChatOllama(model="qwen3.5:2b")
    chain = prompt_with_format | model | parser

    # 测试不同评论
    reviews = [
        "这个产品很好用，质量不错，就是价格有点贵。",
        "完全不值这个价，用了一周就坏了，客服态度也很差。"
    ]

    for review in reviews:
        print(f"评论：{review}")
        try:
            result = chain.invoke({"review": review})
            print(f"分析结果：{result}")
        except Exception as e:
            print(f"解析失败：{e}")
        print()


# =============================================================================
# 示例 5: 错误处理
# =============================================================================

def error_handling():
    """
    JSON 解析的错误处理

    AI 可能输出不合法的 JSON
    """
    print("=" * 60)
    print("示例 5: JSON 解析错误处理")
    print("=" * 60)

    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import PromptTemplate

    parser = JsonOutputParser()

    prompt = PromptTemplate.from_template("""
请输出以下信息的 JSON：
- 名字：随便一个名字
- 年龄：一个数字

{format_instructions}

JSON 输出：""")

    prompt_with_format = prompt.partial(
        format_instructions=parser.get_format_instructions()
    )

    model = ChatOllama(model="qwen3.5:2b")
    chain = prompt_with_format | model | parser

    try:
        result = chain.invoke({})
        print(f"解析成功：{result}")
    except Exception as e:
        print(f"解析失败：{e}")
        print()
        print("提示：JSON 解析失败通常是因为 AI 输出的格式不正确")
        print("可以尝试：")
        print("  1. 在 Prompt 中强调'只输出 JSON，不要其他内容'")
        print("  2. 使用更强的模型")
        print("  3. 添加更多示例（Few-Shot）")
    print()


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  JSON 结构化输出 - JsonOutputParser")
    print("  说明：让 AI 输出 JSON 格式并解析")
    print("=" * 70 + "\n")

    print("【运行前检查】")
    print("  1. 已安装 Ollama 服务")
    print("  2. 已下载模型：ollama pull qwen3.5:2b")
    print()

    # 运行示例
    # simplest_json_output()
    # using_json_output_parser()
    # with_json_schema()
    # product_review_analysis()
    error_handling()

    print("=" * 70)
    print("  接下来学习：pydantic_parser.py（强类型解析）")
    print("=" * 70 + "\n")
