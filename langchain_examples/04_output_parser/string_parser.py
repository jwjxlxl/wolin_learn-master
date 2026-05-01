# =============================================================================
# 字符串解析
# =============================================================================
#  
# 用途：教学演示 - 使用 OutputParser 解析 LLM 输出
#
# 核心概念：
#   - OutputParser 的作用：把 AI 的文本输出转成程序可用的格式
#   - StrOutputParser: 最简单的字符串解析
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


# =============================================================================
# 第一部分：理解 OutputParser
# =============================================================================
"""
什么是 OutputParser？

🤔 为什么要解析？

   AI 的原始输出：
   - 是一个 Message 对象
   - 可能包含多余内容（"好的，让我来解释..."）
   - 格式不固定

   程序需要的输出：
   - 简洁的字符串
   - 结构化的数据（JSON、列表等）
   - 特定类型（数字、布尔值等）

💡 OutputParser 的作用
   翻译官：把 AI 的"人话"转成"机器能用的数据"

📊 常见的 Parser 类型
   - StrOutputParser: 转字符串（最简单）
   - JsonOutputParser: 转 JSON
   - PydanticOutputParser: 转 Pydantic 模型（强类型）
   - CommaSeparatedListOutputParser: 转列表
"""


# =============================================================================
# 示例 1: 不使用 Parser vs 使用 Parser
# =============================================================================

def without_parser():
    """
    不使用 Parser 的情况

    需要手动处理 response 对象
    """
    print("=" * 60)
    print("示例 1: 不使用 Parser（原始方式）")
    print("=" * 60)

    from langchain_ollama import ChatOllama

    model = ChatOllama(model="qwen3.5:2b")

    # 调用模型
    response = model.invoke("请用一句话介绍 Python。")

    # 需要手动提取 .content
    print(f"response 类型：{type(response)}")
    print(f"response.content 类型：{type(response.content)}")
    print(f"内容：{response.content}")
    print()


def with_parser():
    """
    使用 Parser 的情况

    自动转成字符串，更简洁
    """
    print("=" * 60)
    print("示例 1: 使用 Parser（推荐方式）")
    print("=" * 60)

    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate

    # 创建 Pipeline：Prompt + Model + Parser
    prompt = PromptTemplate.from_template("请用一句话介绍{topic}。")
    model = ChatOllama(model="qwen3.5:2b")
    parser = StrOutputParser()

    chain = prompt | model | parser

    # 调用 Pipeline
    result = chain.invoke({"topic": "Python"})

    # 直接返回字符串，不是 Message 对象
    print(f"result 类型：{type(result)}")
    print(f"内容：{result}")
    print()


# =============================================================================
# 示例 2: StrOutputParser 的作用
# =============================================================================

def str_parser_details():
    """
    详细了解 StrOutputParser

    它做了什么处理？
    """
    print("=" * 60)
    print("示例 2: StrOutputParser 详解")
    print("=" * 60)

    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.messages import AIMessage

    # StrOutputParser 的核心逻辑很简单：
    # 输入：AIMessage 对象
    # 输出：.content 属性（字符串）

    parser = StrOutputParser()

    # 模拟一个 AI 回复
    mock_response = AIMessage(content="Python 是一种高级编程语言。")

    # 解析
    result = parser.invoke(mock_response)

    print(f"输入：{mock_response}")
    print(f"输出：{result}")
    print(f"输出类型：{type(result)}")
    print()

    # 在 Pipeline 中使用
    model = ChatOllama(model="qwen3.5:2b")
    chain = model | parser

    result = chain.invoke("什么是 AI？")
    print(f"Pipeline 输出：{result}")
    print()


# =============================================================================
# 示例 3: 实用场景 - 批量处理
# =============================================================================

def batch_processing():
    """
    使用 Pipeline 批量处理多个输入
    """
    print("=" * 60)
    print("示例 3: 批量处理（Batch Processing）")
    print("=" * 60)

    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate

    prompt = PromptTemplate.from_template("用{num}个字介绍{topic}。")
    model = ChatOllama(model="qwen3.5:2b")
    parser = StrOutputParser()

    chain = prompt | model | parser

    # 批量输入
    inputs = [
        {"topic": "Python", "num": 10},
        {"topic": "Java", "num": 10},
        {"topic": "C++", "num": 10},
    ]

    # batch() 批量处理
    results = chain.batch(inputs)

    for i, (input_data, result) in enumerate(zip(inputs, results), 1):
        print(f"{i}. {input_data['topic']}: {result}")

    print()


# =============================================================================
# 示例 4: 实用场景 - 提取特定信息
# =============================================================================

def extract_information():
    """
    使用 Prompt + Parser 提取特定信息
    """
    print("=" * 60)
    print("示例 4: 提取特定信息")
    print("=" * 60)

    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate

    # 设计 Prompt 让 AI 输出特定格式
    prompt = PromptTemplate.from_template("""
请从以下文本中提取人名、年龄、职业，每行一个字段。
格式：
人名：xxx
年龄：xxx
职业：xxx

文本：{text}

提取结果：""")

    model = ChatOllama(model="qwen3.5:2b")
    parser = StrOutputParser()

    chain = prompt | model | parser

    text = "小明今年 25 岁，是一名软件工程师，在一家科技公司工作。"
    result = chain.invoke({"text": text})

    print(f"输入：{text}")
    print(f"提取结果:\n{result}")
    print()


# =============================================================================
# 示例 5: 处理 AI 输出中的多余内容
# =============================================================================

def clean_output():
    """
    使用 Parser 清理 AI 输出中的多余内容
    """
    print("=" * 60)
    print("示例 5: 清理多余内容")
    print("=" * 60)

    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate

    # 设计 Prompt 让 AI 只输出关键内容
    prompt = PromptTemplate.from_template("""
请直接输出答案，不要有任何解释。

问题：{question}
答案：""")

    model = ChatOllama(model="qwen3.5:2b")
    parser = StrOutputParser()

    chain = prompt | model | parser

    # 测试
    result = chain.invoke({"question": "1 + 1 等于几？"})
    print(f"问题：1 + 1 等于几？")
    print(f"答案：{result}")
    print()


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  字符串解析 - StrOutputParser")
    print("  说明：把 AI 输出转成字符串")
    print("=" * 70 + "\n")

    print("【运行前检查】")
    print("  1. 已安装 Ollama 服务")
    print("  2. 已下载模型：ollama pull qwen3.5:2b")
    print()

    # 运行示例
    without_parser()
    with_parser()
    str_parser_details()
    batch_processing()
    extract_information()
    clean_output()

    print("=" * 70)
    print("  接下来学习：json_parser.py（JSON 结构化输出）")
    print("=" * 70 + "\n")
