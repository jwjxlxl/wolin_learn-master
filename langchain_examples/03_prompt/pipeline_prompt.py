# =============================================================================
# Pipeline 组合
# =============================================================================
#  
# 用途：教学演示 - 使用 Pipeline 组合多个组件
#
# 核心概念：
#   - Pipeline = "流水线"
#   - 多个步骤串联成一个流程
# =============================================================================

# -----------------------------------------------------------------------------
# 运行前检查
# -----------------------------------------------------------------------------
# 1. 已安装 Ollama 服务
# 2. 已下载模型：ollama pull qwen3:4b
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
# 第一部分：理解 Pipeline
# =============================================================================
"""
什么是 Pipeline？

🏭 定义
   Pipeline = "流水线"
   把多个步骤串联起来，形成一个完整流程

🎯 为什么需要 Pipeline？

   不用 Pipeline:
   1. 手动创建 Prompt
   2. 调用 Model
   3. 解析 Output
   4. 代码分散，不易维护

   用 Pipeline:
   Prompt → Model → Output Parser
   (一条链，一气呵成)

💡 生活化比喻
   Pipeline 就像餐厅的流水线：
   点单 → 做菜 → 上菜
   每个环节自动流转到下一个
"""


# =============================================================================
# 示例 1: 最简单的 Pipeline
# =============================================================================

def simplest_pipeline():
    """
    最简单的 Pipeline 示例

    Prompt + Model = 最简单的链
    """
    print("=" * 60)
    print("示例 1: 最简单的 Pipeline")
    print("=" * 60)

    from langchain_core.prompts import PromptTemplate
    from langchain_ollama import ChatOllama

    # 创建 Prompt 模板
    prompt = PromptTemplate.from_template("请用一句话解释{concept}。")

    # 创建模型
    model = ChatOllama(model="qwen3:4b")

    # 组合成 Pipeline
    # | 操作符表示"连接到下一个"
    chain = prompt | model

    # 调用 Pipeline
    # invoke() 会自动：
    # 1. 用输入填充 Prompt
    # 2. 调用 Model
    # 3. 返回结果
    response = chain.invoke({"concept": "人工智能"})

    # response 是 Message 对象，.content 获取文本
    print(f"AI 回复：{response.content}")
    print()


# =============================================================================
# 示例 2: Pipeline 的工作流程
# =============================================================================

def pipeline_flow():
    """
    详细展示 Pipeline 的数据流转
    """
    print("=" * 60)
    print("示例 2: Pipeline 数据流转")
    print("=" * 60)

    from langchain_core.prompts import PromptTemplate
    from langchain_ollama import ChatOllama

    prompt = PromptTemplate.from_template("翻译：{text}")
    model = ChatOllama(model="qwen3:4b")

    # 创建链
    chain = prompt | model

    print("数据流转过程:")
    print()

    # 第 1 步：输入
    input_data = {"text": "Hello, world!"}
    print(f"1. 输入：{input_data}")

    # 第 2 步：Prompt 处理
    formatted = prompt.format(**input_data)
    print(f"2. Prompt: {formatted}")

    # 第 3 步：Model 处理
    response = chain.invoke(input_data)
    print(f"3. Model 输出：{response.content}")
    print()


# =============================================================================
# 示例 3: 三段式 Pipeline（完整流程）
# =============================================================================

def three_stage_pipeline():
    """
    Prompt + Model + Parser 完整流程
    """
    print("=" * 60)
    print("示例 3: 三段式 Pipeline（完整流程）")
    print("=" * 60)

    from langchain_core.prompts import PromptTemplate
    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import StrOutputParser

    # 1. Prompt 模板
    prompt = PromptTemplate.from_template(
        "请用一句话解释{concept}。"
    )

    # 2. 模型
    model = ChatOllama(model="qwen3:4b")

    # 3. 输出解析器
    # StrOutputParser: 把 Message 对象转成字符串
    parser = StrOutputParser()

    # 组合成完整 Pipeline
    chain = prompt | model | parser

    # 调用
    # 返回的是字符串（不是 Message 对象）
    result = chain.invoke({"concept": "机器学习"})

    print(f"最终输出（字符串）: {result}")
    print(f"类型：{type(result)}")
    print()


# =============================================================================
# 示例 4: Pipeline 的强大之处 - 链式组合
# =============================================================================

def chained_pipeline():
    """
    多个 Pipeline 链式组合

    第一个 Pipeline 的输出 → 第二个 Pipeline 的输入
    """
    print("=" * 60)
    print("示例 4: 链式组合（多步骤处理）")
    print("=" * 60)

    from langchain_core.prompts import PromptTemplate
    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import StrOutputParser

    # Pipeline 1: 生成内容
    generate_prompt = PromptTemplate.from_template(
        "列出 3 个关于{topic}的关键词，用逗号分隔。"
    )
    model = ChatOllama(model="qwen3:4b")
    parser = StrOutputParser()

    generate_chain = generate_prompt | model | parser

    # Pipeline 2: 解释内容
    explain_prompt = PromptTemplate.from_template(
        "请解释以下关键词：{keywords}"
    )
    explain_chain = explain_prompt | model | parser

    # 组合两个 Pipeline
    # 第一个的输出作为第二个的输入
    full_chain = generate_chain | explain_chain

    # 调用
    result = full_chain.invoke({"topic": "人工智能"})

    print("完整处理流程:")
    print(f"输入：人工智能")
    print(f"输出：{result}")
    print()


# =============================================================================
# 示例 5: 实用场景 - 文章生成器
# =============================================================================

def article_generator():
    """
    使用 Pipeline 创建文章生成器

    多步骤：生成大纲 → 写正文 → 起标题
    """
    print("=" * 60)
    print("示例 5: 文章生成器（实用场景）")
    print("=" * 60)

    from langchain_core.prompts import PromptTemplate
    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import StrOutputParser

    model = ChatOllama(model="qwen3:4b")
    parser = StrOutputParser()

    # 步骤 1: 生成大纲
    outline_prompt = PromptTemplate.from_template("""
请为"{topic}"主题生成一个简单的大纲，包含 3 个要点。
格式：
1. ...
2. ...
3. ...
""")
    outline_chain = outline_prompt | model | parser

    # 步骤 2: 根据大纲写正文
    content_prompt = PromptTemplate.from_template("""
请根据以下大纲写一篇短文：
{outline}

要求：简洁明了，200 字左右。
""")
    content_chain = content_prompt | model | parser

    # 步骤 3: 起标题
    title_prompt = PromptTemplate.from_template("""
请为以下内容起一个吸引人的标题（10 字以内）：
{content}

标题：""")
    title_chain = title_prompt | model | parser

    # 组合完整流程
    # 注意：这里需要手动串联，因为输出要作为下一个的输入
    print("生成文章...")
    print()

    # 第 1 步：生成大纲
    topic = "人工智能的发展"
    print(f"主题：{topic}")
    outline = outline_chain.invoke({"topic": topic})
    print(f"大纲:\n{outline}")
    print()

    # 第 2 步：写正文
    content = content_chain.invoke({"outline": outline})
    print(f"正文:\n{content}")
    print()

    # 第 3 步：起标题
    title = title_chain.invoke({"content": content})
    print(f"标题：{title}")
    print()


# =============================================================================
# 示例 6: RunnableLambda（自定义处理步骤）
# =============================================================================

def custom_pipeline_step():
    """
    使用 RunnableLambda 添加自定义处理步骤
    """
    print("=" * 60)
    print("示例 6: 自定义处理步骤（RunnableLambda）")
    print("=" * 60)

    from langchain_core.prompts import PromptTemplate
    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableLambda

    # 自定义处理函数
    def add_word_count(text):
        """在文本后添加字数统计"""
        return f"{text}\n\n[字数：{len(text)}]"

    prompt = PromptTemplate.from_template("请用 50 字介绍{topic}。")
    model = ChatOllama(model="qwen3:4b")
    parser = StrOutputParser()

    # 组合 Pipeline，插入自定义步骤
    chain = prompt | model | parser | RunnableLambda(add_word_count)

    result = chain.invoke({"topic": "Python"})
    print(result)
    print()


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  Pipeline 组合")
    print("  说明：把多个组件串联成流水线")
    print("=" * 70 + "\n")

    print("【运行前检查】")
    print("  1. 已安装 Ollama 服务")
    print("  2. 已下载模型：ollama pull qwen3:4b")
    print()

    # 运行示例
    simplest_pipeline()
    pipeline_flow()
    three_stage_pipeline()
    chained_pipeline()
    article_generator()
    custom_pipeline_step()

    print("=" * 70)
    print("  接下来学习：04_output_parser/string_parser.py")
    print("=" * 70 + "\n")
