# =============================================================================
# 简单链
# =============================================================================
#  
# 用途：教学演示 - 使用 Chain 组合多个组件
#
# 核心概念：
#   - Chain = "功能组合包"
#   - Prompt + LLM + Parser = 完整功能
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
# 第一部分：理解 Chain
# =============================================================================
"""
什么是 Chain？

🔗 定义
   Chain = "链" = 多个组件连接在一起
   Prompt + Model + Parser = 一个完整功能

🎯 为什么需要 Chain？

   不用 Chain:
   # 每次都要写这么多代码
   prompt = PromptTemplate.from_template(...)
   model = ChatOllama(...)
   parser = StrOutputParser()
   formatted = prompt.format(...)
   response = model.invoke(formatted)
   result = parser.invoke(response)

   用 Chain:
   chain = prompt | model | parser
   result = chain.invoke({...})

💡 生活化比喻
   Chain = "套餐"
   单点：米饭 + 菜 + 汤（分开点麻烦）
   套餐：一键搞定（方便）
"""


# =============================================================================
# 示例 1: 回顾 Pipeline（最简单的 Chain）
# =============================================================================

def review_pipeline():
    """
    回顾之前学过的 Pipeline

    其实那就是最简单的 Chain
    """
    print("=" * 60)
    print("示例 1: 回顾 Pipeline（最简单的 Chain）")
    print("=" * 60)

    from langchain_core.prompts import PromptTemplate
    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import StrOutputParser

    # 创建 Chain
    prompt = PromptTemplate.from_template("请用一句话解释{topic}。")
    model = ChatOllama(model="qwen3:4b")
    parser = StrOutputParser()

    chain = prompt | model | parser

    # 调用
    result = chain.invoke({"topic": "人工智能"})

    print(f"AI 回复：{result}")
    print()


# =============================================================================
# 示例 2: 实用的翻译 Chain
# =============================================================================

def translation_chain():
    """
    创建一个实用的翻译链
    """
    print("=" * 60)
    print("示例 2: 翻译 Chain（实用）")
    print("=" * 60)

    from langchain_core.prompts import PromptTemplate
    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import StrOutputParser

    # 翻译 Chain
    prompt = PromptTemplate.from_template("""
你是一位专业翻译，请将以下文本从{source_lang}翻译成{target_lang}。

要求：
1. 保持原意
2. 翻译自然流畅
3. 只输出翻译结果

原文：{text}

译文：""")

    model = ChatOllama(model="qwen3:4b")
    parser = StrOutputParser()

    chain = prompt | model | parser

    # 使用
    texts = [
        {"source_lang": "英语", "target_lang": "中文", "text": "Hello, world!"},
        {"source_lang": "中文", "target_lang": "英语", "text": "今天天气真好！"},
        {"source_lang": "英语", "target_lang": "日语", "text": "Thank you!"},
    ]

    for t in texts:
        result = chain.invoke(t)
        print(f"{t['source_lang']} → {t['target_lang']}: {result}")
    print()


# =============================================================================
# 示例 3: 笑话生成 Chain
# =============================================================================

def joke_generator_chain():
    """
    创建一个笑话生成链
    """
    print("=" * 60)
    print("示例 3: 笑话生成 Chain（有趣）")
    print("=" * 60)

    from langchain_core.prompts import PromptTemplate
    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import StrOutputParser

    # 笑话生成 Chain
    prompt = PromptTemplate.from_template("""
你是一位幽默大师，请根据主题生成一个冷笑话。

主题：{topic}

要求：
1. 简短（50 字以内）
2. 有创意
3. 结尾有意想不到的反转

冷笑话：""")

    model = ChatOllama(model="qwen3:4b")
    parser = StrOutputParser()

    chain = prompt | model | parser

    # 生成笑话
    topics = ["程序员", "爱情", "工作", "学习"]

    for topic in topics:
        joke = chain.invoke({"topic": topic})
        print(f"【{topic}】{joke}")
    print()


# =============================================================================
# 示例 4: 诗歌创作 Chain
# =============================================================================

def poem_chain():
    """
    创建一个诗歌创作链
    """
    print("=" * 60)
    print("示例 4: 诗歌创作 Chain（文艺）")
    print("=" * 60)

    from langchain_core.prompts import PromptTemplate
    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import StrOutputParser

    # 诗歌创作 Chain
    prompt = PromptTemplate.from_template("""
你是一位诗人，请以"{topic}"为题，写一首四句短诗。

要求：
1. 每句 5-7 个字
2. 有意境
3. 押韵更好

短诗：""")

    model = ChatOllama(model="qwen3:4b")
    parser = StrOutputParser()

    chain = prompt | model | parser

    # 创作诗歌
    topics = ["春天", "月亮", "思念", "人生"]

    for topic in topics:
        print(f"【{topic}】")
        poem = chain.invoke({"topic": topic})
        print(poem)
        print()


# =============================================================================
# 示例 5: 可复用的 Chain 函数
# =============================================================================

def reusable_chain_function():
    """
    把 Chain 封装成函数，方便复用
    """
    print("=" * 60)
    print("示例 5: 可复用的 Chain 函数")
    print("=" * 60)

    from langchain_core.prompts import PromptTemplate
    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import StrOutputParser

    def create_translation_chain(source_lang, target_lang):
        """
        创建翻译 Chain 的工厂函数

        Args:
            source_lang: 源语言
            target_lang: 目标语言

        Returns:
            一个可以直接调用的 chain
        """
        prompt = PromptTemplate.from_template(f"""
你是一位专业翻译，请将文本从{source_lang}翻译成{target_lang}。

要求：
1. 保持原意
2. 只输出翻译结果

原文：{{text}}

译文：""")

        model = ChatOllama(model="qwen3:4b")
        parser = StrOutputParser()

        return prompt | model | parser

    # 创建特定方向的翻译链
    en_to_zh = create_translation_chain("英语", "中文")
    zh_to_en = create_translation_chain("中文", "英语")

    # 使用
    print("英译中：")
    result = en_to_zh.invoke({"text": "Artificial intelligence is amazing!"})
    print(result)
    print()

    print("中译英：")
    result = zh_to_en.invoke({"text": "中国的历史文化非常悠久。"})
    print(result)
    print()


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  简单链 - Simple Chain")
    print("  说明：用 Pipeline 组合组件")
    print("=" * 70 + "\n")

    print("【运行前检查】")
    print("  1. 已安装 Ollama 服务")
    print("  2. 已下载模型：ollama pull qwen3:4b")
    print()

    # 运行示例
    review_pipeline()
    translation_chain()
    joke_generator_chain()
    poem_chain()
    reusable_chain_function()

    print("=" * 70)
    print("  接下来学习：sequential_chain.py（多步骤链）")
    print("=" * 70 + "\n")
