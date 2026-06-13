# =============================================================================
# 简单链 — 用 LCEL（LangChain Expression Language）串联组件
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解 Pipeline 的核心概念：Prompt → Model → Parser 一气呵成
#   ✅ 使用 | 操作符把组件串联成"链"
#   ✅ 封装可复用的 Chain 函数
# =============================================================================

# -----------------------------------------------------------------------------
# 运行前检查
# -----------------------------------------------------------------------------
# 1. 已安装 Ollama 服务
# 2. 已下载模型：ollama pull qwen3.5:2b
# -----------------------------------------------------------------------------

# 设置 UTF-8 编码（Windows 中文环境必需）
import sys
import io
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True
)

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser


# =============================================================================
# 第一部分：理解 LCEL Pipeline
# =============================================================================
"""
什么是 LCEL Pipeline（管道）？

🏭 核心概念：
   Pipeline = "流水线" = 用 | 操作符把多个组件串起来，数据自动流转

   没有 Pipeline 时（手动分步）：
      formatted = prompt.format(concept="人工智能")     # 第1步：填模板
      response = model.invoke(formatted)                # 第2步：调模型
      result = response.content                         # 第3步：取文本
   → 3 个中间变量，代码分散

   有了 Pipeline：
      chain = prompt | model | parser
      result = chain.invoke({"concept": "人工智能"})
   → 一行定义，一行调用，清晰简洁

💡 生活化比喻：
   Pipeline = 餐厅流水线
   点单 → 做菜 → 上菜
   每个环节自动流转到下一个，不需要服务员手工传递
"""


# =============================================================================
# 示例 1: 三段式 Pipeline — Prompt + Model + Parser
# =============================================================================

def three_stage_pipeline():
    """
    演示最经典的 LCEL 三段式：prompt | model | parser。

    核心概念：
    - | 操作符：LangChain 的"管道符号"，左边输出自动成为右边输入
    - PromptTemplate：填空题模板，用 {变量名} 做占位符
    - StrOutputParser：把 Model 返回的 Message 对象转成纯字符串
    - invoke()：往管道里投入数据，自动完成所有步骤

    数据流：
    {"concept": "机器学习"}
      → PromptTemplate 填模板
      → ChatOllama 调模型
      → StrOutputParser 转字符串
      → "机器学习是..."（纯文本）
    """
    print("=" * 60)
    print("示例 1: 三段式 Pipeline（Prompt → Model → Parser）")
    print("=" * 60)

    # 1. 创建模板：{concept} 是占位符
    prompt = PromptTemplate.from_template(
        "请用一句话解释{concept}。"
    )

    # 2. 创建模型
    model = ChatOllama(model="qwen3.5:2b")

    # 3. 创建解析器：把 Message 对象转成字符串
    parser = StrOutputParser()

    # 4. 用 | 串成 Pipeline
    chain = prompt | model | parser

    # 5. 投入数据，自动完成所有步骤
    result = chain.invoke({"concept": "人工智能"})

    # result 已经是纯字符串了，不是 Message 对象
    print(f"输出：{result}")
    print(f"类型：{type(result).__name__}（已是字符串，无需 .content）")
    print()


# =============================================================================
# 示例 2: 实用场景 — 翻译链
# =============================================================================

def translation_chain():
    """
    一个可直接使用的翻译 Chain。

    展示如何通过 Prompt 中的 {变量} 来灵活控制 Chain 的行为。
    同一个 Chain，不同输入参数 → 不同翻译方向。
    """
    print("=" * 60)
    print("示例 2: 翻译链（实用场景）")
    print("=" * 60)

    # 创建翻译专用的 Prompt
    prompt = PromptTemplate.from_template("""
你是一位专业翻译，请将以下文本从{source_lang}翻译成{target_lang}。

要求：
1. 保持原意
2. 翻译自然流畅
3. 只输出翻译结果，不要解释

原文：{text}

译文：""")

    model = ChatOllama(model="qwen3.5:2b")
    chain = prompt | model | StrOutputParser()

    # 同一个 Chain，不同参数 → 不同翻译方向
    tasks = [
        {"source_lang": "英语", "target_lang": "中文", "text": "Hello, world!"},
        {"source_lang": "中文", "target_lang": "英语", "text": "今天天气真好！"},
    ]

    for task in tasks:
        result = chain.invoke(task)
        print(f"{task['source_lang']}→{task['target_lang']}: {task['text']}")
        print(f"  译文: {result}")
    print()


# =============================================================================
# 示例 3: 可复用的 Chain 工厂函数
# =============================================================================

def reusable_chain_factory():
    """
    演示如何把 Chain 封装成函数，方便在多处复用。

    核心概念：
    - Chain 工厂函数 = 一个"生产 Chain 的函数"
    - 好处：一次编写，到处使用；修改只需改一处

    生活化比喻：
    直接创建 Chain = 每次想吃包子就自己和面、调馅、蒸
    工厂函数       = 写一个"做包子"的菜谱，想吃了照着做就行
    """
    print("=" * 60)
    print("示例 3: 可复用的 Chain 工厂函数")
    print("=" * 60)

    def create_translator(source_lang: str, target_lang: str):
        """
        创建一个特定语言方向的翻译 Chain。

        Args:
            source_lang: 源语言
            target_lang: 目标语言

        Returns:
            一个配置好的翻译 Chain，调用时只需传入 {"text": "..."}
        """
        prompt = PromptTemplate.from_template(f"""
你是一位专业翻译，请将文本从{source_lang}翻译成{target_lang}。
只输出翻译结果，不要解释。

原文：{{{{text}}}}

译文：""")
        model = ChatOllama(model="qwen3.5:2b")
        return prompt | model | StrOutputParser()

    # 创建两个不同方向的翻译 Chain
    en_to_zh = create_translator("英语", "中文")
    zh_to_en = create_translator("中文", "英语")

    # 使用时只需传 {text}
    print("英→中:", en_to_zh.invoke({"text": "Artificial intelligence is amazing!"}))
    print("中→英:", zh_to_en.invoke({"text": "中国的历史文化非常悠久。"}))
    print()


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  简单链 — LCEL Pipeline")
    print("  用 | 操作符串联 Prompt → Model → Parser")
    print("=" * 70 + "\n")

    print("【运行前检查】")
    print("  1. 已安装 Ollama 服务")
    print("  2. 已下载模型：ollama pull qwen3.5:2b")
    print()

    # ★ 按顺序运行：从基础到实用
    three_stage_pipeline()
    translation_chain()
    reusable_chain_factory()

    print("=" * 70)
    print("  接下来学习：sequential_chain.py（多步骤顺序链）")
    print("=" * 70 + "\n")
