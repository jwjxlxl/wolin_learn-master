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

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from utils.model_utils import get_model, setup_encoding
setup_encoding()

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser


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

    model = get_model("qwen")

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
# 主程序入口
# =============================================================================

if __name__ == '__main__':

    translation_chain()
