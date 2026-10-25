# =============================================================================
# Prompt 模板 — 把重复的提示词变成可复用的"填空题"
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 使用 PromptTemplate 创建可复用的提示词模板
#   ✅ 理解模板的用途：复用、一致性、防注入
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from utils.model_utils import get_model, setup_encoding
setup_encoding()

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


# =============================================================================
# 示例 1: 最简单的模板 — from_template() + format()
# =============================================================================

def simplest_template():
    """
    PromptTemplate 的最简用法。

    {变量名} 是占位符，format() 方法填入实际值。
    """
    print(f"\n-- 示例 1: 最简单的模板")

    # 创建模板 — {language} 和 {concept} 是占位符
    prompt = PromptTemplate.from_template("请用{language}解释{concept}。")

    # 填入实际值
    formatted = prompt.format(language="小学生能懂的语言", concept="机器学习")
    print(f"填好的 Prompt: {formatted}")

    # 调用模型
    model = ChatOllama(model="qwen3.5:2b")
    response = model.invoke(formatted)

    print(f"回复: {response.content}")


# =============================================================================
# 示例 2: 实用场景 — 翻译助手模板
# =============================================================================

def translation_template():
    """
    用模板做一个可复用的翻译助手。

    同一个模板，填入不同的 {source_lang} 和 {target_lang}，
    就能完成不同方向的翻译。
    """
    print(f"\n-- 示例 2: 翻译助手模板")

    prompt = PromptTemplate.from_template("""
你是一位专业翻译，请将以下文本从{source_lang}翻译成{target_lang}。
只输出翻译结果，不要解释。

原文: {text}
译文:""")

    model = get_model("qwen")

    # 英译中
    r = model.invoke(prompt.format(source_lang="英语", target_lang="中文",
                   text="Artificial intelligence will change the world."))
    print(f"英→中: {r.content}")

    # 中译英
    r = model.invoke(prompt.format(source_lang="中文", target_lang="英语",
                   text="今天天气真好，适合出去散步。"))
    print(f"中→英: {r.content}")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n>>> 03_prompt/prompt_template — Prompt 模板基础\n")

    translation_template()
