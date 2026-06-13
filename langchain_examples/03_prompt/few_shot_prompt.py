# =============================================================================
# Few-Shot Prompt — 给 AI 几个例子，它就能模仿格式回答
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解 Few-Shot Learning 的核心思想："给例子→模仿"
#   ✅ 手写 Few-Shot 模板做情感分类
#   ✅ 使用 LangChain 的 FewShotPromptTemplate 类更优雅地管理示例
# =============================================================================

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)


"""
什么是 Few-Shot Prompting？

  不用 Few-Shot:
    你: "把下面的话分类为正面或负面"
    AI: "好的，这句话是..."（可能不理解你想要的格式）

  用 Few-Shot:
    你: "例子1: '今天天气真好' → 正面"
        "例子2: '我心情不好' → 负面"
        "现在分类: '这个餐厅很难吃'"
    AI: "负面"（明白了格式和要求）

  生活化比喻: Few-Shot = 教小孩做题
    "看，这道题是这样做的..."（给例子）
    "现在你来做这道..."（让小孩模仿）
"""

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


# =============================================================================
# 示例 1: 手写 Few-Shot — 最简单的"给例子"
# =============================================================================

def simplest_few_shot():
    """
    直接在 Prompt 里写几个"问→答"的例子，让模型模仿格式。

    这是 Few-Shot 最基本的形式——不需要任何特殊类，纯 Prompt 技巧。
    """
    print(f"\n-- 示例 1: 手写 Few-Shot — 首都问答")

    prompt = PromptTemplate.from_template("""
请模仿下面的例子回答问题。

例子:
  问: 法国首都是哪里？  答: 巴黎
  问: 美国首都是哪里？  答: 华盛顿
  问: 中国首都是哪里？  答: 北京

问题: 日本首都是哪里？
答:""")

    model = ChatOllama(model="qwen3.5:2b")
    response = model.invoke(prompt.format())
    print(f"回复: {response.content}")


# =============================================================================
# 示例 2: 实用场景 — 情感分类
# =============================================================================

def sentiment_classification():
    """
    用 Few-Shot 让模型做情感分类——不需要训练模型，几个例子就搞定。

    这种模式的威力在于：你可以随时调整"例子"来改变分类标准，
    比如增加"中性"类别或"愤怒"类别，只需加一个例子。
    """
    print(f"\n-- 示例 2: 情感分类（Few-Shot）")

    prompt = PromptTemplate.from_template("""
请判断以下评论的情感倾向（正面/负面）。

例子:
  评论: "这家餐厅的食物很好吃！"        情感: 正面
  评论: "等了一个小时才上菜，味道还很咸。" 情感: 负面
  评论: "电影特效很震撼，剧情也很感人。"    情感: 正面

评论: "{review}"
情感:""")

    model = ChatOllama(model="qwen3.5:2b")

    for review in ["这个产品太棒了，我非常喜欢！", "完全不值这个价，浪费钱。"]:
        r = model.invoke(prompt.format(review=review))
        print(f"  \"{review}\" → {r.content}")


# =============================================================================
# 示例 3: FewShotPromptTemplate — LangChain 封装类
# =============================================================================

def using_langchain_fewshot():
    """
    当示例很多时，手写 Prompt 会很乱。LangChain 提供了
    FewShotPromptTemplate 类来优雅地管理示例列表。

    三步走：
    1. 准备示例列表 [{"question": ..., "answer": ...}, ...]
    2. 创建 example_prompt 定义每个示例的格式
    3. 创建 FewShotPromptTemplate 自动拼接
    """
    from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

    print(f"\n-- 示例 3: FewShotPromptTemplate（LangChain 封装）")

    # 第 1 步: 准备示例
    examples = [
        {"question": "法国首都是哪里？", "answer": "巴黎"},
        {"question": "美国首都是哪里？", "answer": "华盛顿"},
        {"question": "中国首都是哪里？", "answer": "北京"},
    ]

    # 第 2 步: 定义每个示例的显示格式
    example_prompt = PromptTemplate.from_template("问: {question}\n答: {answer}")

    # 第 3 步: 创建 FewShotPromptTemplate
    few_shot = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="请模仿下面的例子回答问题:",
        suffix="问题: {question}\n答:",
        input_variables=["question"],
    )

    model = ChatOllama(model="qwen3.5:2b")
    response = model.invoke(few_shot.format(question="日本首都是哪里？"))
    print(f"回复: {response.content}")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n>>> 03_prompt/few_shot — Few-Shot 示例\n")

    simplest_few_shot()
    sentiment_classification()
    using_langchain_fewshot()

    # 接下来学习: 04_output_parser/string_parser.py（输出解析）
