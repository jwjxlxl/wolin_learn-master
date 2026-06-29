# =============================================================================
# 思维链模式 — Chain of Thought (CoT)
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解 CoT 的核心思想：让模型在回答前把推理过程一步步写出来
#   ✅ 区分 Zero-shot CoT（提示词触发）与 Few-shot CoT（示例触发）
#   ✅ 对比直接回答与 CoT 回答的质量差异
#
# 运行前检查：
# 1. 已安装 Ollama 并下载模型：ollama pull qwen3.5:2b
# =============================================================================

import sys
import os
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 确保项目根目录和 agent/ 目录可导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from langchain_core.prompts import ChatPromptTemplate
from utils.model_utils import get_model


# =============================================================================
# 核心概念：Chain of Thought（思维链）
# =============================================================================
"""
提出背景：Google Research 2022 年论文《Chain-of-Thought Prompting Elicits
Reasoning in Large Language Models》。

核心思想：让模型在回答前，把推理过程一步步写出来。不是一口气报出答案，
而是把整个推理过程展示出来。

生活化比喻：
  直接回答 = 考试时只写答案，不写步骤 → 容易算错
  CoT = 考试时一步步写解题过程 → 更稳健，更容易发现错误

适用场景：逻辑推理、数值计算、逐步分析类问题
"""


# =============================================================================
# 示例 1: 直接回答 vs Zero-shot CoT 对比
# =============================================================================

def direct_vs_cot():
    """
    对比直接回答与 Zero-shot CoT 的差异。

    Zero-shot CoT 技巧：在 prompt 中追加 "请一步步思考" 或 "Let's think step by step"，
    无需任何示例，模型自动进入推理模式。

    示例题目：
      小王比小李大 3 岁，小张的年龄是小李的两倍。
      如果三个人的年龄加起来是 41 岁，问小王多大？

    正确答案：小李 10 岁，小王 13 岁，小张 20 岁 → 小王 13 岁
    """
    print(f"\n-- 示例 1: 直接回答 vs Zero-shot CoT 对比")

    model = get_model()
    if model is None:
        print("  【跳过】请安装 Ollama 并下载模型：ollama pull qwen3.5:2b")
        return

    question = (
        "小王比小李大3岁，小张的年龄是小李的两倍。"
        "如果三个人的年龄加起来是41岁，问小王多大？"
    )

    # --- 方式 1：直接提问 ---
    print("\n  【方式 1：直接提问（无 CoT）】")
    direct_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个回答问题的助手。请直接给出答案。"),
        ("user", question),
    ])
    chain = direct_prompt | model
    direct_response = chain.invoke({})
    print(f"  问题：{question}")
    print(f"  回答：{direct_response.content}")

    # --- 方式 2：Zero-shot CoT ---
    print("\n  【方式 2：Zero-shot CoT（添加'请一步步思考'）】")
    cot_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个回答问题的助手。请先一步步地推理，再给出最终答案。"),
        ("user", question),
    ])
    chain = cot_prompt | model
    cot_response = chain.invoke({})
    print(f"  问题：{question}")
    print(f"  回答：{cot_response.content}")

    print(f"\n  【对比】CoT 方式是否更接近正确答案？观察推理过程的完整性。")


# =============================================================================
# 示例 2: Few-shot CoT — 示例触发的思维链
# =============================================================================

def few_shot_cot():
    """
    Few-shot CoT：提供 2 个带推理步骤的示例，让模型模仿。

    相比 Zero-shot CoT，Few-shot 通过示例明确告诉模型"我期望你这样推理"，
    效果通常更好。

    示例题目：逻辑推理
      前提1：所有猫都怕水
      前提2：咪咪是一只猫
      结论：咪咪怕水吗？
    """
    print(f"\n-- 示例 2: Few-shot CoT — 示例触发的思维链")

    model = get_model()
    if model is None:
        print("  【跳过】请安装 Ollama 并下载模型：ollama pull qwen3.5:2b")
        return

    # 构建 Few-shot CoT prompt
    # 注意：Few-shot 通过示例展示推理格式，模型会模仿这种推理方式
    prompt_text = """你是一个逻辑推理助手。请按照以下示例的方式，一步步推理后给出答案。

示例 1：
  问：小明有 5 个苹果，他吃了 2 个，又买了 3 个，现在有几个？
  推理过程：
    1. 小明最初有 5 个苹果
    2. 吃了 2 个 → 5 - 2 = 3 个
    3. 又买了 3 个 → 3 + 3 = 6 个
  答：小明现在有 6 个苹果。

示例 2：
  问：所有鸟都会飞。企鹅是鸟。企鹅会飞吗？
  推理过程：
    1. 前提1说"所有鸟都会飞"
    2. 企鹅是鸟 → 根据前提1，企鹅应该会飞
    3. 但现实中企鹅不会飞，说明前提1不完全准确
    4. 如果严格按照前提1推理 → 企鹅会飞
  答：根据给定前提，企鹅会飞。（虽然与事实不符，但推理成立）

现在请推理：
  问：{}"""

    question = "前提1：所有猫都怕水。前提2：咪咪是一只猫。咪咪怕水吗？"

    print(f"  问题：{question}")
    print(f"  推理过程：")

    full_prompt = prompt_text.format(question)
    response = model.invoke(full_prompt)
    # 缩进输出
    for line in response.content.split("\n"):
        print(f"    {line}")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  Chain of Thought（思维链）模式")
    print("  让模型一步步写出推理过程，提升逻辑推理和数学计算的稳健性")
    print("=" * 70 + "\n")

    direct_vs_cot()
    few_shot_cot()

    print("\n" + "=" * 70)
    print("  接下来学习：self_ask.py（Self-Ask 自问自答模式）")
    print("=" * 70 + "\n")
