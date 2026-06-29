# =============================================================================
# 反思与迭代优化 — Reflexion
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解 Reflexion 的核心思想：犯错后总结失败原因，带着反思重试
#   ✅ 实现代码优化版 Reflexion：生成代码 → 评估 → 反思 → 重新生成
#   ✅ 实现创意优化版 Reflexion：文字创作 → 评估 → 反思 → 重写
#   ✅ 理解 Reflexion 与 Evaluator-Optimizer 的区别
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
# 核心概念：Reflexion（反思与迭代优化）
# =============================================================================
"""
提出背景：2023 年论文《Reflexion: Language Agents with Verbal
Reinforcement Learning》。

核心思想：Agent 具备自我纠错的能力，犯错后会总结失败原因，
再带着反思尝试下一次。

生活化比喻：
  普通 Agent = 学生答题 → 答错 → 老师说错 → 结束
  Reflexion = 学生答题 → 答错 → 老师说错 → 学生反思"我哪里错了"
             → 带着反思再答一次 → 可能答对

与 Evaluator-Optimizer 的区别：
  Evaluator-Optimizer: 评估 → 给分 → 改进 → 循环
                      （侧重"打分"和"改进"，不一定解释原因）
  Reflexion:          评估 → 生成"反思"文本（哪里错了 + 怎么改）→ 带着反思重试
                      （侧重"自我诊断"，有明确的失败分析）

适用场景：代码生成（运行报错 → 读报错 → 反思 → 重试）、流程执行类场景
"""


# =============================================================================
# 示例 1: 代码 Reflexion — 生成代码 → 评估 → 反思 → 重试
# =============================================================================

def code_reflexion():
    """
    Reflexion 代码优化循环：
      1. 生成代码（首次可能含 bug）
      2. 评估代码质量/正确性
      3. 如果不合格，生成"反思"（具体哪里错了，如何改进）
      4. 带着反思重新生成代码
      5. 重复直到达标或达到最大轮次

    示例："写一个快速排序函数"
      首次可能生成有 bug 的版本（如边界处理错误）
      反思后修正
    """
    print(f"\n-- 示例 1: 代码 Reflexion — 生成 → 评估 → 反思 → 重试")

    model = get_model()
    if model is None:
        print("  【跳过】请安装 Ollama 并下载模型：ollama pull qwen3.5:2b")
        return

    task = "请用 Python 写一个快速排序函数，要求：能处理空列表、单个元素、已排序列表等边界情况。"
    max_rounds = 3
    reflection_history = []

    print(f"  任务：{task}")
    print()

    for round_num in range(max_rounds):
        print(f"  {'='*50}")
        print(f"  【第 {round_num + 1} 轮】")

        # --- 生成代码 ---
        context = ""
        if reflection_history:
            context = "\n之前的反思：\n" + "\n".join(reflection_history)

        generate_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个 Python 编程专家。请编写高质量的 Python 代码。如果有之前的反思，请特别关注反思中指出的问题。"),
            ("user", "任务：{task}{context}\n\n请写出代码："),
        ])

        code_response = (generate_prompt | model).invoke({
            "task": task,
            "context": context,
        })

        code = code_response.content
        print(f"\n  生成的代码：")
        for line in code.split("\n"):
            print(f"    {line}")

        # --- 评估代码 ---
        print(f"\n  【评估】")
        eval_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个代码评审专家。请从正确性、边界处理、代码风格三个维度评估以下 Python 代码，给出 1-10 的分数，并说明具体问题。"),
            ("user", """请评估以下 Python 代码：

{code}

请输出：
  分数：X/10
  问题：...（列出具体问题，如果没有就说"无明显问题"）"""),
        ])

        eval_response = (eval_prompt | model).invoke({"code": code})
        eval_text = eval_response.content
        for line in eval_text.split("\n"):
            print(f"    {line}")

        # --- 判断是否需要反思 ---
        # 检查是否包含高分（>=8）或"无明显问题"
        if "无明显问题" in eval_text or "无明显问题" in eval_text or ("9" in eval_text[:50]) or ("10" in eval_text[:50]):
            print(f"\n  ✅ 代码质量达标，无需继续优化。")
            break

        # --- 生成反思 ---
        print(f"\n  【反思】")
        reflect_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个善于自我反思的程序员。请分析代码评审中指出的问题，总结具体的改进方向。"),
            ("user", """以下是我的代码和评审意见：

代码：
{code}

评审意见：
{eval_text}

请反思：我具体哪里写错了？下次应该怎么做才能改进？请用简洁的要点列出。"""),
        ])

        reflection = (reflect_prompt | model).invoke({
            "code": code,
            "eval_text": eval_text,
        })

        reflection_text = reflection.content
        print(f"  反思内容：")
        for line in reflection_text.split("\n"):
            print(f"    {line}")

        reflection_history.append(f"第{round_num + 1}轮反思：{reflection_text}")

        if round_num < max_rounds - 1:
            print(f"\n  → 带着反思进入下一轮...\n")

    print(f"\n  【统计】共进行了 {round_num + 1} 轮")


# =============================================================================
# 示例 2: 创意 Reflexion — 写诗 → 评估 → 反思 → 重写
# =============================================================================

def creative_reflexion():
    """
    Reflexion 创意优化：
      文字创作场景下的反思迭代。

    示例："写一首关于 AI 的四行诗，要求押韵"
    """
    print(f"\n-- 示例 2: 创意 Reflexion — 写诗 → 评估 → 反思 → 重写")

    model = get_model()
    if model is None:
        print("  【跳过】请安装 Ollama 并下载模型：ollama pull qwen3.5:2b")
        return

    task = "写一首关于 AI 的四行诗，要求押韵。"
    max_rounds = 3
    reflection = ""

    print(f"  任务：{task}")
    print()

    for round_num in range(max_rounds):
        print(f"  {'='*50}")
        print(f"  【第 {round_num + 1} 轮】")

        # --- 生成 ---
        context = f"\n\n之前的反思：{reflection}" if reflection else ""
        gen_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个诗人。{context}"),
            ("user", "请写一首四行诗，主题：{task}"),
        ])

        poem = (gen_prompt | model).invoke({
            "context": "请根据之前的反思改进你的作品。" if reflection else "",
            "task": task,
        })

        poem_text = poem.content.strip()
        print(f"\n  作品：")
        for line in poem_text.split("\n"):
            print(f"    {line}")

        # --- 评估 ---
        print(f"\n  【评估】")
        eval_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个诗歌评审专家。请从主题表达、押韵、意境三个维度评分（1-10），并说明问题。"),
            ("user", """请评估以下诗歌：
{poem}

请输出：
  主题表达：X/10
  押韵：X/10
  意境：X/10
  问题：..."""),
        ])

        eval_response = (eval_prompt | model).invoke({"poem": poem_text})
        eval_text = eval_response.content
        for line in eval_text.split("\n"):
            print(f"    {line}")

        # --- 检查是否达标（押韵 >= 8）---
        if "押韵" in eval_text:
            # 检查押韵分数
            import re
            rhyme_match = re.search(r'押韵[:：]\s*(\d+)', eval_text)
            if rhyme_match and int(rhyme_match.group(1)) >= 8:
                print(f"\n  ✅ 押韵达标，作品完成！")
                break

        # --- 反思 ---
        print(f"\n  【反思】")
        reflect_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个善于反思的诗人。请分析评审意见，总结具体改进方向。"),
            ("user", """我的诗：
{poem}

评审意见：
{eval_text}

请反思：哪里需要改进？具体要怎么做？"""),
        ])

        reflect_response = (reflect_prompt | model).invoke({
            "poem": poem_text,
            "eval_text": eval_text,
        })
        reflection = reflect_response.content
        for line in reflection.split("\n"):
            print(f"    {line}")

        if round_num < max_rounds - 1:
            print(f"\n  → 带着反思重写...\n")

    print(f"\n  【统计】共进行了 {round_num + 1} 轮")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  Reflexion（反思与迭代优化）模式")
    print("  犯错 → 总结失败原因 → 带着反思重试")
    print("=" * 70 + "\n")

    code_reflexion()
    # creative_reflexion()

    print("\n" + "=" * 70)
    print("  接下来学习：role_playing.py（角色扮演式智能体 / 多智能体协作）")
    print("=" * 70 + "\n")
