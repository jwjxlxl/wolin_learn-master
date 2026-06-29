# =============================================================================
# 树状思维模式 — Tree of Thoughts (ToT)
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解 ToT 的核心思想：生成多条思路分支，评估后选出最佳
#   ✅ 实现 ToT 生成模式：同一问题生成 3 个不同思路，各自推理后选最优
#   ✅ 实现 ToT 投票模式：3 个推理路径 + 评估打分 + 投票选最优
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
from helpers import multi_candidate_generate


# =============================================================================
# 核心概念：Tree of Thoughts（树状思维）
# =============================================================================
"""
提出背景：Princeton 和 DeepMind 2023 年论文《Tree of Thoughts:
Deliberate Problem Solving with Large Language Models》。

核心思想：不是单线思维，而是生成多条思路分支，像树一样展开，
再通过评估机制选出最佳分支。

生活化比喻：
  CoT = 一条路走到底 → 走错就全错
  ToT = 同时探索 A/B/C 三条路 → 评估哪条最合理 → 选最优

适用场景：复杂规划、创意生成、解谜任务
"""


# =============================================================================
# 示例 1: ToT 生成模式 — 多分支探索 + 选优
# =============================================================================

def tot_generation():
    """
    ToT 生成模式：对同一个问题生成 3 个不同的思路分支，
    每个分支独立推理，最后评估选出最优。

    示例：创意写作 — 写一个关于 AI 改变教育的短故事开头
    """
    print(f"\n-- 示例 1: ToT 生成模式 — 多分支探索 + 选优")

    model = get_model()
    if model is None:
        print("  【跳过】请安装 Ollama 并下载模型：ollama pull qwen3.5:2b")
        return

    question = "写一个关于AI如何改变未来教育的短故事开头（100字以内）。"

    print(f"  题目：{question}")
    print()

    # --- 步骤 1：生成多个思路分支 ---
    print("  【步骤 1】生成 3 个不同的思路分支...")
    branches = multi_candidate_generate(model, question, n=3)

    for i, branch in enumerate(branches):
        print(f"\n  ── 分支 {i + 1} ──")
        # 缩进输出
        for line in branch.split("\n"):
            print(f"    {line}")
        print()

    # --- 步骤 2：评估每个分支 ---
    print("\n  【步骤 2】评估每个分支...")
    evaluate_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个写作评审专家。请从创意性、可读性、感染力三个维度评分（1-10分），并选出最佳。"),
        ("user", """请评估以下 3 个故事开头，分别打分并说明理由，最后选出最佳：

分支 1：
{branch1}

分支 2：
{branch2}

分支 3：
{branch3}

请按以下格式输出：
  分支 1 评分：X/10，理由：...
  分支 2 评分：X/10，理由：...
  分支 3 评分：X/10，理由：...
  最佳选择：分支 N"""),
    ])

    eval_response = (evaluate_prompt | model).invoke({
        "branch1": branches[0],
        "branch2": branches[1],
        "branch3": branches[2],
    })

    print(f"\n  评审意见：")
    for line in eval_response.content.split("\n"):
        print(f"    {line}")

    # --- 步骤 3：输出最优 ---
    print(f"\n  ✅ 已根据评审意见选出最佳分支")


# =============================================================================
# 示例 2: ToT 投票模式 — 多路径推理 + 投票选优
# =============================================================================

def tot_voting():
    """
    ToT 投票模式：对同一问题生成 3 个推理路径，
    让模型评估每个路径的逻辑质量，投票选最优。

    示例：规划题 — 一个水池有进水管和出水管，进水管每分钟进水 10 升，
    出水管每分钟出水 6 升。水池容量 200 升，初始为空，多久能满？
    """
    print(f"\n-- 示例 2: ToT 投票模式 — 多路径推理 + 投票选优")

    model = get_model()
    if model is None:
        print("  【跳过】请安装 Ollama 并下载模型：ollama pull qwen3.5:2b")
        return

    question = (
        "一个水池有进水管和出水管。进水管每分钟进水 10 升，"
        "出水管每分钟出水 6 升。水池容量 200 升，初始为空，"
        "多久能装满水池？请一步步推理。"
    )

    print(f"  题目：{question}")
    print()

    # --- 步骤 1：生成 3 个独立的推理路径 ---
    print("  【步骤 1】生成 3 个独立的推理路径...")
    reasoning_paths = multi_candidate_generate(model, question, n=3)

    for i, path in enumerate(reasoning_paths):
        print(f"\n  ── 推理路径 {i + 1} ──")
        for line in path.split("\n"):
            print(f"    {line}")
        print()

    # --- 步骤 2：交叉评估 ---
    print("\n  【步骤 2】交叉评估：让模型判断哪个推理路径最正确...")

    vote_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个数学题评审专家。请分析以下 3 个推理路径，判断哪个是正确的。"),
        ("user", """题目：{question}

推理路径 A：
{path_a}

推理路径 B：
{path_b}

推理路径 C：
{path_c}

请逐个分析每个路径：
  1. 检查推理步骤是否有逻辑错误
  2. 检查计算是否正确
  3. 给出最终判断：哪个路径是正确的？正确答案是什么？"""),
    ])

    vote_response = (vote_prompt | model).invoke({
        "question": question,
        "path_a": reasoning_paths[0],
        "path_b": reasoning_paths[1],
        "path_c": reasoning_paths[2],
    })

    print(f"\n  评审意见：")
    for line in vote_response.content.split("\n"):
        print(f"    {line}")

    print(f"\n  【对比】注意：ToT 的核心不是'多数投票'，而是"
          f"生成多个思路 → 评估 → 选出逻辑最严谨的那一个。")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  Tree of Thoughts（树状思维）模式")
    print("  生成多条思路分支，评估后选出最佳，适合复杂规划和创意任务")
    print("=" * 70 + "\n")

    tot_generation()
    # tot_voting()

    print("\n" + "=" * 70)
    print("  推理范式学习完毕！接下来进入 02_action/（行动范式）")
    print("=" * 70 + "\n")
