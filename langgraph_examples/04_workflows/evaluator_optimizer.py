# =============================================================================
# 评估器-优化器（Evaluator-Optimizer）— 生成 → 评估 → 不满意则重试
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解评估器-优化器循环：生成 → 评估 → 不满意则改进 → 再评估
#   ✅ 使用结构化输出做自动评估（打分 + 反馈）
#   ✅ 实现带退出保护的循环（最大迭代次数 + 质量阈值）
#
# 运行前检查：
# 1. 已安装依赖：pip install langgraph langchain-core langchain-ollama pydantic
# 2. 已安装 Ollama 并下载模型：ollama pull qwen3.5:2b
# =============================================================================

import sys
import os
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 确保项目根目录可导入（用于 utils/ 共享模块）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from typing import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from utils.model_utils import get_model


# =============================================================================
# 核心概念：评估器-优化器模式
# =============================================================================
"""
什么是评估器-优化器模式？

  这是一个经典的迭代改进模式：

    START → generate（生成初稿）
                ↓
          evaluate（评估质量）
                ↓
    (质量达标?) ──是──→ END（输出终稿）
         │
        否
         ↓
    optimize（根据反馈改进）
         │
         └──→ 回到 generate（重新生成）


  和提示链的区别：
    提示链：生成 → 评估 → 改进 → 润色（确定性路径，只走一次）
    评估器-优化器：生成 → 评估 → 改进 → 生成 → 评估 → ...（循环直到达标）

  生活化比喻：
    提示链          = 写作文 → 老师批改 → 修改 → 誊写（一次通过）
    评估器-优化器    = 写作文 → 老师批改 → 修改 → 老师再批改 → 再修改...（反复打磨）

  关键设计：
    - 必须设置最大迭代次数（防止无限循环）
    - 每次评估给出具体反馈（不只是"不好"，要说"哪里不好"）
    - 退出条件：分数达标 OR 达到最大次数
"""


# =============================================================================
# 示例 1: 代码优化器 — 写 Python 函数 → 评估 → 改进
# =============================================================================

def code_optimizer():
    """
    评估器-优化器实战：让 AI 写一个 Python 函数，然后自动评估和改进。

    循环：generate → evaluate → (分数 < 7 且 未超过 3 次?) → optimize → 回到 generate
                              ↘ (分数 >= 7 或 超过 3 次) → END
    """
    print(f"\n-- 示例: 评估器-优化器 — 代码优化循环")

    model = get_model()
    if model is None:
        print("  【跳过】请安装 Ollama 并下载模型：ollama pull qwen3.5:2b")
        return

    # 1. 结构化输出：评估器的打分格式
    class CodeEvaluation(BaseModel):
        """代码评估结果"""
        score: int = Field(description="1-10 分，10 分为满分")
        is_good_enough: bool = Field(description="分数 >= 7 则为 true")
        issues: str = Field(description="具体问题和改进建议")

    evaluator = model.with_structured_output(CodeEvaluation)

    # 2. 定义状态
    class OptimizerState(TypedDict):
        task: str              # 编程任务描述
        code: str              # 当前代码
        score: int             # 当前评分
        issues: str            # 当前问题
        iteration: int         # 当前迭代次数
        max_iterations: int    # 最大迭代次数
        history: str           # 历史记录（用于追踪改进过程）

    # 3. 定义节点
    def generate(state: OptimizerState):
        """生成节点：写代码（首次）或根据反馈改进（后续）"""
        iteration = state["iteration"]
        if iteration == 0:
            print(f"  [节点: generate] 首次生成代码（任务: {state['task']}）")
            prompt = f"写一个 Python 函数：{state['task']}。只输出代码和简要注释。"
        else:
            print(f"  [节点: generate] 第 {iteration} 次改进（上次评分: {state['score']}/10）")
            prompt = (
                f"改进以下代码，解决评估中提到的问题：\n\n"
                f"任务：{state['task']}\n"
                f"当前代码：\n{state['code']}\n"
                f"上次评分：{state['score']}/10\n"
                f"问题：{state['issues']}\n\n"
                f"请只输出改进后的完整代码。"
            )

        response = model.invoke(prompt)
        new_code = response.content
        return {
            "code": new_code,
            "iteration": iteration + 1,
            "history": state["history"] + f"\n[迭代 {iteration + 1}] 评分={state['score']}/10" if iteration > 0 else "",
        }

    def evaluate(state: OptimizerState):
        """评估节点：用结构化输出打分 + 反馈"""
        print(f"  [节点: evaluate] 评估代码质量...")
        evaluation: CodeEvaluation = evaluator.invoke(
            f"评估以下代码的质量（1-10分），关注：正确性、可读性、效率、边界处理。\n\n"
            f"任务：{state['task']}\n"
            f"```python\n{state['code']}\n```"
        )
        print(f"    评分: {evaluation.score}/10 — {'达标' if evaluation.is_good_enough else '需改进'}")
        print(f"    问题: {evaluation.issues[:60]}...")
        return {
            "score": evaluation.score,
            "is_good_enough": evaluation.is_good_enough,
            "issues": evaluation.issues,
        }

    def optimize(state: OptimizerState):
        """优化节点：根据评估反馈改进代码
        注意：这里不直接改代码，而是把反馈信息传递回 generate 节点
        """
        print(f"  [节点: optimize] 整理反馈 → 准备重新生成")
        return {}  # generate 节点会读取 state 中的 issues

    def should_continue(state: OptimizerState):
        """路由函数：决定继续循环还是结束"""
        if state.get("is_good_enough"):
            print(f"  [路由] 质量达标 (评分 {state['score']}/10) → 结束")
            return END
        if state["iteration"] >= state["max_iterations"]:
            print(f"  [路由] 已达最大迭代次数 {state['max_iterations']} → 结束")
            return END
        print(f"  [路由] 评分 {state['score']}/10 未达标 → 继续优化（第 {state['iteration'] + 1} 次）")
        return "optimize"

    # 4. 构建图
    graph = (
        StateGraph(OptimizerState)
        .add_node("generate", generate)
        .add_node("evaluate", evaluate)
        .add_node("optimize", optimize)
        .add_edge(START, "generate")
        .add_edge("generate", "evaluate")
        .add_conditional_edges("evaluate", should_continue, ["optimize", END])
        .add_edge("optimize", "generate")  # ★ 循环的关键：优化后回到生成
        .compile()
    )

    # 5. 测试
    result = graph.invoke({
        "task": "判断一个字符串是否是回文（忽略大小写和空格）",
        "code": "",
        "score": 0,
        "issues": "",
        "iteration": 0,
        "max_iterations": 3,
        "history": "",
    })

    print(f"\n  === 最终结果 ===")
    print(f"  【迭代次数】{result['iteration']}")
    print(f"  【最终评分】{result['score']}/10")
    print(f"  【最终代码】\n{result['code']}")
    print()


# =============================================================================
# 示例 2: 纯逻辑演示 — 无需 LLM 也能理解的评估器-优化器
# =============================================================================

def evaluator_optimizer_no_llm():
    """
    不依赖 LLM 的评估器-优化器演示。

    用简单规则模拟"评估"和"优化"，帮助学生理解循环结构本身。
    场景：猜数字 — 不断调整猜测直到猜对。
    """
    print(f"\n-- 示例 2: 纯逻辑演示 — 猜数字游戏")

    class GuessState(TypedDict):
        target: int           # 目标数字
        guess: int            # 当前猜测
        attempts: int         # 尝试次数
        feedback: str         # 反馈（太高/太低/正确）
        max_attempts: int     # 最大尝试次数

    def make_guess(state: GuessState):
        """生成猜测：首次随机，后续根据反馈调整"""
        if state["attempts"] == 0:
            # 首次：取中间值
            guess = 50
        else:
            # 二分搜索：根据反馈调整
            if "太高" in state["feedback"]:
                guess = state["guess"] - max(1, (100 - state["guess"]) // 2)
            else:
                guess = state["guess"] + max(1, state["guess"] // 2)

        print(f"  [节点: make_guess] 猜测: {guess}")
        return {"guess": guess, "attempts": state["attempts"] + 1}

    def evaluate_guess(state: GuessState):
        """评估：对比猜测和目标"""
        if state["guess"] == state["target"]:
            feedback = f"正确！答案就是 {state['target']}"
        elif state["guess"] > state["target"]:
            feedback = "太高了，往下猜"
        else:
            feedback = "太低了，往上猜"
        print(f"  [节点: evaluate_guess] {feedback}")
        return {"feedback": feedback}

    def should_continue(state: GuessState):
        """路由：猜对或超过次数则结束，否则继续"""
        if state["guess"] == state["target"]:
            print(f"  [路由] 猜对了！→ 结束")
            return END
        if state["attempts"] >= state["max_attempts"]:
            print(f"  [路由] 达到最大尝试次数 → 结束")
            return END
        print(f"  [路由] 没猜对 → 继续（第 {state['attempts'] + 1} 次）")
        return "make_guess"

    # 构建图（循环结构：make_guess → evaluate_guess → 回到 make_guess）
    graph = (
        StateGraph(GuessState)
        .add_node("make_guess", make_guess)
        .add_node("evaluate_guess", evaluate_guess)
        .add_edge(START, "make_guess")
        .add_edge("make_guess", "evaluate_guess")
        .add_conditional_edges("evaluate_guess", should_continue, ["make_guess", END])
        .compile()
    )

    # 测试
    import random
    target = random.randint(1, 100)
    print(f"  【目标数字（隐藏）】{target}")
    result = graph.invoke({
        "target": target,
        "guess": 0,
        "attempts": 0,
        "feedback": "",
        "max_attempts": 10,
    })
    print(f"\n  【结果】{result['feedback']}（共尝试 {result['attempts']} 次）")
    print()


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  评估器-优化器（Evaluator-Optimizer）")
    print("  生成 → 评估 → 不满意则改进 → 再评估 → ...")
    print("=" * 70 + "\n")

    print("【运行前检查】")
    print("  1. 已安装依赖：pip install langgraph langchain-core langchain-ollama pydantic")
    print("  2. 示例 2 无需 LLM（纯逻辑演示）")
    print()

    # 示例 2 不需要 LLM
    evaluator_optimizer_no_llm()

    # 示例 1 需要 LLM
    code_optimizer()

    print("=" * 70)
    print("  接下来学习：parallelization.py（并行化 — Send API）")
    print("=" * 70 + "\n")
