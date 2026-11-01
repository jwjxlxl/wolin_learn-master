"""评估器-优化器（Evaluator-Optimizer）：生成 → 评估 → 按反馈重试，直到达标或达到上限。

一句话：**让 LLM 自己给自己当评委，写不好就按意见改稿，改到达标为止。**

  ✅ 适合：有明确评判标准、一次很难写对的任务（写代码、写文案、翻译润色）
  ❌ 不适合：没有客观标准的任务（评委自己都说不清好坏，循环就没有意义）

流程：带环 ≠ 死循环，靠条件边 + 次数上限收住

  START → generate → evaluate ─┬─ (达标 / 到上限) → END
                    ↑          │
                    └──────────┘ 不达标则回到 generate，带上 evaluate 写回的 issues

与 03_agent_loop 的 ReAct 对比：
  两者都是"LLM 循环"，但 ReAct 每轮调用的是**工具**（查外部事实），
  本模式每轮调用的是**评估器**（主观质量打分），反馈来自模型自身。
"""

import os
import sys
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from utils.model_utils import get_model


class CodeEvaluation(BaseModel):
    """评委的结构化输出：评分 + 改进建议，两者都要，缺了改稿无从下手。"""

    score: int = Field(ge=1, le=10, description="代码质量评分，1 到 10 分")
    issues: str = Field(description="具体问题和改进建议")


class OptimizerState(TypedDict):
    """循环所需的最小状态集：任务原文 + 三个本次产物 + 两个循环控制变量。"""

    task: str            # 原始需求，每轮都要带上，防止改着改着跑偏
    code: str            # 当前稿件，被反复覆盖更新
    score: int           # 当前评分，条件边据此判断是否收工
    issues: str          # 当前反馈，下一轮 generate 据此改稿
    iteration: int       # 已生成次数
    max_iterations: int  # 兜底上限，防止永远不达标时无限烧 token


def code_optimizer(provider: str = "ollama"):
    """每次评估后由代码判断是否继续，最多生成 max_iterations 次。"""
    print("\n-- 示例 3: 代码生成与评估循环")
    model = get_model(provider)
    # 同一个模型换副"面孔"当评委：强制输出 CodeEvaluation 结构，好拿分数
    evaluator = model.with_structured_output(CodeEvaluation)

    def generate(state: OptimizerState):
        """写稿/改稿：首轮从零写，之后带着上一轮的 issues 改。"""
        if state["iteration"] == 0:
            prompt = f"写一个 Python 函数：{state['task']}。只输出完整代码。"
        else:
            prompt = (
                f"根据反馈改进以下 Python 函数，只输出完整代码。\n"
                f"任务：{state['task']}\n当前代码：\n{state['code']}\n"
                f"反馈：{state['issues']}"
            )
        response = model.invoke(prompt)
        # iteration +1 是本循环的计数器，也是后面判断是否达到上限的依据
        return {"code": response.content, "iteration": state["iteration"] + 1}

    def evaluate(state: OptimizerState):
        """评委：只看代码不改进，把评分和意见写回状态，交给条件边裁决。"""
        evaluation = evaluator.invoke(
            f"按正确性、可读性和边界处理评估以下代码。\n"
            f"任务：{state['task']}\n代码：\n{state['code']}"
        )
        print(f"  第 {state['iteration']} 次：{evaluation.score}/10，反馈：{evaluation.issues}")
        return {"score": evaluation.score, "issues": evaluation.issues}

    def should_continue(state: OptimizerState):
        """循环出口：达标（>=7 分）或到 iteration 上限就收工，否则退回 generate 改。"""
        if state["score"] >= 7 or state["iteration"] >= state["max_iterations"]:
            return END
        return "generate"

    graph = (
        StateGraph(OptimizerState)
        .add_node("generate", generate)
        .add_node("evaluate", evaluate)
        .add_edge(START, "generate")
        .add_edge("generate", "evaluate")
        # 唯一的一条条件边撑起整个循环：回到 generate 就是环形边
        .add_conditional_edges("evaluate", should_continue, ["generate", END])
        .compile()
    )

    # 状态字段必须给全；max_iterations 是本demo唯一的防死循环保险丝
    result = graph.invoke({
        "task": "判断字符串是否为回文，忽略大小写和空格",
        "code": "",
        "score": 0,
        "issues": "",
        "iteration": 0,
        "max_iterations": 3,
    })
    print(f"  共生成 {result['iteration']} 次，最终评分 {result['score']}/10")
    print(result["code"])


if __name__ == "__main__":
    code_optimizer()
