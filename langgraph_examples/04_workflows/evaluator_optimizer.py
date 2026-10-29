"""评估器-优化器：生成、评估、按反馈重试，直到达标或达到上限。"""

import os
import sys
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from utils.model_utils import get_model


class CodeEvaluation(BaseModel):
    """模型给出的代码质量评分与反馈。"""

    score: int = Field(ge=1, le=10, description="代码质量评分，1 到 10 分")
    issues: str = Field(description="具体问题和改进建议")


class OptimizerState(TypedDict):
    task: str
    code: str
    score: int
    issues: str
    iteration: int
    max_iterations: int


def code_optimizer(provider: str = "ollama"):
    """每次评估后由代码判断是否继续，最多生成 max_iterations 次。"""
    print("\n-- 示例 3: 代码生成与评估循环")
    model = get_model(provider)
    evaluator = model.with_structured_output(CodeEvaluation)

    def generate(state: OptimizerState):
        if state["iteration"] == 0:
            prompt = f"写一个 Python 函数：{state['task']}。只输出完整代码。"
        else:
            prompt = (
                f"根据反馈改进以下 Python 函数，只输出完整代码。\n"
                f"任务：{state['task']}\n当前代码：\n{state['code']}\n"
                f"反馈：{state['issues']}"
            )
        response = model.invoke(prompt)
        return {"code": response.content, "iteration": state["iteration"] + 1}

    def evaluate(state: OptimizerState):
        evaluation = evaluator.invoke(
            f"按正确性、可读性和边界处理评估以下代码。\n"
            f"任务：{state['task']}\n代码：\n{state['code']}"
        )
        print(f"  第 {state['iteration']} 次：{evaluation.score}/10，反馈：{evaluation.issues}")
        return {"score": evaluation.score, "issues": evaluation.issues}

    def should_continue(state: OptimizerState):
        if state["score"] >= 7 or state["iteration"] >= state["max_iterations"]:
            return END
        return "generate"

    graph = (
        StateGraph(OptimizerState)
        .add_node("generate", generate)
        .add_node("evaluate", evaluate)
        .add_edge(START, "generate")
        .add_edge("generate", "evaluate")
        .add_conditional_edges("evaluate", should_continue, ["generate", END])
        .compile()
    )

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
