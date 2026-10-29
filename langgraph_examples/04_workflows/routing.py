"""LLM 路由：模型识别意图，LangGraph 按预设分支处理。"""

import os
import sys
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from utils.model_utils import get_model


class RouteDecision(BaseModel):
    """客服问题分类结果。"""

    category: Literal["presales", "aftersales"] = Field(
        description="商品功能或价格属于 presales；订单、物流或退换属于 aftersales"
    )


class RoutingState(TypedDict):
    question: str
    category: str
    result: str


def routing_demo(provider: str = "ollama"):
    """与规则分支相比，只把分类步骤换成 LLM，条件边写法不变。"""
    print("\n-- 示例 1: 客服问题智能路由")
    classifier = get_model(provider).with_structured_output(RouteDecision)

    def classify(state: RoutingState):
        decision = classifier.invoke(f"请分类这条客服问题：{state['question']}")
        return {"category": decision.category}

    def presales(state: RoutingState):
        return {"result": f"已转售前顾问：{state['question']}"}

    def aftersales(state: RoutingState):
        return {"result": f"已转售后专员：{state['question']}"}

    def choose_branch(state: RoutingState):
        return state["category"]

    graph = (
        StateGraph(RoutingState)
        .add_node("classify", classify)
        .add_node("presales", presales)
        .add_node("aftersales", aftersales)
        .add_edge(START, "classify")
        .add_conditional_edges(
            "classify", choose_branch,
            {"presales": "presales", "aftersales": "aftersales"},
        )
        .add_edge("presales", END)
        .add_edge("aftersales", END)
        .compile()
    )

    questions = [
        "这款耳机支持主动降噪吗？",
        "收到耳机后左耳没有声音，如何退换？",
    ]
    for question in questions:
        result = graph.invoke({"question": question, "category": "", "result": ""})
        print(f"  {question} → {result['category']} → {result['result']}")


if __name__ == "__main__":
    routing_demo()
