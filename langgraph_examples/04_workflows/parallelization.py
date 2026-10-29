"""Send 动态扇出，使用 reducer 汇聚多个独立查询的结果。"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


ORDER_DATA = {
    "ORD-001": {
        "订单": "已发货",
        "物流": "预计明日送达",
        "售后": "签收后七天内可申请退换",
    }
}


class QueryState(TypedDict):
    order_id: str
    sources: list[str]
    results: Annotated[list[str], operator.add]
    summary: str


class WorkerState(TypedDict):
    order_id: str
    source: str


def parallel_order_lookup():
    """为每个数据源创建一个 Send，所有 Worker 完成后再汇总。"""
    print("\n-- 示例 2: 并行查询订单信息")

    def dispatch(state: QueryState):
        return [
            Send("lookup", {"order_id": state["order_id"], "source": source})
            for source in state["sources"]
        ]

    def lookup(state: WorkerState):
        value = ORDER_DATA.get(state["order_id"], {}).get(state["source"], "暂无数据")
        return {"results": [f"{state['source']}：{value}"]}

    def summarize(state: QueryState):
        return {"summary": "\n".join(sorted(state["results"]))}

    graph = (
        StateGraph(QueryState)
        .add_node("lookup", lookup)
        .add_node("summarize", summarize)
        .add_conditional_edges(START, dispatch, ["lookup"])
        .add_edge("lookup", "summarize")
        .add_edge("summarize", END)
        .compile()
    )

    result = graph.invoke({
        "order_id": "ORD-001",
        "sources": ["订单", "物流", "售后"],
        "results": [],
        "summary": "",
    })
    print(result["summary"])


if __name__ == "__main__":
    parallel_order_lookup()
