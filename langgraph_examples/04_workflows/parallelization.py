"""并行化工作流（Parallelization）：Send 动态扇出 + reducer 扇入汇聚。

=============================================================================
一、这个模式解决什么问题？
=============================================================================
用户问"我的订单什么情况？"，需要同时去三个系统查：订单系统、物流系统、售后系统。

  ❌ 顺序执行（提示链）：先查订单 → 再查物流 → 再查售后
     总耗时 = 三次查询耗时之和，而且三者毫无依赖关系，白白串行等待。

  ✅ 并行化：同时派三个人分头去查，谁查完谁先回来，最后汇总成一份报告
     总耗时 ≈ 最慢的那一次查询。

一句话概括：**把互不依赖的子任务同时跑，最后把结果汇总。**（Fan-out 扇出 + Fan-in 扇入）

=============================================================================
二、什么时候用？
=============================================================================
  ✅ 适合：子任务之间**没有依赖**、可独立执行（查物流不需要先知道订单状态）
  ❌ 不适合：后一步依赖前一步的输出（那应该用 04_workflows 的提示链 / 提示链接模式）

=============================================================================
三、LangGraph 里靠两个机制实现
=============================================================================
  1. Send() —— 动态扇出
     普通 add_edge 是"固定边"，写死一条；条件边可以返回一个 Send 列表，
     LangGraph 会为列表中**每一项并行创建一个任务实例**：
     同一个节点函数被调用 N 次，每次拿到不同的私有输入。
     关键是 N 在运行时才知道（这里取决于 sources 列表里有几个数据源），
     所以用 Send，而不是写死 3 条边。

  2. reducer（operator.add）—— 扇入汇聚
     多个 Worker 并发出去后，都要往同一个字段 results 里写结果。
     如果没有 reducer，后写完的会覆盖先写完的，只剩一条。
     Annotated[list[str], operator.add] 让每次返回值**追加**进列表而非覆盖，
     天然支持并发写入。（和 03_agent_loop 里 add_messages 的思路完全一致）

=============================================================================
四、本文件的执行流程
=============================================================================
                    ┌──> lookup(查订单) ──┐
  START ─ dispatch ─┼──> lookup(查物流) ──┼──> summarize ──> END
     (Send 扇出)    └──> lookup(查售后) ──┘   (reducer 汇聚后汇总)

  dispatch   分发员：按 sources 列表生成 3 个 Send，不干活、只派活
  lookup     干活的：被调用 3 次，各自查自己那份数据
  summarize  汇总员：等所有 lookup 都结束后才执行，把 results 拼成最终答案

=============================================================================
五、两个 TypedDict 为什么要分开？
=============================================================================
  QueryState   主状态：整张图共享，含汇聚字段 results
  WorkerState  工人状态：Send 携带的**私有输入**，每个 Worker 只看到自己那一小份
  这样 lookup 节点天然隔离，不会误读别人的数据；它要交回结果，
  只需 return 一个属于 QueryState 的字段（results），LangGraph 会自动写回主状态。

注意事项：
  - Worker 之间不能互相通信，读到的共享状态是扇出那一刻的快照
  - 不要试图在 Worker 里读其它 Worker 的结果，那是 summarize 的活
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from langgraph_examples.utils import save_graph_png

# 模拟三个外部数据源（真实场景就是订单库 / 物流接口 / 售后接口）
ORDER_DATA = {
    "ORD-001": {
        "订单": "已发货",
        "物流": "预计明日送达",
        "售后": "签收后七天内可申请退换",
    }
}


class QueryState(TypedDict):
    """主状态：整张图共享，负责扇出配置和扇入汇聚。"""

    order_id: str   # 要查的订单号
    sources: list[str]  # 要查哪些数据源，决定扇出几个任务
    results: Annotated[list[str], operator.add]  # 汇聚字段：并发写入靠 reducer 追加
    summary: str    # 最终汇总结果


class WorkerState(TypedDict):
    """工人状态：Send 携带的私有输入，每个 Worker 只看到自己那一小份。"""

    order_id: str   # 从主状态透传过来的订单号
    source: str     # 我只负责查这一个数据源


def parallel_order_lookup():
    """为每个数据源创建一个 Send，所有 Worker 完成后再汇总。"""
    print("\n-- 示例 2: 并行查询订单信息")

    def dispatch(state: QueryState):
        """扇出节点：不干活，只按 sources 列表派活。

        返回 Send 列表是本模式的核心——LangGraph 会为列表中每一项并行创建一个
        lookup 任务实例，各自带着不同的 source 独立执行。
        例如："sources": ["订单", "物流"]
        [
            Send("lookup", {"order_id": "ORD-001", "source": "订单"}),
            Send("lookup", {"order_id": "ORD-001", "source": "物流"}),
            Send("lookup", {"order_id": "ORD-001", "source": "售后"}),
        ]
        """
        return [
            Send("lookup", {"order_id": state["order_id"], "source": source})
            for source in state["sources"]
        ]

    def lookup(state: WorkerState):
        """工人节点：被调用 N 次（N = sources 长度），每次只查自己那份数据。

        注意只返回"新增的那一条"，不读也不覆盖别人的结果，
        operator.add 会自动把它追加进 results 列表。
        """
        value = ORDER_DATA.get(state["order_id"], {}).get(state["source"], "暂无数据")
        return {"results": [f"{state['source']}：{value}"]}

    def summarize(state: QueryState):
        """汇聚节点：等所有 lookup 都结束后才执行（LangGraph 自动做同步屏障）。

        state["results"] 此刻已是被追加好的完整列表，排序保证输出稳定。
        """
        return {"summary": "\n".join(sorted(state["results"]))}

    graph = (
        StateGraph(QueryState)
        .add_node("lookup", lookup)
        .add_node("summarize", summarize)
        # 条件边返回 Send 列表 → 动态扇出；第三个参数仍需列出可能的节点名
        .add_conditional_edges(START, dispatch, ["lookup"])
        # 所有并行的 lookup 分支都结束后，才会走到 summarize（汇聚点）
        .add_edge("lookup", "summarize")
        .add_edge("summarize", END)
        .compile()
    )

    # 保存图为 PNG（复用 utils/graph_helpers.py 的 save_graph_png）
    save_graph_png(graph, "parallization_workflow.png")

    # 初始状态必须把所有字段都给全，尤其 results 要给空列表作为汇聚起点
    result = graph.invoke({
        "order_id": "ORD-001",
        "sources": ["订单", "物流", "售后"],
        "results": [],
        "summary": "",
    })
    print(result["summary"])


if __name__ == "__main__":
    parallel_order_lookup()
