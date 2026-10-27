# =============================================================================
# 09_checkpointing — 03 改需求后恢复：update_state / 时间旅行 / fork
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 用 update_state 改写历史状态（模拟用户改需求）后继续执行
#   ✅ 用 get_state_history + checkpoint_id 回到过去某个检查点重跑（时间旅行）
#   ✅ 用 fork 从某个历史状态派生一个独立的新会话
#
# 运行前检查：
# 1. 已安装依赖：pip install langgraph langchain-core
# 2. 本示例为确定性演示（无需 API Key），直接运行：
#      python 03_time_travel_fork.py
#
# 场景：一个「计划」图会依次执行 A、B、C 三步。用户中途改了需求，希望
#       从某一步「改写并重跑」，或者「另起一个分支」试另一种方案。
# =============================================================================

import sys
import os
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class PlanState(TypedDict):
    """计划状态。plan 是普通字符串字段（赋值即覆盖，便于演示「改写」）。"""
    plan: str   # 计划文本
    step: int   # 当前步骤号


def step_a(state: PlanState) -> dict:
    print("  [步骤A] 执行 A")
    return {"plan": state["plan"] + "A; ", "step": 1}


def step_b(state: PlanState) -> dict:
    print("  [步骤B] 执行 B")
    return {"plan": state["plan"] + "B; ", "step": 2}


def step_c(state: PlanState) -> dict:
    print("  [步骤C] 执行 C")
    return {"plan": state["plan"] + "C; ", "step": 3}


def build_graph(checkpointer):
    """构建 A→B→C 顺序计划图。checkpointer 由调用方传入。"""
    return (
        StateGraph(PlanState)
        .add_node("step_a", step_a)
        .add_node("step_b", step_b)
        .add_node("step_c", step_c)
        .add_edge(START, "step_a")
        .add_edge("step_a", "step_b")
        .add_edge("step_b", "step_c")
        .add_edge("step_c", END)
        .compile(checkpointer=checkpointer)
    )


if __name__ == "__main__":
    print("=" * 60)
    print("  03 改需求后恢复：update_state / 时间旅行 / fork")
    print("=" * 60)

    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "plan-001"}}

    # 先完整跑一遍，作为基准
    result = graph.invoke({"plan": "", "step": 0}, config=config)
    print(f"\n  完整执行后：plan = {result['plan']!r}")

    # ---------------------------------------------------------------------
    # 示例 1：改需求 —— update_state 改写状态后继续
    # ---------------------------------------------------------------------
    print(f"\n-- 示例: 改需求 — update_state 改写状态后继续")
    # 用户改需求：把计划直接改写成「只做 Z」，并从 step_a 之后重新走 B、C
    graph.update_state(config, {"plan": "用户改需求：只做 Z; ", "step": 0}, as_node="step_a")
    result = graph.invoke(None, config=config)
    print(f"  改写后继续执行：plan = {result['plan']!r}")
    print("  说明：as_node='step_a' 表示这次改写相当于 step_a 产生的，")
    print("        于是从 step_a 的后续节点（step_b、step_c）继续执行。")

    # ---------------------------------------------------------------------
    # 示例 2：时间旅行 —— 回滚到历史 Checkpoint 重跑
    # ---------------------------------------------------------------------
    print(f"\n-- 示例: 时间旅行 — 回到历史 Checkpoint 重跑")
    # 重新跑一条干净流程，便于观察历史
    config2 = {"configurable": {"thread_id": "plan-002"}}
    graph.invoke({"plan": "", "step": 0}, config=config2)

    print("  历史检查点（从新到旧）：")
    history = list(graph.get_state_history(config2))
    for i, h in enumerate(history):
        cid = h.config["configurable"].get("checkpoint_id", "")
        print(f"    [{i}] step={h.metadata.get('step')}, next={h.next}, "
              f"plan={h.values.get('plan')!r}, cid={cid[:8]}…")

    # 找到「step_a 之后、step_b 之前」的检查点：其 next 是 ('step_b',)
    past = next(h for h in history if h.next == ("step_b",))
    past_cid = past.config["configurable"]["checkpoint_id"]
    print(f"  回滚到：step={past.metadata.get('step')}, plan={past.values.get('plan')!r}")

    # 从该检查点重跑（时间旅行）：只重跑 step_b、step_c，不会重复 step_a
    result = graph.invoke(None, {"configurable": {"thread_id": "plan-002", "checkpoint_id": past_cid}})
    print(f"  重跑后：plan = {result['plan']!r}")

    # ---------------------------------------------------------------------
    # 示例 3：fork —— 从历史状态派生独立分支（换 thread_id）
    # ---------------------------------------------------------------------
    print(f"\n-- 示例: fork — 从历史状态派生独立新会话")
    # 把「step_a 之后」的状态「种」进一个全新 thread_id，作为独立分支的起点
    fork_cfg = {"configurable": {"thread_id": "plan-fork"}}
    graph.update_state(fork_cfg, {"plan": past.values["plan"], "step": past.values["step"]}, as_node="step_a")
    fork_result = graph.invoke(None, config=fork_cfg)
    print(f"  fork 分支（plan-fork）：plan = {fork_result['plan']!r}")

    # 现在只改 fork 分支的需求，验证它与原 thread 互不影响
    graph.update_state(fork_cfg, {"plan": "分支方案：改为做 X; ", "step": 0}, as_node="step_a")
    fork_result = graph.invoke(None, config=fork_cfg)
    print(f"  改 fork 分支后：plan = {fork_result['plan']!r}")
    print(f"  原 thread（plan-002）不受影响：plan = {graph.get_state(config2).values.get('plan')!r}")
    print("  说明：fork = 把某时刻状态复制到新 thread_id，此后两个会话各自独立演进。")

    print("\n💡 小结：")
    print("  1. 用户「中断」后继续 → interrupt() + Command(resume=...)（见 01/02）。")
    print("  2. 用户「改需求」后重跑 → update_state 改写状态，或回滚到历史 checkpoint 重跑。")
    print("  3. 想「试试另一种方案」→ fork 一个独立分支，不动原会话。")
