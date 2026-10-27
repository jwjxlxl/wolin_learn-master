# =============================================================================
# 09_checkpointing — 01 断点续跑与 Checkpoint 机制（Checkpointer + thread_id）
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解 Checkpointer 是「断点续跑」的前提，会挂载它并指定 thread_id
#   ✅ 说清 Checkpoint 在什么时机生成、里面存了什么内容
#   ✅ 用 interrupt() + Command(resume=...) 实现「暂停 → 恢复」
#
# 运行前检查：
# 1. 已安装依赖：pip install langgraph langchain-core
# 2. 本示例为确定性演示（无需 API Key），直接运行：
#      python 01_checkpoint_basics.py
#
# 说明：本文件使用 InMemorySaver（内存检查点）。它是 LangGraph 1.x 推荐的内存
#       检查点实现，MemorySaver 是其旧名（08_student_helper 里仍用旧名，两者等价）。
#       真正的「落盘持久化 + 进程崩溃恢复」见同目录 02_persistent_resume.py。
# =============================================================================

import sys
import os
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 确保项目根目录可导入（用于 utils/ 共享模块）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from langgraph_examples.utils.graph_helpers import save_graph_png


# =============================================================================
# State 定义：一笔订单的共享状态
# =============================================================================

class OrderState(TypedDict):
    price: float                          # 订单金额
    stage: str                            # 当前阶段说明
    notes: Annotated[list, operator.add]  # 处理日志（operator.add 实现累加，而非覆盖）


# =============================================================================
# Node 定义：打折 → 运费 → 人工确认（暂停）→ 出结果
# =============================================================================

def apply_discount(state: OrderState) -> dict:
    """节点 1：打折 —— 金额打 9 折。"""
    new_price = round(state["price"] * 0.9, 2)
    print(f"  [打折] 金额 {state['price']} → {new_price}")
    return {"price": new_price, "notes": [f"已打 9 折，金额变为 {new_price}"]}


def add_shipping(state: OrderState) -> dict:
    """节点 2：运费 —— 加 8 元运费。"""
    new_price = round(state["price"] + 8, 2)
    print(f"  [运费] 金额 {state['price']} → {new_price}")
    return {"price": new_price, "notes": [f"加运费 8 元，金额变为 {new_price}"]}


def confirm_node(state: OrderState) -> dict:
    """节点 3：人工确认 —— interrupt() 在此暂停，等待外部确认后继续。"""
    confirm = interrupt({
        "type": "confirm",
        "current_price": state["price"],
        "hint": "确认支付请输入 yes，取消输入 no",
    })
    answer = confirm.get("answer", "") if isinstance(confirm, dict) else ""
    ok = str(answer).lower() == "yes"
    print(f"  [确认] 用户选择：{answer}")
    return {"stage": "已确认" if ok else "已取消",
            "notes": ["订单已确认"] if ok else ["订单已取消"]}


def format_result(state: OrderState) -> dict:
    """节点 4：出结果 —— 汇总最终应付金额。"""
    print(f"  [出结果] 阶段：{state['stage']}")
    return {"notes": [f"最终应付 {state['price']} 元"]}


def build_graph(checkpointer):
    """构建订单处理图。checkpointer 由调用方传入（本文件传 InMemorySaver）。"""
    return (
        StateGraph(OrderState)
        .add_node("apply_discount", apply_discount)
        .add_node("add_shipping", add_shipping)
        .add_node("confirm", confirm_node)
        .add_node("format_result", format_result)
        .add_edge(START, "apply_discount")
        .add_edge("apply_discount", "add_shipping")
        .add_edge("add_shipping", "confirm")
        .add_edge("confirm", "format_result")
        .add_edge("format_result", END)
        .compile(checkpointer=checkpointer)
    )


if __name__ == "__main__":
    print("=" * 60)
    print("  01 断点续跑与 Checkpoint 机制")
    print("=" * 60)

    # 编译：必须挂 Checkpointer，图才有「记忆」，才能断点续跑
    graph = build_graph(checkpointer=InMemorySaver())
    save_graph_png(graph, "09_checkpoint_basics.png")

    # thread_id 标识一次会话；同一 thread_id 的多次 invoke 共享同一份状态
    config = {"configurable": {"thread_id": "order-001"}}

    # ---------------------------------------------------------------------
    # 示例 1：挂 Checkpointer + thread_id，让图「有记忆」
    # ---------------------------------------------------------------------
    print(f"\n-- 示例: 挂 Checkpointer + thread_id — 让图可恢复")
    # 首次 invoke：跑到 confirm 的 interrupt() 处自动暂停
    result = graph.invoke({"price": 100, "stage": "", "notes": []}, config=config)
    print(f"  暂停时 price = {result.get('price')}，stage = {result.get('stage')!r}")

    # 用 get_state 查看「当前停在哪个节点」
    snapshot = graph.get_state(config)
    print(f"  当前 next（待执行节点）：{snapshot.next}")
    print(f"  当前 state 值：{snapshot.values}")

    # ---------------------------------------------------------------------
    # 示例 2：Checkpoint 在什么时机生成、存了什么内容
    # ---------------------------------------------------------------------
    print(f"\n-- 示例: Checkpoint 的生成时机与存储内容")
    # get_state_history 从最新到最旧列出该 thread 的所有检查点
    print("  该 thread 的完整检查点历史（从新到旧）：")
    for i, hist in enumerate(graph.get_state_history(config)):
        cid = hist.config["configurable"].get("checkpoint_id", "")
        meta = hist.metadata
        print(f"  [{i}] step={meta.get('step')}, source={meta.get('source')}, "
              f"checkpoint_id={cid[:8]}…, next={hist.next}")
        print(f"      values={hist.values}")
    print("  结论：")
    print("   · 时机：每执行完一个 super-step（一批节点，直到需要暂停/输入处）就写一个检查点")
    print("   · 内容：各 state 字段当前值(values) + 待执行节点(next) + checkpoint_id + 元信息(metadata)")

    # ---------------------------------------------------------------------
    # 示例 3：断点续跑 — 用 Command(resume=...) 从暂停处继续
    # ---------------------------------------------------------------------
    print(f"\n-- 示例: 断点续跑 — Command(resume=...) 从暂停处继续")
    # 模拟外部确认：支付 yes，图从 confirm 节点恢复并继续走到结束
    result = graph.invoke(Command(resume={"answer": "yes"}), config=config)
    print(f"  恢复后：stage = {result.get('stage')!r}，price = {result.get('price')}")
    print(f"  恢复后 notes = {result.get('notes')}")

    # 已结束：next == ()（空元组）
    final_snapshot = graph.get_state(config)
    print(f"  最终 next：{final_snapshot.next}（空表示已结束）")

    print("\n💡 思考：")
    print("  1. 换一个 thread_id 重新 invoke，就是另一笔「独立订单」，互不干扰。")
    print("  2. 去掉 checkpointer（compile 不带 checkpointer=），interrupt() 会报错，")
    print("     因为图无法「记住」暂停点。")
    print("  3. 本文件用 InMemorySaver，状态只存在内存里，进程退出即丢失；")
    print("     持久化落盘请看同目录 02_persistent_resume.py。")
