# =============================================================================
# 09_checkpointing — 02 状态持久化与进程崩溃恢复（SqliteSaver）
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解内存检查点（InMemorySaver）与落盘检查点（SqliteSaver）的区别
#   ✅ 用 SqliteSaver 把运行状态持久化到本地 sqlite 文件
#   ✅ 进程崩溃/重启后，用 thread_id 找到最新 Checkpoint 并恢复执行
#
# 运行前检查：
# 1. 已安装依赖：pip install langgraph langgraph-checkpoint-sqlite
# 2. 本示例为确定性演示（无需 API Key），分两次运行以模拟「崩溃 → 恢复」：
#      python 02_persistent_resume.py          # 第一次：运行到中断后「崩溃」
#      python 02_persistent_resume.py resume   # 第二次：从磁盘恢复并继续
#
# 为什么需要 SqliteSaver？
#   上一节用 InMemorySaver，状态只存在进程内存里，程序一退出就没了。
#   真实服务里 Worker 随时可能崩溃/重启，必须把检查点写到磁盘（或数据库），
#   重启后才能「找回记忆」。SqliteSaver 就把检查点持久化到一个 .sqlite 文件。
# =============================================================================

import sys
import os
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver

# 检查点数据库文件（放在本文件同目录下）
DB_PATH = os.path.join(os.path.dirname(__file__), "checkpoints.sqlite")

# 该会话的 thread_id（两次运行必须一致，才能找回同一份状态）
CONFIG = {"configurable": {"thread_id": "task-001"}}


class TaskState(TypedDict):
    """一个「多步任务」的共享状态。"""
    value: int   # 当前累计值
    step: int    # 已执行步骤数


def step_a(state: TaskState) -> dict:
    """节点 1：加 1。"""
    print(f"  [步骤A] value {state['value']} → {state['value'] + 1}")
    return {"value": state["value"] + 1, "step": state["step"] + 1}


def human_confirm(state: TaskState) -> dict:
    """节点 2：人工确认 —— interrupt() 暂停，等待外部输入后继续。"""
    cmd = interrupt({
        "type": "confirm",
        "value_so_far": state["value"],
        "hint": "请输入要继续累加的数（例如 5）",
    })
    extra = cmd.get("extra", 0) if isinstance(cmd, dict) else 0
    print(f"  [确认] 外部输入 extra = {extra}")
    return {"value": state["value"] + extra, "step": state["step"] + 1}


def step_b(state: TaskState) -> dict:
    """节点 3：再乘 10。"""
    print(f"  [步骤B] value {state['value']} → {state['value'] * 10}")
    return {"value": state["value"] * 10, "step": state["step"] + 1}


def build_graph(checkpointer):
    """构建多步任务图。checkpointer 由调用方传入。"""
    return (
        StateGraph(TaskState)
        .add_node("step_a", step_a)
        .add_node("human_confirm", human_confirm)
        .add_node("step_b", step_b)
        .add_edge(START, "step_a")
        .add_edge("step_a", "human_confirm")
        .add_edge("human_confirm", "step_b")
        .add_edge("step_b", END)
        .compile(checkpointer=checkpointer)
    )


def run_then_crash():
    """模拟第一个进程：运行到人工确认处，然后「崩溃」（with 退出 = 关闭连接）。"""
    print("=" * 60)
    print("  第一个进程：运行到人工确认处，然后模拟崩溃")
    print("=" * 60)
    # 打开 sqlite 检查点；with 退出 = 关闭连接 = 模拟崩溃，但数据已落盘
    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        graph = build_graph(checkpointer)
        result = graph.invoke({"value": 0, "step": 0}, CONFIG)
        snapshot = graph.get_state(CONFIG)
        print(f"\n  已暂停在：{snapshot.next}")
        print(f"  当前状态：value = {snapshot.values.get('value')}, step = {snapshot.values.get('step')}")
    print(f"\n>>> 进程崩溃！连接关闭（但状态已写入 {os.path.basename(DB_PATH)}）")
    print(f">>> 现在运行：python 02_persistent_resume.py resume")


def resume_after_crash():
    """模拟第二个进程：重新打开同一 sqlite 文件，找到最新 Checkpoint 并恢复。"""
    print("=" * 60)
    print("  第二个进程：从磁盘恢复并继续执行")
    print("=" * 60)
    # 重新打开同一 sqlite 文件（新进程、新图实例、新 checkpointer 实例）
    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        graph = build_graph(checkpointer)

        # 1) 用 thread_id 找回之前的状态
        snapshot = graph.get_state(CONFIG)
        print(f"  找回的状态：value = {snapshot.values.get('value')}, step = {snapshot.values.get('step')}")
        print(f"  暂停在：{snapshot.next}")

        # 2) 用 get_state_history 找到最新 Checkpoint
        history = list(graph.get_state_history(CONFIG))
        latest = history[0]
        print(f"  最新 Checkpoint：id = {latest.config['configurable'].get('checkpoint_id')}")
        print(f"     source = {latest.metadata.get('source')}, next = {latest.next}")

        # 3) 恢复执行：从 interrupt 处继续，传入外部输入
        result = graph.invoke(Command(resume={"extra": 5}), CONFIG)
        print(f"\n  恢复后：value = {result.get('value')}, step = {result.get('step')}")
        print("  计算过程：0 → +1 = 1 → 暂停 → 恢复 +5 = 6 → ×10 = 60")


if __name__ == "__main__":
    # 命令行参数：python 02_persistent_resume.py resume  → 只执行「恢复」
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "resume":
        resume_after_crash()
    else:
        run_then_crash()
