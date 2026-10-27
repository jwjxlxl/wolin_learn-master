# =============================================================================
# Checkpointing 测试 — 验证断点恢复 / 持久化 / 时间旅行
# =============================================================================
#
# 对应教学模块：09_checkpointing/
# 所有测试均为确定性演示，不涉及 LLM 调用。
# =============================================================================

import os
import tempfile
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver


def _build_interrupt_graph(checkpointer):
    """构建「加一 → 暂停 → 乘十」的确定性图（对应 01/02 教学文件）。"""

    class TaskState(TypedDict):
        value: int

    def step_a(state: TaskState):
        return {"value": state["value"] + 1}

    def pause(state: TaskState):
        cmd = interrupt({"hint": "输入 extra"})
        extra = cmd.get("extra", 0) if isinstance(cmd, dict) else 0
        return {"value": state["value"] + extra}

    def step_b(state: TaskState):
        return {"value": state["value"] * 10}

    return (
        StateGraph(TaskState)
        .add_node("step_a", step_a)
        .add_node("pause", pause)
        .add_node("step_b", step_b)
        .add_edge(START, "step_a")
        .add_edge("step_a", "pause")
        .add_edge("pause", "step_b")
        .add_edge("step_b", END)
        .compile(checkpointer=checkpointer)
    )


def _build_plan_graph(checkpointer):
    """构建「A→B→C」顺序计划图（对应 03 教学文件）。"""

    class PlanState(TypedDict):
        plan: str
        step: int

    def step_a(state: PlanState):
        return {"plan": state["plan"] + "A; ", "step": 1}

    def step_b(state: PlanState):
        return {"plan": state["plan"] + "B; ", "step": 2}

    def step_c(state: PlanState):
        return {"plan": state["plan"] + "C; ", "step": 3}

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


class TestCheckpointBasics:
    """01_checkpoint_basics.py 对应：Checkpointer + thread_id + 断点续跑"""

    def test_invoke_creates_checkpoint(self):
        """挂 Checkpointer 后 invoke 应产生检查点，且暂停在 interrupt 节点"""
        graph = _build_interrupt_graph(InMemorySaver())
        config = {"configurable": {"thread_id": "t1"}}
        graph.invoke({"value": 0}, config)
        snapshot = graph.get_state(config)
        assert snapshot.next == ("pause",)
        assert snapshot.values["value"] == 1

    def test_resume_continues_from_interrupt(self):
        """Command(resume=...) 应从 interrupt 处继续，而非重头开始"""
        graph = _build_interrupt_graph(InMemorySaver())
        config = {"configurable": {"thread_id": "t1"}}
        graph.invoke({"value": 0}, config)
        result = graph.invoke(Command(resume={"extra": 5}), config)
        assert result["value"] == 60  # 0+1=1 → +5=6 → ×10=60

    def test_get_state_history_has_multiple_checkpoints(self):
        """get_state_history 应返回多个检查点，含 source/step 元信息"""
        graph = _build_interrupt_graph(InMemorySaver())
        config = {"configurable": {"thread_id": "t1"}}
        graph.invoke({"value": 0}, config)
        history = list(graph.get_state_history(config))
        assert len(history) >= 3  # input + 若干 super-step 检查点
        assert all("source" in h.metadata for h in history)


class TestSqlitePersistence:
    """02_persistent_resume.py 对应：SqliteSaver 落盘 + 崩溃恢复"""

    def test_sqlite_saver_persists_across_reconnect(self):
        """关闭连接后重新打开同一 sqlite 文件，应找回状态并继续执行"""
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "checkpoints.sqlite")
            config = {"configurable": {"thread_id": "t1"}}

            # 第一个「进程」：运行到 interrupt 后关闭连接（模拟崩溃）
            with SqliteSaver.from_conn_string(db) as checkpointer:
                graph = _build_interrupt_graph(checkpointer)
                graph.invoke({"value": 0}, config)
                assert graph.get_state(config).next == ("pause",)

            # 第二个「进程」：重新打开，找回最新检查点并恢复
            with SqliteSaver.from_conn_string(db) as checkpointer:
                graph = _build_interrupt_graph(checkpointer)
                assert graph.get_state(config).values["value"] == 1
                result = graph.invoke(Command(resume={"extra": 5}), config)
                assert result["value"] == 60


class TestTimeTravelAndFork:
    """03_time_travel_fork.py 对应：update_state / 时间旅行 / fork"""

    def test_update_state_rewrites_then_continues(self):
        """update_state 改写状态后，应从指定节点的后续节点继续"""
        graph = _build_plan_graph(InMemorySaver())
        config = {"configurable": {"thread_id": "t1"}}
        graph.invoke({"plan": "", "step": 0}, config)
        graph.update_state(config, {"plan": "改需求: X; ", "step": 0}, as_node="step_a")
        result = graph.invoke(None, config)
        assert result["plan"] == "改需求: X; B; C; "

    def test_time_travel_replay_from_checkpoint(self):
        """用 checkpoint_id 回到历史检查点重跑，不重复已执行节点"""
        graph = _build_plan_graph(InMemorySaver())
        config = {"configurable": {"thread_id": "t1"}}
        graph.invoke({"plan": "", "step": 0}, config)
        history = list(graph.get_state_history(config))
        past = next(h for h in history if h.next == ("step_b",))
        cid = past.config["configurable"]["checkpoint_id"]
        result = graph.invoke(None, {"configurable": {"thread_id": "t1", "checkpoint_id": cid}})
        assert result["plan"] == "A; B; C; "

    def test_fork_creates_independent_thread(self):
        """fork 到新 thread_id 后，两个会话各自独立演进"""
        graph = _build_plan_graph(InMemorySaver())
        config = {"configurable": {"thread_id": "t1"}}
        graph.invoke({"plan": "", "step": 0}, config)

        history = list(graph.get_state_history(config))
        past = next(h for h in history if h.next == ("step_b",))

        fork_cfg = {"configurable": {"thread_id": "fork"}}
        graph.update_state(fork_cfg, {"plan": past.values["plan"], "step": past.values["step"]}, as_node="step_a")
        graph.update_state(fork_cfg, {"plan": "分支: X; ", "step": 0}, as_node="step_a")
        fork_result = graph.invoke(None, fork_cfg)

        assert fork_result["plan"] == "分支: X; B; C; "
        # 原 thread 不受 fork 分支影响
        assert graph.get_state(config).values["plan"] == "A; B; C; "
