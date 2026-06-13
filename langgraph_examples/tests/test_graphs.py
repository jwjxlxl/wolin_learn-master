# =============================================================================
# 图结构集成测试 — 验证图能正确编译和执行
# =============================================================================
#
# 测试各个教学文件中的图结构，确保：
#   - 图可以编译成功
#   - invoke() 不抛出异常
#   - 状态正确传递
#
# 注意：涉及 LLM 调用的测试在 CI 中会被跳过
# =============================================================================

import pytest
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class TestSimpleGraph:
    """测试最简图（simple_graph.py 同结构）"""

    def test_two_node_graph_compiles(self):
        """两个节点的顺序图应成功编译"""
        class TestState(TypedDict):
            count: int
            text: str

        def node_a(state: TestState):
            return {"count": 1, "text": "Hello"}

        def node_b(state: TestState):
            return {"count": state["count"] + 1}

        graph = (
            StateGraph(TestState)
            .add_node("a", node_a)
            .add_node("b", node_b)
            .add_edge(START, "a")
            .add_edge("a", "b")
            .add_edge("b", END)
            .compile()
        )

        result = graph.invoke({"count": 0, "text": ""})
        assert result["count"] == 2
        assert result["text"] == "Hello"

    def test_state_merge_not_replace(self):
        """节点返回值是合并而非替换（LangGraph 核心概念）"""
        class TestState(TypedDict):
            field_a: str
            field_b: int

        def node_a(state: TestState):
            return {"field_a": "updated"}  # 只更新 field_a

        def node_b(state: TestState):
            return {"field_b": 99}  # 只更新 field_b

        graph = (
            StateGraph(TestState)
            .add_node("a", node_a)
            .add_node("b", node_b)
            .add_edge(START, "a")
            .add_edge("a", "b")
            .add_edge("b", END)
            .compile()
        )

        result = graph.invoke({"field_a": "original", "field_b": 0})
        # field_a 被 node_a 更新，field_b 被 node_b 更新
        assert result["field_a"] == "updated"
        assert result["field_b"] == 99


class TestConditionalBranch:
    """测试条件分支图"""

    def test_conditional_routing(self):
        """条件边应根据路由函数选择正确路径"""
        class TestState(TypedDict):
            value: int
            path: str

        def start(state: TestState):
            return {}

        def high_path(state: TestState):
            return {"path": "high"}

        def low_path(state: TestState):
            return {"path": "low"}

        def router(state: TestState):
            if state["value"] > 50:
                return "high"
            return "low"

        graph = (
            StateGraph(TestState)
            .add_node("start", start)
            .add_node("high_path", high_path)
            .add_node("low_path", low_path)
            .add_edge(START, "start")
            .add_conditional_edges("start", router, {
                "high": "high_path",
                "low": "low_path",
            })
            .add_edge("high_path", END)
            .add_edge("low_path", END)
            .compile()
        )

        r1 = graph.invoke({"value": 80, "path": ""})
        assert r1["path"] == "high"

        r2 = graph.invoke({"value": 30, "path": ""})
        assert r2["path"] == "low"


class TestLoopGraph:
    """测试循环图"""

    def test_bounded_loop(self):
        """循环应有退出条件（不会无限循环）"""
        class TestState(TypedDict):
            count: int
            max_count: int

        def increment(state: TestState):
            return {"count": state["count"] + 1}

        def should_continue(state: TestState):
            if state["count"] >= state["max_count"]:
                return END
            return "increment"

        graph = (
            StateGraph(TestState)
            .add_node("increment", increment)
            .add_edge(START, "increment")
            .add_conditional_edges("increment", should_continue, ["increment", END])
            .compile()
        )

        result = graph.invoke({"count": 0, "max_count": 5})
        assert result["count"] == 5
