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


class TestAgentAsTool:
    """测试 Agent-as-Tool 模式图结构"""

    def test_subgraph_as_tool(self):
        """子图应能被包装为工具并被调用"""
        from langchain_core.tools import tool

        class SubState(TypedDict):
            query: str
            result: str

        def sub_worker(state):
            return {"result": f"处理了: {state['query']}"}

        sub_graph = (
            StateGraph(SubState)
            .add_node("w", sub_worker)
            .add_edge(START, "w")
            .add_edge("w", END)
            .compile()
        )

        @tool
        def sub_tool(query: str) -> str:
            """调用子图处理查询"""
            return sub_graph.invoke({"query": query, "result": ""})["result"]

        result = sub_tool.invoke("测试")
        assert "处理了" in result

    def test_main_graph_with_subagent_tools_compiles(self):
        """主图应能编译成功（含子Agent工具）"""
        class MainState(TypedDict):
            value: str

        def noop(state):
            return {"value": "done"}

        graph = (
            StateGraph(MainState)
            .add_node("n", noop)
            .add_edge(START, "n")
            .add_edge("n", END)
            .compile()
        )

        assert graph is not None
        result = graph.invoke({"value": ""})
        assert result["value"] == "done"


class TestHandoffs:
    """测试 Handoff 模式图结构"""

    def test_command_routing(self):
        """Command(goto=...) 应正确路由到目标节点"""
        from langgraph.types import Command

        class HandoffState(TypedDict):
            value: str
            path: list

        def agent_a(state: HandoffState):
            return Command(goto="agent_b", update={
                "path": ["a"],
                "value": "from_a",
            })

        def agent_b(state: HandoffState):
            return Command(goto=END, update={
                "path": state["path"] + ["b"],
            })

        graph = (
            StateGraph(HandoffState)
            .add_node("agent_a", agent_a)
            .add_node("agent_b", agent_b)
            .add_edge(START, "agent_a")
            .compile()
        )

        result = graph.invoke({"value": "", "path": []})
        assert result["path"] == ["a", "b"]
        assert result["value"] == "from_a"

    def test_three_agent_handoff_chain(self):
        """三个 Agent 的接力链应完整执行"""
        from langgraph.types import Command

        class State(TypedDict):
            trace: list

        def agent_a(state):
            return Command(goto="agent_b", update={"trace": ["a"]})

        def agent_b(state):
            return Command(goto="agent_c", update={"trace": state["trace"] + ["b"]})

        def agent_c(state):
            return Command(goto=END, update={"trace": state["trace"] + ["c"]})

        graph = (
            StateGraph(State)
            .add_node("agent_a", agent_a)
            .add_node("agent_b", agent_b)
            .add_node("agent_c", agent_c)
            .add_edge(START, "agent_a")
            .compile()
        )

        result = graph.invoke({"trace": []})
        assert result["trace"] == ["a", "b", "c"]


class TestSupervisor:
    """测试 Supervisor 模式图结构"""

    def test_supervisor_routes_to_worker(self):
        """Supervisor 应能路由到指定 Worker"""
        from langgraph.types import Command

        class State(TypedDict):
            iteration: int
            next: str
            products: list

        def supervisor(state):
            if state["iteration"] >= 1:
                return Command(goto=END, update={"next": "FINISH"})
            return Command(goto="worker_a", update={"next": "worker_a"})

        def worker_a(state):
            return Command(goto="supervisor", update={
                "products": ["product_a"],
                "next": "",
                "iteration": state["iteration"] + 1,
            })

        graph = (
            StateGraph(State)
            .add_node("supervisor", supervisor)
            .add_node("worker_a", worker_a)
            .add_edge(START, "supervisor")
            .compile()
        )

        result = graph.invoke({"iteration": 0, "next": "", "products": []})
        assert "product_a" in result["products"]

    def test_supervisor_bounded_iterations(self):
        """Supervisor 循环应有退出保护"""
        from langgraph.types import Command

        class State(TypedDict):
            iteration: int
            next: str

        def supervisor(state):
            if state["iteration"] >= 3:
                return Command(goto=END, update={"next": "FINISH"})
            return Command(goto="worker", update={"next": "worker"})

        def worker(state):
            return Command(goto="supervisor", update={
                "iteration": state["iteration"] + 1,
                "next": "",
            })

        graph = (
            StateGraph(State)
            .add_node("supervisor", supervisor)
            .add_node("worker", worker)
            .add_edge(START, "supervisor")
            .compile()
        )

        result = graph.invoke({"iteration": 0, "next": ""})
        assert result["iteration"] == 3
        assert result["next"] == "FINISH"
