# =============================================================================
# graph_helpers.py 单元测试
# =============================================================================
#
# 测试 graph_helpers 中的共享函数：
#   - create_tool_node(): 工具执行节点工厂
#   - create_router():    条件路由函数工厂
#   - build_react_agent(): Agent 图构建
#
# 运行：pytest langgraph_examples/tests/ -v
# =============================================================================

import pytest
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END

from langgraph_examples.utils.graph_helpers import (
    create_tool_node,
    create_router,
)


# ── 测试数据 ──────────────────────────────────────────────────────────

@tool
def add(a: int, b: int) -> int:
    """计算两个数的和"""
    return a + b


@tool
def greet(name: str) -> str:
    """生成问候语"""
    return f"你好，{name}！"


# ── create_tool_node() 测试 ───────────────────────────────────────────

class TestCreateToolNode:
    """测试工具执行节点工厂"""

    def test_single_tool_execution(self):
        """单个工具调用应正确执行并返回 ToolMessage"""
        tools = [add]
        tool_node = create_tool_node(tools)

        # 模拟 AIMessage 带一个 tool_call
        ai_message = AIMessage(
            content="",
            tool_calls=[{
                "name": "add",
                "args": {"a": 3, "b": 4},
                "id": "call_001",
            }]
        )
        state = {"messages": [ai_message]}

        result = tool_node(state)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], ToolMessage)
        assert result["messages"][0].content == "7"
        assert result["messages"][0].tool_call_id == "call_001"

    def test_multiple_tool_calls(self):
        """多个工具调用应全部执行并返回对应 ToolMessage"""
        tools = [add, greet]
        tool_node = create_tool_node(tools)

        ai_message = AIMessage(
            content="",
            tool_calls=[
                {"name": "add", "args": {"a": 10, "b": 20}, "id": "call_01"},
                {"name": "greet", "args": {"name": "小明"}, "id": "call_02"},
            ]
        )
        state = {"messages": [ai_message]}

        result = tool_node(state)

        assert len(result["messages"]) == 2
        assert result["messages"][0].content == "30"
        assert result["messages"][1].content == "你好，小明！"

    def test_empty_tool_calls(self):
        """空 tool_calls 应返回空列表"""
        tools = [add]
        tool_node = create_tool_node(tools)

        ai_message = AIMessage(content="直接回答", tool_calls=[])
        state = {"messages": [ai_message]}

        result = tool_node(state)

        assert result["messages"] == []

    def test_returns_string_for_non_string_result(self):
        """工具返回非字符串结果应被转换为字符串"""
        @tool
        def return_int(x: int) -> int:
            """将输入乘以 2 并返回"""
            return x * 2

        tools = [return_int]
        tool_node = create_tool_node(tools)

        ai_message = AIMessage(
            content="",
            tool_calls=[{
                "name": "return_int",
                "args": {"x": 21},
                "id": "call_x",
            }]
        )
        state = {"messages": [ai_message]}

        result = tool_node(state)

        assert result["messages"][0].content == "42"
        assert isinstance(result["messages"][0].content, str)


# ── create_router() 测试 ──────────────────────────────────────────────

class TestCreateRouter:
    """测试条件路由函数工厂"""

    def test_returns_tool_node_when_tool_calls_exist(self):
        """有 tool_calls 时应返回 'tool_node'"""
        router = create_router()

        ai_message = AIMessage(
            content="",
            tool_calls=[{"name": "add", "args": {}, "id": "1"}]
        )
        state = {"messages": [ai_message]}

        result = router(state)
        assert result == "tool_node"

    def test_returns_end_when_no_tool_calls(self):
        """没有 tool_calls 时应返回 END"""
        router = create_router()

        ai_message = AIMessage(content="直接回答，不需要工具")
        state = {"messages": [ai_message]}

        result = router(state)
        assert result == END

    def test_handles_message_without_tool_calls_attr(self):
        """消息对象没有 tool_calls 属性时应返回 END"""
        router = create_router()

        # HumanMessage 没有 tool_calls 属性
        human_message = HumanMessage(content="你好")
        state = {"messages": [human_message]}

        result = router(state)
        assert result == END
