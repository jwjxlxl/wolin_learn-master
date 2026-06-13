# =============================================================================
# LangGraph 共享工具 — 提取常见的图构建模式
# =============================================================================
#
# 目标：减少跨文件的重复代码，教授良好的模块化设计
#
# 包含：
#   - create_tool_node():  通用的工具执行节点工厂
#   - create_router():     通用的条件路由函数工厂
#   - build_react_agent(): 一行构建 ReAct Agent 图
#   - save_graph_png():    保存图的可视化 PNG
# =============================================================================

from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END

# get_model 已移至项目根 utils/model_utils.py，作为跨子项目共享的通用函数
# 教学文件请使用: from utils.model_utils import get_model


def create_tool_node(tools: list):
    """
    创建一个通用的工具执行节点函数。

    工作原理：
      1. 从消息列表最后一条（AIMessage）中取出 tool_calls
      2. 逐个执行对应的工具函数
      3. 把每个工具结果包装成 ToolMessage，返回给模型

    Args:
        tools: @tool 装饰器标记的工具函数列表

    Returns:
        一个可注册为 LangGraph 节点的函数 f(state) → dict

    用法：
        tools = [add, multiply]
        graph.add_node("tool_node", create_tool_node(tools))
    """
    tools_by_name = {t.name: t for t in tools}

    def tool_node(state):
        messages = state["messages"]
        last_message = messages[-1]
        results = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]      # 工具名，如 "add"
            tool_args = tool_call["args"]      # 工具参数，如 {"a": 3, "b": 4}
            tool_id = tool_call["id"]          # 调用 ID，用于匹配 ToolMessage

            tool_func = tools_by_name[tool_name]
            result = tool_func.invoke(tool_args)

            results.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_id,
            ))

        return {"messages": results}

    return tool_node


def create_router():
    """
    创建一个通用的条件路由函数。

    检查消息列表最后一条是否有 tool_calls：
      - 有 → 返回 "tool_node"（继续执行工具）
      - 没有 → 返回 END（模型已给出最终回答）

    Returns:
        一个可注册为条件边的路由函数 f(state) → str

    用法：
        graph.add_conditional_edges("llm", create_router(), ["tool_node", END])
    """
    def should_continue(state):
        last_message = state["messages"][-1]
        has_tool_calls = hasattr(last_message, 'tool_calls') and last_message.tool_calls
        if has_tool_calls:
            return "tool_node"
        return END

    return should_continue


def build_react_agent(model, tools, system_prompt=None):
    """
    一行代码构建 ReAct Agent 图。

    封装了最常见的 Agent 构建模式：
      llm_call → (有 tool_calls?) → tool_node → 回到 llm_call → ... → END

    Args:
        model: 已绑定工具的模型实例（model.bind_tools(tools) 的结果）
        tools: 工具函数列表
        system_prompt: 可选的系统提示词字符串

    Returns:
        编译好的 LangGraph 图对象，可直接 invoke()

    用法：
        model = get_qwen_client()
        tools = [add, multiply]
        agent = build_react_agent(model.bind_tools(tools), tools)
        result = agent.invoke({"messages": [HumanMessage(content="3+4=?")]})
    """
    from typing import Annotated
    from typing_extensions import TypedDict
    from langgraph.graph.message import add_messages

    # 工具执行节点
    tool_executor = create_tool_node(tools)
    # 路由函数
    router = create_router()

    # 状态定义（使用内置的 add_messages 自动追加消息）
    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]

    # LLM 调用节点
    def llm_call(state: AgentState):
        messages = state["messages"]
        response = model.invoke(messages)
        return {"messages": [response]}

    # 构建图
    graph = (
        StateGraph(AgentState)
        .add_node("llm_call", llm_call)
        .add_node("tool_node", tool_executor)
        .add_edge(START, "llm_call")
        .add_conditional_edges("llm_call", router, ["tool_node", END])
        .add_edge("tool_node", "llm_call")
        .compile()
    )

    return graph


def save_graph_png(graph, filename: str = "graph.png"):
    """
    将 LangGraph 图保存为 PNG 图片。

    需要安装 graphviz 依赖：pip install langgraph[graph]

    Args:
        graph: 已编译的 LangGraph 图对象
        filename: 输出文件名（相对于 images/ 目录）

    用法：
        save_graph_png(agent, "my_agent.png")
    """
    import os
    png_data = graph.get_graph(xray=True).draw_mermaid_png()

    # 确保 images 目录存在
    images_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images")
    os.makedirs(images_dir, exist_ok=True)

    output_path = os.path.join(images_dir, filename)
    with open(output_path, "wb") as f:
        f.write(png_data)

    print(f"  图已保存到: {output_path}")
