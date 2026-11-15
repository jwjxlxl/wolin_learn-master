"""进阶选学：给两轮知识库 Agent 对话添加 Langfuse 回调追踪。

运行方式：python -m langfuse_demo.qa_agent_with_tracing
在项目根目录安装 requirements.txt 中的依赖，并配置 .env 中的
ALIYUN_API_KEY、LANGFUSE_PUBLIC_KEY、LANGFUSE_SECRET_KEY 和 LANGFUSE_BASE_URL。
"""

import uuid

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langfuse import propagate_attributes
from langfuse.langchain import CallbackHandler
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from langfuse_demo.common import get_langfuse_client, require_setting, setup_console_encoding
from langfuse_demo.rag_with_tracing import format_context, search_documents
from utils.model_utils import get_qwen_client


def build_qa_agent(model=None):
    """构建只有一个知识库工具的 LangGraph Agent。"""
    @tool
    def search_knowledge(query: str) -> str:
        """检索与 RAG 示例相同的课程资料。"""
        return format_context(search_documents(query))

    if model is None:
        model = get_qwen_client()
    if model is None:
        raise RuntimeError("无法创建通义千问模型，请检查 ALIYUN_API_KEY")
    # 每轮首次模型调用都指定知识库工具，保证课堂能看到工具调用轨迹。
    model_with_tools = model.bind_tools([search_knowledge], tool_choice="required")

    def llm_call(state: MessagesState):
        """让模型决定是否调用知识库工具。"""
        messages = [
            SystemMessage(content="你是知识库问答助手。回答课程概念前先检索知识库；资料不足时说明不知道。"),
            *state["messages"],
        ]
        response = (
            model.invoke(messages)
            if isinstance(state["messages"][-1], ToolMessage)
            else model_with_tools.invoke(messages)
        )
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("llm", llm_call)
    graph.add_node("tools", ToolNode([search_knowledge]))
    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", tools_condition)
    graph.add_edge("tools", "llm")
    return graph.compile()


def run_agent_turn(graph, question: str, history: list, config: dict) -> tuple[str, list]:
    """把前几轮的消息连同新问题传给 Agent。"""
    result = graph.invoke(
        {"messages": [*history, HumanMessage(content=question)]}, config=config
    )
    final_message = result["messages"][-1]
    return str(final_message.content or ""), result["messages"]


def qa_agent_with_tracing(langfuse_client, graph, handler) -> None:
    """给两轮连续对话统一设置 Langfuse 回调与会话标识。"""
    print("\n-- 示例 6（进阶选学）: 智能问答 Agent + Langfuse 追踪")
    session_id = f"demo-session-{uuid.uuid4().hex[:8]}"
    user_id = "student-demo"
    questions = (
        "什么是 Langfuse 里的 Span？",
        "刚才提到的 Span 与 Trace 有什么关系？",
    )
    history = []
    for index, question in enumerate(questions, start=1):
        config = {"run_name": "agent-execution", "callbacks": [handler]}
        # 每轮一条 Trace；历史消息负责连续对话，session_id 负责平台分组。
        with propagate_attributes(
            user_id=user_id, session_id=session_id, tags=["langfuse-demo", "qa-agent"]
        ):
            with langfuse_client.start_as_current_observation(
                as_type="span", name="qa-agent-turn", input=question
            ) as root_span:
                answer, history = run_agent_turn(graph, question, history, config)
                root_span.update(output=answer)
        print(f"\n【问题 {index}】{question}\n【回答】{answer}")

    tool_names = [
        call["name"]
        for message in history
        for call in getattr(message, "tool_calls", [])
    ]
    print(f"工具调用轨迹：{' → '.join(tool_names)}")
    print(f"\n会话标识：{session_id}")
    print("请在 Traces 中查看两轮调用，在 Sessions 中查看同一会话。")
    print("观察节点可区分普通 Span、模型 Generation 和工具调用。")


def main() -> None:
    """连接 Langfuse，运行 Agent，并在脚本结束前发送追踪数据。"""
    setup_console_encoding()
    try:
        require_setting("ALIYUN_API_KEY")
        langfuse_client = get_langfuse_client()
    except RuntimeError as exc:
        print(f"跳过 Agent 示例，请先配置模型与 Langfuse 项目密钥：{exc}")
        return
    try:
        graph = build_qa_agent()
        qa_agent_with_tracing(langfuse_client, graph, CallbackHandler())
    finally:
        langfuse_client.flush()


if __name__ == "__main__":
    main()
