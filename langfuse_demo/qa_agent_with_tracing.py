# =============================================================================
# Langfuse 可观测性接入 — LangGraph 智能问答 Agent 链路追踪
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解为什么多步骤 Agent 需要可观测性（print 调试不出规模）
#   ✅ 用 langfuse.langchain.CallbackHandler 给 LangGraph 图打点，代码零侵入
#   ✅ 用 session_id / user_id / tags 让 trace 可分组、可过滤
#   ✅ 在 Langfuse 控制台看到完整的 Span 树（LLM 调用、工具调用、耗时、Token 用量）
#
# 运行前检查：
# 1. 已安装依赖：pip install langfuse langgraph langchain-core langchain-openai python-dotenv
# 2. 在项目根目录 .env 中配置（参考 langfuse_demo/.env.example）：
#      ALIYUN_API_KEY       调用 Qwen 云端模型（工具调用场景本地小模型不够稳定）
#      LANGFUSE_PUBLIC_KEY  Langfuse 项目 Settings -> API Keys 获取
#      LANGFUSE_SECRET_KEY
#      LANGFUSE_BASE_URL    如 https://us.cloud.langfuse.com
# 3. 没有 Langfuse 账号？https://langfuse.com/cloud 可免费注册
# 4. 未配置 Langfuse Key 时 Agent 仍可正常运行，只是不会上报 trace
# =============================================================================

import sys
import os
import io
import uuid

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 确保项目根目录可导入（用于 utils/ 共享模块）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()  # ★ 必须在 import langfuse 之前调用，否则 Langfuse 读不到 API Key

from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from utils.model_utils import get_model


# =============================================================================
# 核心概念：为什么 Agent 需要 Langfuse
# =============================================================================
"""
可观测性（Observability）解决什么问题

  一个 LangGraph Agent 一次回答背后可能是：
    llm 节点判断要不要用工具 → tool 节点执行 → llm 节点再看结果 → ... → 最终回答

  step 一多，print 调试就不够用了：
    - 哪一步最慢？是 LLM 还是工具？
    - 模型到底看到了什么上下文、吐了什么 tool_calls？
    - 这次对话花了多少 token、多少钱？
    - 同一个用户的多轮对话是否连贯？

  Langfuse 用 CallbackHandler 挂载到 graph.invoke() 的 config 里，
  自动把每一次 LLM 调用、每一次工具调用记录成一棵 Span 树，
  代码本身完全不用改动业务逻辑。
"""


def get_langfuse_handler():
    """
    创建 Langfuse CallbackHandler。

    未配置 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY 时返回 None，
    Agent 仍可正常运行，只是不会上报 trace（不阻塞没有账号的学生）。
    """
    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        print("  【提示】未配置 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY，跳过 Langfuse 追踪")
        print("  【提示】前往 https://langfuse.com/cloud 免费注册后，在 .env 中配置 Key")
        return None

    from langfuse import get_client
    from langfuse.langchain import CallbackHandler

    langfuse = get_client()
    if not langfuse.auth_check():
        print("  【警告】Langfuse 鉴权失败，请检查 LANGFUSE_PUBLIC_KEY / SECRET_KEY / BASE_URL")
        return None

    print(f"  【Langfuse】已连接：{os.getenv('LANGFUSE_BASE_URL')}")
    return CallbackHandler()


def build_qa_agent():
    """
    构建带工具调用的智能问答 Agent（标准 LangGraph ReAct 结构）。

    工具：
    - search_knowledge: 模拟知识库搜索（AI 可观测性相关概念）
    - calculate: 安全的数学计算
    - get_weather: 模拟天气查询

    这部分和 05_practical/search_qa_agent.py 结构一致 —
    本文件的重点不是"怎么搭图"，而是"怎么给已有的图加追踪"。
    """
    # 1. 定义工具
    @tool
    def search_knowledge(query: str) -> str:
        """在知识库中搜索 AI 可观测性相关概念（trace、span、Langfuse 等）"""
        kb = {
            "trace": "Trace（链路）代表一次完整的用户请求，包含从输入到输出的所有步骤。",
            "span": "Span（跨度）代表 trace 中的一个步骤，比如一次 LLM 调用或一次工具调用。",
            "Langfuse": "Langfuse 是一个开源的 LLM 可观测性平台，用于追踪、评估和调试 AI 应用。",
            "token": "Token 用量决定了每次 LLM 调用的成本，Langfuse 会自动统计并计费。",
        }
        for key, value in kb.items():
            if key in query:
                print(f"  [工具: search_knowledge] 匹配关键词 '{key}'")
                return value
        print(f"  [工具: search_knowledge] 未找到匹配")
        return f"知识库中未找到关于'{query}'的信息。"

    @tool
    def calculate(expression: str) -> str:
        """执行数学计算（安全的 eval 环境）"""
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            print(f"  [工具: calculate] {expression} = {result}")
            return str(result)
        except Exception as e:
            return f"计算错误: {e}"

    @tool
    def get_weather(city: str) -> str:
        """查询指定城市的天气（模拟数据，用于演示工具调用追踪）"""
        weather_kb = {
            "北京": "晴，18-26℃，空气质量良",
            "上海": "多云，20-27℃，空气质量优",
            "深圳": "小雨，24-29℃，空气质量优",
        }
        result = weather_kb.get(city, f"暂无 {city} 的天气数据")
        print(f"  [工具: get_weather] {city} -> {result}")
        return result

    tools = [search_knowledge, calculate, get_weather]
    tools_by_name = {t.name: t for t in tools}

    # 2. 获取模型并绑定工具（云端 Qwen，工具调用更稳定）
    model = get_model("qwen")
    if model is None:
        return None
    model_with_tools = model.bind_tools(tools)

    # 3. 系统提示词
    system_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个智能问答助手，可以搜索知识库、执行计算和查询天气。请根据用户的问题合理使用工具。"),
        MessagesPlaceholder("messages"),
    ])

    # 4. 状态与节点
    class QAState(TypedDict):
        messages: Annotated[list, add_messages]

    def llm_call(state: QAState):
        """LLM 节点：拼接提示词后调用模型"""
        full_messages = system_prompt.format_messages(messages=state["messages"])
        response = model_with_tools.invoke(full_messages)
        return {"messages": [response]}

    def tool_executor(state: QAState):
        """工具执行节点"""
        last_msg = state["messages"][-1]
        results = []
        for tc in last_msg.tool_calls:
            tool_func = tools_by_name[tc["name"]]
            result = tool_func.invoke(tc["args"])
            results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        return {"messages": results}

    def route_check(state: QAState):
        """路由检查：有 tool_calls 就去工具节点，否则结束"""
        last_msg = state["messages"][-1]
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            return "tool"
        return END

    # 5. 构建图 — 标准 ReAct Agent 结构
    graph = (
        StateGraph(QAState)
        .add_node("llm", llm_call)
        .add_node("tool", tool_executor)
        .add_edge(START, "llm")
        .add_conditional_edges("llm", route_check, ["tool", END])
        .add_edge("tool", "llm")
        .compile()
    )
    return graph


def qa_agent_with_tracing():
    """
    运行智能问答 Agent，并通过 Langfuse CallbackHandler 上报追踪数据。

    关键点：
      - config={"callbacks": [handler]} 是接入 Langfuse 唯一需要加的代码
      - metadata 里的 langfuse_session_id 把同一个用户的多轮对话分组显示
      - langfuse_tags 让你之后在控制台按问题类型筛选 trace
      - 脚本是短生命周期进程，结束前必须 flush()，否则最后一批数据可能来不及发送
    """
    print(f"\n-- 示例: 智能问答 Agent + Langfuse 追踪")

    handler = get_langfuse_handler()

    graph = build_qa_agent()
    if graph is None:
        print("  【跳过】请配置 ALIYUN_API_KEY 后重试")
        return

    # 同一个 session_id 代表同一位用户的一次多轮对话，
    # Langfuse 会把它们分组显示在 Sessions 视图，方便还原完整对话过程。
    session_id = f"demo-session-{uuid.uuid4().hex[:8]}"
    user_id = "student-demo"
    print(f"  【会话】session_id={session_id}  user_id={user_id}")

    questions = [
        ("什么是 Langfuse 里的 span？", "knowledge"),
        ("计算 (18 + 24) * 3", "calculate"),
        ("北京今天天气怎么样？", "weather"),
    ]

    for i, (question, category) in enumerate(questions, start=1):
        print(f"\n【问题 {i}】{question}")

        # LangGraph 内部节点/工具调用的追踪名，与下面根 span 的展示名区分开
        config = {"run_name": "agent-execution"}
        if handler:
            config["callbacks"] = [handler]
            config["metadata"] = {
                "langfuse_user_id": user_id,
                "langfuse_session_id": session_id,
                "langfuse_tags": ["langfuse_demo", "qa_agent", category],
            }

        def run_agent():
            result = graph.invoke({"messages": [HumanMessage(content=question)]}, config=config)
            final = result["messages"][-1]
            return final.content if hasattr(final, 'content') and final.content else ""

        if handler:
            # 用一个根 span 包住整次调用，把 trace 的 input/output 收敛成
            # "用户问题 / 最终回答"，而不是 LangGraph 内部完整的 messages 状态。
            # CallbackHandler 产生的 LLM / 工具 span 依然会自动挂在这个根 span 下面。
            from langfuse import get_client
            langfuse = get_client()
            with langfuse.start_as_current_observation(
                as_type="span", name=f"qa-agent-{category}", input=question,
            ) as root_span:
                answer = run_agent()
                root_span.update(output=answer)
        else:
            answer = run_agent()

        if answer:
            print(f"【回答】{answer}")

    if handler:
        from langfuse import get_client
        get_client().flush()
        print(f"\n  【Langfuse】{len(questions)} 条 trace 已发送（session_id={session_id}）")
        print(f"  【Langfuse】前往控制台查看：{os.getenv('LANGFUSE_BASE_URL')}")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  Langfuse 可观测性接入 — LangGraph 智能问答 Agent")
    print("  给已有 Agent 加追踪，业务代码零改动")
    print("=" * 70 + "\n")

    qa_agent_with_tracing()

    print("\n" + "=" * 70)
    print("  接下来可以在 Langfuse 控制台探索：")
    print("    Traces 视图   — 查看每一次问答的完整 Span 树")
    print("    Sessions 视图 — 按 session_id 查看多轮对话")
    print("    按 tags 筛选  — knowledge / calculate / weather")
    print("=" * 70 + "\n")
