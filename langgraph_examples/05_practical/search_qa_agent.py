# =============================================================================
# 综合实战 — 智能问答 Agent
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 构建一个能搜索知识库、执行计算、查询日期的完整 Agent
#   ✅ 用 ChatPromptTemplate 添加系统提示词
#   ✅ 理解 MessagesPlaceholder 的作用
#   ✅ 独立完成 Agent 的完整构建流程
#
# 运行前检查：
# 1. 已安装依赖：pip install langgraph langchain-core langchain-ollama
# 2. 已安装 Ollama 并下载模型：ollama pull qwen3.5:2b
# 3. （可选）配置 .env 中的 ALIYUN_API_KEY 使用云端模型
# =============================================================================

import sys
import os
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 确保项目根目录可导入（用于 utils/ 共享模块）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from typing import Annotated
from typing_extensions import TypedDict
from datetime import datetime
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from utils.model_utils import get_model


# =============================================================================
# 核心概念：添加系统提示词
# =============================================================================
"""
Agent 的角色定义

  之前的 Agent 没有"角色"，直接回答用户问题。
  通过 ChatPromptTemplate + system_prompt，我们可以给 Agent 定义角色：

  system_prompt = "你是一个智能问答助手，你可以搜索知识库、执行计算和查询日期。"

  ChatPromptTemplate.from_messages([
      ("system", "..."),                    ← 角色定义（告诉 Agent"你是谁"）
      MessagesPlaceholder("messages"),       ← 占位符（运行时替换为实际消息列表）
  ])

  这样 Agent 就会按照角色来回答问题，而不是随意发挥。
"""


# =============================================================================
# 示例: 智能问答 Agent — 知识搜索 + 计算 + 日期
# =============================================================================

def search_qa_agent():
    """
    实用案例：构建一个能搜索知识库、计算和查询日期的问答 Agent。

    工具：
    - search_knowledge: 模拟知识库搜索（公司、产品、福利等）
    - calculate: 安全的数学计算
    - get_date: 获取当前日期

    关键点：
      - ChatPromptTemplate + MessagesPlaceholder 添加系统提示
      - 单次对话可触发多个不同类型的工具
      - ReAct 循环自动处理工具编排
    """
    print(f"\n-- 示例: 智能问答 Agent — 知识搜索 + 计算 + 日期")

    # 1. 定义工具
    @tool
    def search_knowledge(query: str) -> str:
        """在知识库中搜索相关信息（公司、产品、福利、地址等）"""
        kb = {
            "公司": "我们公司是一家专注于 AI 技术的高科技企业。",
            "产品": "我们的核心产品是智能助手和自动化解决方案。",
            "福利": "公司提供五险一金、带薪年假、弹性工作制等福利。",
            "地址": "公司位于北京市海淀区中关村软件园。",
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
    def get_date() -> str:
        """获取当前日期和时间"""
        now = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
        print(f"  [工具: get_date] {now}")
        return now

    tools = [search_knowledge, calculate, get_date]
    tools_by_name = {t.name: t for t in tools}

    # 2. 获取模型并绑定工具
    model = get_model()
    if model is None:
        print("  【跳过】请安装 Ollama 并下载模型：ollama pull qwen3.5:2b")
        return
    model_with_tools = model.bind_tools(tools)

    # 3. 添加系统提示词
    system_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个智能问答助手。你可以搜索知识库、执行计算和查询日期。请根据用户的问题合理使用工具。"),
        MessagesPlaceholder("messages"),
    ])

    # 4. 定义状态和节点
    class QAState(TypedDict):
        messages: Annotated[list, add_messages]

    def llm_call(state: QAState):
        """LLM 节点：拼接提示词后调用模型"""
        messages = state["messages"]
        full_messages = system_prompt.format_messages(messages=messages)
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
        """路由检查"""
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

    # 6. 测试不同问题
    tests = [
        "公司的福利有哪些？",
        "计算 123 乘以 456 等于多少",
        "今天几号？",
    ]

    for q in tests:
        print(f"【问题】{q}")
        result = graph.invoke({"messages": [HumanMessage(content=q)]})
        final = result["messages"][-1]
        if hasattr(final, 'content') and final.content:
            print(f"【回答】{final.content}")
        print()


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  综合实战 — 智能问答 Agent")
    print("  构建能搜索知识库、计算、查询日期的完整 Agent")
    print("=" * 70 + "\n")

    print("【运行前检查】")
    print("  1. 已安装依赖：pip install langgraph langchain-core langchain-ollama")
    print("  2. 已安装 Ollama 并下载模型：ollama pull qwen3.5:2b")
    print("  3. 云端 API（可选）：代码改用 get_model(use_cloud=True)")
    print()

    search_qa_agent()

    print("=" * 70)
    print("  🎉 LangGraph 课程全部完成！")
    print("  回顾所学：")
    print("    01_introduction    — LangGraph 是什么 + 最简图")
    print("    02_state/branching — 状态管理 + 条件分支")
    print("    03_agent_loop      — ReAct Agent 循环（核心）")
    print("    04_workflows       — 提示链 + 路由")
    print("    05_practical       — 综合实战（本文件）")
    print("=" * 70 + "\n")
