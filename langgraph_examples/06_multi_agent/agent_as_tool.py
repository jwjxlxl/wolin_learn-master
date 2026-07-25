# =============================================================================
# Agent-as-Tool 模式 — 主 Agent 调用子 Agent 作为工具
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解 Agent-as-Tool：主 Agent 把子 Agent 当作可调用的工具
#   ✅ 用编译好的子图包装为 @tool
#   ✅ 掌握"控制权始终在主 Agent"的架构特点
#
# 运行前检查：
# 1. 已安装依赖：pip install langgraph langchain-core langchain-ollama pydantic
# 2. 已安装 Ollama 并下载模型：ollama pull qwen3.5:2b
# 3. 示例 1 无需 LLM（纯逻辑演示，可立即运行）
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
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from utils.model_utils import get_model


# =============================================================================
# 核心概念：Agent-as-Tool 是什么？
# =============================================================================
"""
Agent-as-Tool（Agent 作为工具）

  场景：公司智能客服总台
  你打电话到公司，前台接待员（主 Agent）接听后判断：
    - IT 问题 → 转接 IT 支持专员（子 Agent 1）
    - 人事问题 → 转接 HR 专员（子 Agent 2）
    - 报销问题 → 转接财务专员（子 Agent 3）

  关键特征：
    1. 每个子 Agent 是一个完整的 ReAct Agent（有自己的 LLM + 工具）
    2. 主 Agent 把子 Agent 当作普通工具调用
    3. 子 Agent 执行完毕返回结果 → 主 Agent 继续决策
    4. 控制权始终在主 Agent 手里，不会"丢失"

  与 Handoff 的区别：
    Agent-as-Tool = 前台打电话给某部门，等对方回复后继续（前台始终掌控通话）
    Handoff       = 前台把电话转接出去后就不管了（控制权完全移交）
"""

# =============================================================================
# 示例 : LLM 版 — 真正的 Agent-as-Tool（主 Agent 用 LLM 决策）
# =============================================================================

def agent_as_tool_with_llm():
    """
    用 LLM 做决策的 Agent-as-Tool 模式。

    主 Agent 使用 LLM 判断问题归属，子 Agent 使用 LLM 处理专业问题。
    架构与示例 1 完全相同，只是"关键词匹配"换成了"LLM 理解"。

    关键点：
      - 子 Agent 用 build_react_agent() 一行创建（复用 graph_helpers）
      - 主 Agent 标准 ReAct 循环，只是工具是"另一个 Agent"
    """
    print(f"\n-- 示例 2: LLM 版 — 真正的 Agent-as-Tool")

    model = get_model("qwen")
    if model is None:
        print("  【跳过】请安装 Ollama 并下载模型：ollama pull qwen3.5:2b")
        return

    # ===== 子 Agent 1: IT 支持 =====
    @tool
    def reset_password(user_id: str) -> str:
        """重置用户密码"""
        print(f"      [工具: reset_password] 为用户 {user_id} 重置密码")
        return f"已为用户 {user_id} 重置密码，初始密码为 123456。"

    @tool
    def check_system_status(system: str) -> str:
        """查询系统运行状态"""
        print(f"      [工具: check_system_status] 查询 {system}")
        return f"{system} 系统运行正常。"

    it_tools = [reset_password, check_system_status]

    from langgraph_examples.utils.graph_helpers import build_react_agent
    it_model = model.bind_tools(it_tools)
    # 用于解决IT系统问题的子Agent
    # 这个子Agent拥有自己的LLM，有自己的工具集，可以解决IT系统相关的问题
    it_sub_agent = build_react_agent(it_model, it_tools,
                                      system_prompt="你是 IT 支持专员，擅长解决密码、系统等技术问题。")

    # ===== 子 Agent 2: HR 支持 =====
    @tool
    def check_leave_balance(employee_id: str) -> str:
        """查询员工年假余额"""
        print(f"      [工具: check_leave_balance] 查询 {employee_id}")
        return f"员工 {employee_id} 当前年假余额：5 天。"

    @tool
    def lookup_policy(policy_type: str) -> str:
        """查询公司人事政策"""
        print(f"      [工具: lookup_policy] 查询 {policy_type}")
        policies = {
            "考勤": "工作时间：9:00-18:00，弹性打卡。",
            "报销": "差旅报销标准：高铁二等座、酒店 400 元/晚。",
        }
        return policies.get(policy_type, f"未找到 '{policy_type}' 相关政策。")

    hr_tools = [check_leave_balance, lookup_policy]
    hr_model = model.bind_tools(hr_tools)
    # 构建HR的子Agent
    hr_sub_agent = build_react_agent(hr_model, hr_tools,
                                      system_prompt="你是 HR 专员，擅长解答假期、政策等人事问题。")

    # ===== 子 Agent 3: 财务支持 =====
    @tool
    def check_budget(department: str) -> str:
        """查询部门预算使用情况"""
        print(f"      [工具: check_budget] 查询 {department}")
        return f"{department}年度预算 200 万，已使用 72%。"

    @tool
    def calculate_reimbursement(amount: float, category: str) -> str:
        """计算可报销金额"""
        print(f"      [工具: calculate_reimbursement] {category} {amount} 元")
        return f"{category}类报销 {amount} 元，可全额报销。"

    finance_tools = [check_budget, calculate_reimbursement]
    finance_model = model.bind_tools(finance_tools)
    finance_sub_agent = build_react_agent(finance_model, finance_tools,
                                           system_prompt="你是财务专员，擅长处理预算、报销等财务问题。")

    # ===== 子 Agent 打包为工具 =====
    def extract_final_answer(result: dict) -> str:
        for msg in reversed(result["messages"]):
            if hasattr(msg, 'content') and msg.content and msg.type == "ai":
                return msg.content
        return "（子 Agent 未返回有效回答）"

    @tool
    def call_it_support(query: str) -> str:
        """将 IT 相关问题转接给 IT 支持专员（密码、系统状态、电脑故障等）"""
        print(f"    [转接: IT 支持]")
        result = it_sub_agent.invoke({"messages": [HumanMessage(content=query)]})
        return extract_final_answer(result)

    @tool
    def call_hr_support(query: str) -> str:
        """将人事相关问题转接给 HR 专员（假期、政策、考勤、晋升等）"""
        print(f"    [转接: HR 支持]")
        result = hr_sub_agent.invoke({"messages": [HumanMessage(content=query)]})
        return extract_final_answer(result)

    @tool
    def call_finance(query: str) -> str:
        """将财务相关问题转接给财务专员（预算、报销、经费等）"""
        print(f"    [转接: 财务支持]")
        result = finance_sub_agent.invoke({"messages": [HumanMessage(content=query)]})
        return extract_final_answer(result)

    # ===== 主 Agent：总台接待员 =====
    all_tools = [call_it_support, call_hr_support, call_finance]
    main_model = model.bind_tools(all_tools)

    class MainState(TypedDict):
        messages: Annotated[list, add_messages]

    def main_llm(state: MainState):
        messages = state["messages"]
        response = main_model.invoke(messages)
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"  [主 Agent] 需要转接: {[tc['name'] for tc in response.tool_calls]}")
        return {"messages": [response]}

    def main_tool_node(state: MainState):
        last_msg = state["messages"][-1]
        results = []
        for tc in last_msg.tool_calls:
            func = {t.name: t for t in all_tools}[tc["name"]]
            result = func.invoke(tc["args"])
            results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        return {"messages": results}

    def main_router(state: MainState):
        last_msg = state["messages"][-1]
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            return "main_tool"
        return END

    main_graph = (
        StateGraph(MainState)
        .add_node("main_llm", main_llm)
        .add_node("main_tool", main_tool_node)
        .add_edge(START, "main_llm")
        .add_conditional_edges("main_llm", main_router, ["main_tool", END])
        .add_edge("main_tool", "main_llm")
        .compile()
    )

    # ===== 测试 =====
    questions = [
        "我的电脑密码忘记了，能帮我重置吗？",
        "我想查一下年假还剩几天？",
        "技术部今年的预算用得怎么样了？",
    ]

    for q in questions:
        print(f"【员工提问】{q}")
        result = main_graph.invoke({"messages": [HumanMessage(content=q)]})
        final = result["messages"][-1]
        if hasattr(final, 'content') and final.content:
            print(f"【总台回答】{final.content}")
        print()


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  Agent-as-Tool 模式 — 主 Agent 调用子 Agent 作为工具")
    print("  公司智能客服总台：前台接待员转接各部门专员")
    print("=" * 70 + "\n")
    agent_as_tool_with_llm()

    print("=" * 70)
    print("  回顾：主 Agent 把子 Agent 当作工具调用，控制权始终在主 Agent 手里")
    print("  接下来学习：handoffs.py（Handoff 模式 — 控制权完全移交）")
    print("=" * 70 + "\n")
