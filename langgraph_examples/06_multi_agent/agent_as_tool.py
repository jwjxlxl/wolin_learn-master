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
# 示例 1: 无 LLM 版 — 关键词路由的客服总台（理解模式结构）
# =============================================================================

def reception_desk_demo():
    """
    客服总台：不用 LLM，用关键词匹配理解 Agent-as-Tool 的控制流。

    START → main_agent（关键词判断）→ 调用子 Agent → 返回结果 → main_agent 汇总 → END

    三个子 Agent 是独立的 mini StateGraph：
      - IT 支持：reset_password / check_system_status
      - HR 支持：check_leave_balance / lookup_policy
      - 财务：check_budget / calculate_reimbursement
    """
    print(f"\n-- 示例 1: 无 LLM 版 — 关键词路由的客服总台")

    # ===== 子 Agent 1: IT 支持 =====
    @tool
    def reset_password(user_id: str) -> str:
        """重置用户密码"""
        return f"已为用户 {user_id} 重置密码，初始密码为 123456。"

    @tool
    def check_system_status(system: str) -> str:
        """查询系统运行状态"""
        status_map = {
            "OA": "OA 系统运行正常，响应时间 120ms",
            "邮箱": "邮箱系统运行正常，当前在线用户 328 人",
            "VPN": "VPN 系统正在维护中，预计 14:00 恢复",
        }
        return status_map.get(system, f"未找到系统 '{system}' 的状态信息。")

    it_tools = [reset_password, check_system_status]
    it_tools_by_name = {t.name: t for t in it_tools}

    class ITState(TypedDict):
        messages: Annotated[list, add_messages]

    def it_llm(state: ITState):
        """IT Agent 的 LLM 节点（模拟：用关键词匹配代替 LLM）"""
        from langchain_core.messages import AIMessage
        # 只看第一条 HumanMessage（原始问题）
        original_query = ""
        for msg in state["messages"]:
            if hasattr(msg, 'content') and msg.type == "human":
                original_query = msg.content
                break
        print(f"    [子 Agent: IT 支持] 处理: {original_query[:40]}...")

        # 检查是否已经有工具结果（有 ToolMessage 说明工具已执行过，该返回最终回答了）
        has_tool_result = any(hasattr(m, 'type') and m.type == "tool" for m in state["messages"])
        if has_tool_result:
            # 工具已执行，返回最终回答
            tool_msg = [m for m in state["messages"] if hasattr(m, 'type') and m.type == "tool"][-1]
            final_content = f"IT 支持回复：{tool_msg.content}"
            print(f"      最终回答: {final_content[:60]}...")
            return {"messages": [AIMessage(content=final_content)]}

        # 模拟 LLM 决策：选择工具
        if "密码" in original_query:
            tool_call = {"name": "reset_password", "args": {"user_id": "user_001"}, "id": "tc_1"}
        elif "系统" in original_query or "状态" in original_query:
            sys_name = "OA" if "OA" in original_query else "邮箱" if "邮箱" in original_query else "VPN"
            tool_call = {"name": "check_system_status", "args": {"system": sys_name}, "id": "tc_1"}
        else:
            tool_call = None

        if tool_call:
            ai_msg = AIMessage(content="", tool_calls=[tool_call])
            print(f"      决策: 调用 {tool_call['name']}")
        else:
            ai_msg = AIMessage(content="您好，IT 支持专员可以帮您处理密码和系统问题。")
        return {"messages": [ai_msg]}

    def it_tool_node(state: ITState):
        last_msg = state["messages"][-1]
        results = []
        for tc in last_msg.tool_calls:
            func = it_tools_by_name[tc["name"]]
            results.append(ToolMessage(content=str(func.invoke(tc["args"])), tool_call_id=tc["id"]))
        return {"messages": results}

    def it_router(state: ITState):
        last_msg = state["messages"][-1]
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            return "it_tool"
        return END

    it_graph = (
        StateGraph(ITState)
        .add_node("it_llm", it_llm)
        .add_node("it_tool", it_tool_node)
        .add_edge(START, "it_llm")
        .add_conditional_edges("it_llm", it_router, ["it_tool", END])
        .add_edge("it_tool", "it_llm")
        .compile()
    )

    # ===== 子 Agent 2: HR 支持 =====
    @tool
    def check_leave_balance(employee_id: str) -> str:
        """查询员工年假余额"""
        return f"员工 {employee_id} 当前年假余额：5 天，已使用 7 天。"

    @tool
    def lookup_policy(policy_type: str) -> str:
        """查询公司人事政策"""
        policies = {
            "考勤": "工作时间：9:00-18:00，弹性打卡，每月迟到 3 次以内不扣薪。",
            "报销": "差旅报销标准：高铁二等座、经济舱酒店 400 元/晚以内。",
            "晋升": "每年 6 月进行一次晋升评估，需满足当前职级满 1 年且绩效 B+ 以上。",
        }
        return policies.get(policy_type, f"未找到 '{policy_type}' 相关政策。")

    hr_tools = [check_leave_balance, lookup_policy]
    hr_tools_by_name = {t.name: t for t in hr_tools}

    class HRState(TypedDict):
        messages: Annotated[list, add_messages]

    def hr_llm(state: HRState):
        from langchain_core.messages import AIMessage
        original_query = ""
        for msg in state["messages"]:
            if hasattr(msg, 'content') and msg.type == "human":
                original_query = msg.content
                break
        print(f"    [子 Agent: HR 支持] 处理: {original_query[:40]}...")

        has_tool_result = any(hasattr(m, 'type') and m.type == "tool" for m in state["messages"])
        if has_tool_result:
            tool_msg = [m for m in state["messages"] if hasattr(m, 'type') and m.type == "tool"][-1]
            final_content = f"HR 支持回复：{tool_msg.content}"
            print(f"      最终回答: {final_content[:60]}...")
            return {"messages": [AIMessage(content=final_content)]}

        if "年假" in original_query or "假期" in original_query:
            tool_call = {"name": "check_leave_balance", "args": {"employee_id": "emp_001"}, "id": "tc_1"}
        elif "政策" in original_query or "考勤" in original_query or "报销" in original_query or "晋升" in original_query:
            p_type = "考勤" if "考勤" in original_query else "报销" if "报销" in original_query else "晋升"
            tool_call = {"name": "lookup_policy", "args": {"policy_type": p_type}, "id": "tc_1"}
        else:
            tool_call = None

        if tool_call:
            from langchain_core.messages import AIMessage
            ai_msg = AIMessage(content="", tool_calls=[tool_call])
            print(f"      决策: 调用 {tool_call['name']}")
        else:
            ai_msg = AIMessage(content="您好，HR 支持专员可以帮您查询假期、政策等信息。")
        return {"messages": [ai_msg]}

    def hr_tool_node(state: HRState):
        last_msg = state["messages"][-1]
        results = []
        for tc in last_msg.tool_calls:
            func = hr_tools_by_name[tc["name"]]
            results.append(ToolMessage(content=str(func.invoke(tc["args"])), tool_call_id=tc["id"]))
        return {"messages": results}

    def hr_router(state: HRState):
        last_msg = state["messages"][-1]
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            return "hr_tool"
        return END

    hr_graph = (
        StateGraph(HRState)
        .add_node("hr_llm", hr_llm)
        .add_node("hr_tool", hr_tool_node)
        .add_edge(START, "hr_llm")
        .add_conditional_edges("hr_llm", hr_router, ["hr_tool", END])
        .add_edge("hr_tool", "hr_llm")
        .compile()
    )

    # ===== 子 Agent 3: 财务支持 =====
    @tool
    def check_budget(department: str) -> str:
        """查询部门预算使用情况"""
        budgets = {
            "技术部": "技术部年度预算 200 万，已使用 145 万（72.5%），剩余 55 万。",
            "市场部": "市场部年度预算 150 万，已使用 130 万（86.7%），剩余 20 万。",
            "人事部": "人事部年度预算 80 万，已使用 42 万（52.5%），剩余 38 万。",
        }
        return budgets.get(department, f"未找到 '{department}' 的预算信息。")

    @tool
    def calculate_reimbursement(amount: float, category: str) -> str:
        """计算可报销金额"""
        limits = {"交通": 500, "餐饮": 200, "住宿": 400}
        limit = limits.get(category, 0)
        reimbursable = min(amount, limit)
        return f"{category}类报销 {amount} 元，限额 {limit} 元，可报销 {reimbursable} 元。"

    finance_tools = [check_budget, calculate_reimbursement]
    finance_tools_by_name = {t.name: t for t in finance_tools}

    class FinanceState(TypedDict):
        messages: Annotated[list, add_messages]

    def finance_llm(state: FinanceState):
        from langchain_core.messages import AIMessage
        original_query = ""
        for msg in state["messages"]:
            if hasattr(msg, 'content') and msg.type == "human":
                original_query = msg.content
                break
        print(f"    [子 Agent: 财务支持] 处理: {original_query[:40]}...")

        has_tool_result = any(hasattr(m, 'type') and m.type == "tool" for m in state["messages"])
        if has_tool_result:
            tool_msg = [m for m in state["messages"] if hasattr(m, 'type') and m.type == "tool"][-1]
            final_content = f"财务支持回复：{tool_msg.content}"
            print(f"      最终回答: {final_content[:60]}...")
            return {"messages": [AIMessage(content=final_content)]}

        if "预算" in original_query:
            dept = "技术部" if "技术" in original_query else "市场部" if "市场" in original_query else "人事部"
            tool_call = {"name": "check_budget", "args": {"department": dept}, "id": "tc_1"}
        elif "报销" in original_query:
            tool_call = {"name": "calculate_reimbursement", "args": {"amount": 350, "category": "交通"}, "id": "tc_1"}
        else:
            tool_call = None

        if tool_call:
            from langchain_core.messages import AIMessage
            ai_msg = AIMessage(content="", tool_calls=[tool_call])
            print(f"      决策: 调用 {tool_call['name']}")
        else:
            ai_msg = AIMessage(content="您好，财务支持专员可以帮您查询预算和报销信息。")
        return {"messages": [ai_msg]}

    def finance_tool_node(state: FinanceState):
        last_msg = state["messages"][-1]
        results = []
        for tc in last_msg.tool_calls:
            func = finance_tools_by_name[tc["name"]]
            results.append(ToolMessage(content=str(func.invoke(tc["args"])), tool_call_id=tc["id"]))
        return {"messages": results}

    def finance_router(state: FinanceState):
        last_msg = state["messages"][-1]
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            return "finance_tool"
        return END

    finance_graph = (
        StateGraph(FinanceState)
        .add_node("fin_llm", finance_llm)
        .add_node("finance_tool", finance_tool_node)
        .add_edge(START, "fin_llm")
        .add_conditional_edges("fin_llm", finance_router, ["finance_tool", END])
        .add_edge("finance_tool", "fin_llm")
        .compile()
    )

    # ===== 子 Agent 打包为工具 =====
    def extract_final_answer(result: dict) -> str:
        """从子 Agent 结果中提取最终回答"""
        for msg in reversed(result["messages"]):
            if hasattr(msg, 'content') and msg.content and msg.type == "ai":
                return msg.content
        return "（子 Agent 未返回有效回答）"

    @tool
    def call_it_support(query: str) -> str:
        """将 IT 相关问题转接给 IT 支持专员（密码、系统状态等）"""
        result = it_graph.invoke({"messages": [HumanMessage(content=query)]})
        return extract_final_answer(result)

    @tool
    def call_hr_support(query: str) -> str:
        """将人事相关问题转接给 HR 专员（假期、政策等）"""
        result = hr_graph.invoke({"messages": [HumanMessage(content=query)]})
        return extract_final_answer(result)

    @tool
    def call_finance(query: str) -> str:
        """将财务相关问题转接给财务专员（预算、报销等）"""
        result = finance_graph.invoke({"messages": [HumanMessage(content=query)]})
        return extract_final_answer(result)

    # ===== 主 Agent：总台接待员 =====
    all_tools = [call_it_support, call_hr_support, call_finance]
    all_tools_by_name = {t.name: t for t in all_tools}

    class MainState(TypedDict):
        messages: Annotated[list, add_messages]

    def main_llm(state: MainState):
        """总台接待员：判断问题归属哪个部门（关键词匹配模拟 LLM）"""
        # 只看原始 HumanMessage，不看 ToolMessage 避免循环
        original_query = ""
        has_tool_result = any(
            hasattr(m, 'type') and m.type == "tool" for m in state["messages"]
        )
        for msg in state["messages"]:
            if hasattr(msg, 'content') and msg.type == "human":
                original_query = msg.content
                break

        if has_tool_result:
            # 子 Agent 已返回结果，总台接待员汇总回答
            tool_msg = [m for m in state["messages"] if hasattr(m, 'type') and m.type == "tool"][-1]
            print(f"  [主 Agent: 总台接待员] 汇总结果")
            from langchain_core.messages import AIMessage
            return {"messages": [AIMessage(content=tool_msg.content)]}

        print(f"  [主 Agent: 总台接待员] 收到: {original_query}")
        from langchain_core.messages import AIMessage

        if any(k in original_query for k in ["密码", "系统", "VPN", "IT", "电脑", "网络"]):
            tc = {"name": "call_it_support", "args": {"query": original_query}, "id": "tc_main"}
            print(f"    判断: IT 问题 → 转接 IT 支持")
        elif any(k in original_query for k in ["年假", "假期", "政策", "考勤", "晋升", "HR", "人事"]):
            tc = {"name": "call_hr_support", "args": {"query": original_query}, "id": "tc_main"}
            print(f"    判断: 人事问题 → 转接 HR")
        elif any(k in original_query for k in ["预算", "报销", "财务", "经费"]):
            tc = {"name": "call_finance", "args": {"query": original_query}, "id": "tc_main"}
            print(f"    判断: 财务问题 → 转接财务")
        else:
            return {"messages": [AIMessage(content=f"您好！请问您需要什么帮助？您可以咨询 IT、人事或财务相关的问题。")]}

        return {"messages": [AIMessage(content="", tool_calls=[tc])]}

    def main_tool_node(state: MainState):
        last_msg = state["messages"][-1]
        results = []
        for tc in last_msg.tool_calls:
            func = all_tools_by_name[tc["name"]]
            result = func.invoke(tc["args"])
            print(f"    [子 Agent 返回] {result[:60]}...")
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
        "我的电脑密码忘记了，怎么办？",
        "我想查一下年假还剩几天？",
        "技术部今年的预算用得怎么样了？",
        "OA 系统今天怎么这么慢？",
    ]

    for q in questions:
        print(f"【员工提问】{q}")
        result = main_graph.invoke({"messages": [HumanMessage(content=q)]})
        final = result["messages"][-1]
        if hasattr(final, 'content') and final.content:
            print(f"【总台回答】{final.content}")
        print()


# =============================================================================
# 示例 2: LLM 版 — 真正的 Agent-as-Tool（主 Agent 用 LLM 决策）
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

    # reception_desk_demo()
    agent_as_tool_with_llm()

    print("=" * 70)
    print("  回顾：主 Agent 把子 Agent 当作工具调用，控制权始终在主 Agent 手里")
    print("  接下来学习：handoffs.py（Handoff 模式 — 控制权完全移交）")
    print("=" * 70 + "\n")
