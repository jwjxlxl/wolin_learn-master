# =============================================================================
# Agent 智能体 — create_agent：Model + Tools + Prompt = 自主推理
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解 Agent = LLM + 工具 + 决策循环
#   ✅ 使用 create_agent() 一行创建 Agent
#   在LangChain1.0+中，create_agent是推荐的Agent创建方式
#   ✅ 观察 ReAct 模式（思考 → 行动 → 观察 → 循环）
# =============================================================================

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)


"""
什么是 Agent？

  普通 LLM: 你问 → 它答（没有工具，只能"说"）
  Agent:    你问 → 它思考需要什么信息 → 调用工具获取 → 综合回答

  ReAct 循环: Reason(推理) → Act(调用工具) → Observe(观察结果) → 重复

  生活化比喻: Agent = 餐厅服务员
    顾客点餐 → 服务员判断需要什么 → 去厨房/吧台取 → 端给顾客
    复杂订单？多跑几趟！（迭代）
"""

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from langchain_ollama import ChatOllama


# =============================================================================
# 示例 1: 最简单的 Agent — bind_tools
# =============================================================================

def bind_tools_demo(question: str = "北京今天天气怎么样？"):
    """
    用 model.bind_tools() 让模型知道有哪些工具可用。

    这是 Agent 的基础能力: 模型收到问题后，自己决定要不要调工具。
    如果它觉得不需要工具（比如简单问候），就直接回答。
    """
    print(f"\n-- 示例 1: bind_tools — 让模型知道工具有哪些")

    @tool
    def get_weather(city: str) -> str:
        """查询指定城市的天气。"""
        weather_db = {"北京": "晴，25°C", "上海": "多云，28°C", "广州": "小雨，30°C"}
        return weather_db.get(city, f"暂无 {city} 的天气数据")

    model = ChatOllama(model="qwen3.5:2b")
    agent = create_agent(model=model, tools=[get_weather],
                         system_prompt=SystemMessage("你是一个有用的助手"))

    response = agent.invoke({"messages": [HumanMessage(question)]})
    print(f"回复: {response['messages'][-1].content}")


# =============================================================================
# 示例 2: 多工具 Agent — create_agent 创建
# =============================================================================

def create_agent_demo():
    """
    用 create_agent() 创建拥有多个工具的 Agent。

    create_agent 是 LangChain 1.0+ 推荐的方式——一行代码把
    Model + Tools + SystemPrompt 组合成可运行的 Agent。
    """
    print(f"\n-- 示例 2: create_agent — 多工具 Agent")

    @tool
    def get_weather(city: str) -> str:
        """查询指定城市的天气。"""
        weather_db = {"北京": "晴，20°C", "上海": "多云，28°C", "深圳": "晴，29°C"}
        return weather_db.get(city, f"暂无 {city} 的天气数据")

    @tool
    def calculator(expression: str) -> str:
        """计算数学表达式。"""
        try:
            return f"计算结果: {eval(expression)}"
        except Exception as e:
            return f"计算错误: {e}"

    model = ChatOllama(model="qwen3.5:2b")
    agent = create_agent(model=model, tools=[get_weather, calculator],
                         system_prompt="你是一个有用的助手，请简洁回答。")

    questions = ["北京今天天气怎么样？", "23 加 45 等于多少？"]
    for q in questions:
        result = agent.invoke({"messages": [HumanMessage(content=q)]})
        print(f"  问: {q}")
        print(f"  答: {result['messages'][-1].content}\n")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n>>> 09_agent/agent — create_agent 智能体\n")

    # bind_tools_demo()
    create_agent_demo()

    # 接下来学习: agent_memory.py（Agent 记忆管理）
