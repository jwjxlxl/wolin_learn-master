"""
Agent 开发范式教学模块 — 共享辅助函数

提供各范式共用的工具函数：
  - simulate_search():       模拟搜索返回预设知识
  - multi_candidate_generate(): 多次候选生成（用于 ToT）
  - run_agent_loop():        通用手动 Agent 循环
  - format_conversation_history(): 格式化对话历史
  - parse_react_response():  解析 Thought/Action/Observation 格式
"""

import sys
import os
import re

# 将项目根目录加入 sys.path，用于导入 utils/model_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.model_utils import get_model


# =============================================================================
# 模拟搜索
# =============================================================================

# 预设知识库，用于 Self-Ask 等需要搜索的范式演示
_SIMULATED_KNOWLEDGE = {
    "2016年奥斯卡最佳男主角": "2016年奥斯卡最佳男主角是莱昂纳多·迪卡普里奥（Leonardo DiCaprio），凭借《荒野猎人》获奖。",
    "莱昂纳多·迪卡普里奥 出生": "莱昂纳多·迪卡普里奥出生于1974年11月11日，美国加利福尼亚州洛杉矶。",
    "莱昂纳多·迪卡普里奥 年龄": "莱昂纳多·迪卡普里奥出生于1974年11月11日。",
    "Python之父": "Python 之父是吉多·范罗苏姆（Guido van Rossum），荷兰程序员。",
    "吉多·范罗苏姆 出生": "吉多·范罗苏姆（Guido van Rossum）出生于1956年1月31日，荷兰哈勒姆。",
    "杭州天气": "杭州昨天气温15°C，多云转晴，东南风3级。",
    "水的沸点": "标准大气压下，水的沸点是100°C（212°F）。",
    "太阳系最大行星": "太阳系中最大的行星是木星（Jupiter），直径约14.3万公里。",
    "中国最高峰": "中国最高峰是珠穆朗玛峰，海拔8848.86米。",
    "光的速度": "光在真空中的传播速度约为 299,792,458 米/秒（约 30万公里/秒）。",
}


def simulate_search(query: str) -> str:
    """
    模拟搜索：根据查询关键词返回预设知识。

    用于 Self-Ask 等需要搜索的范式演示，无需真实网络请求。

    参数:
        query: 用户查询文本

    返回:
        匹配到的知识文本，未匹配时返回"未找到相关信息"
    """
    query_lower = query.lower()

    for key, value in _SIMULATED_KNOWLEDGE.items():
        if key.lower() in query_lower or query_lower in key.lower():
            return value

    return f"未找到关于「{query}」的相关信息。"


# =============================================================================
# 多候选生成（用于 ToT）
# =============================================================================

def multi_candidate_generate(model, prompt: str, n: int = 3) -> list[str]:
    """
    多次候选生成：对同一 prompt 调用模型 n 次，返回不同结果。

    用于 Tree of Thoughts 范式，生成多个独立思路分支。

    参数:
        model: LangChain 模型实例（已通过 get_model() 获取）
        prompt: 输入 prompt
        n: 候选数量，默认 3

    返回:
        包含 n 个不同回答的列表
    """
    candidates = []

    for i in range(n):
        # 加入轻微提示词变化以增加多样性
        varied_prompt = prompt
        if i == 1:
            varied_prompt = prompt + "\n请从不同角度思考。"
        elif i == 2:
            varied_prompt = prompt + "\n请提供一种创意的解法。"

        response = model.invoke(varied_prompt)
        candidates.append(response.content)

    return candidates


# =============================================================================
# 通用 Agent 循环
# =============================================================================

def run_agent_loop(
    llm,
    system_prompt: str,
    tools: list,
    initial_question: str,
    max_iterations: int = 10,
    parse_fn=None,
    execute_fn=None,
) -> dict:
    """
    通用手动 Agent 循环：LLM → 解析 → 执行工具 → 观察 → 再思考。

    用于 prompt 版 ReAct 和 Reflexion 等需要自定义循环的场景。
    不依赖 LangGraph，纯手动实现 Thought→Action→Observation 循环。

    参数:
        llm: LangChain 模型实例
        system_prompt: 系统提示词，定义 ReAct 格式
        tools: 工具列表（dict，name -> callable）
        initial_question: 初始问题
        max_iterations: 最大迭代次数
        parse_fn: 自定义解析函数，接收 LLM 响应文本，返回 (action_name, action_input, thought)
                  默认尝试匹配 Thought:/Action:/Observation: 格式
        execute_fn: 自定义执行函数，接收 action_name 和 action_input，返回结果

    返回:
        {
            "steps": [...],       # 每步记录
            "final_answer": str,  # 最终答案
            "iteration_count": int
        }
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_question},
    ]

    steps = []
    final_answer = None

    for iteration in range(max_iterations):
        # 调用 LLM
        response = llm.invoke(
            [{"role": m["role"], "content": m["content"]} for m in messages]
        )
        response_text = response.content

        # 解析响应
        if parse_fn:
            action_name, action_input, thought = parse_fn(response_text)
        else:
            action_name, action_input, thought = _default_parse(response_text)

        step_info = {
            "iteration": iteration + 1,
            "thought": thought,
        }

        if action_name:
            # 需要执行工具
            print(f"  [思考] {thought}")
            print(f"  [行动] 调用工具: {action_name}({action_input})")

            if execute_fn:
                observation = execute_fn(action_name, action_input)
            else:
                tool_func = tools.get(action_name)
                if tool_func:
                    observation = tool_func(action_input) if action_input else tool_func()
                else:
                    observation = f"未知工具: {action_name}"

            print(f"  [观察] {str(observation)[:120]}")
            steps.append({**step_info, "action": action_name, "observation": str(observation)})

            # 将观察加入消息历史
            messages.append({
                "role": "assistant",
                "content": response_text,
            })
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}",
            })
        else:
            # 已得出最终答案
            final_answer = response_text
            step_info["final_answer"] = True
            steps.append(step_info)
            break

    if final_answer is None:
        final_answer = f"达到最大迭代次数（{max_iterations}），未能得出结论。"

    return {
        "steps": steps,
        "final_answer": final_answer,
        "iteration_count": len(steps),
    }


def _default_parse(response_text: str) -> tuple[str | None, str | None, str]:
    """
    默认 ReAct 格式解析：尝试匹配 Thought:/Action:/Action Input:/Observation:

    返回:
        (action_name, action_input, thought)
        如果没有 Action，action_name 和 action_input 为 None
    """
    # 提取 Thought
    thought_match = re.search(r'Thought:\s*(.+?)(?=\nAction:|$)', response_text, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else response_text.strip()

    # 提取 Action
    action_match = re.search(r'Action:\s*(.+?)(?:\n|$)', response_text)
    action_name = action_match.group(1).strip() if action_match else None

    # 提取 Action Input
    input_match = re.search(r'Action Input:\s*(.+?)(?:\n|$)', response_text)
    action_input = input_match.group(1).strip() if input_match else None

    return action_name, action_input, thought


# =============================================================================
# 格式化对话历史
# =============================================================================

def format_conversation_history(messages: list[tuple[str, str]]) -> str:
    """
    格式化对话历史为可读字符串。

    用于 Role-playing 范式打印输出。

    参数:
        messages: [(角色名, 消息内容), ...]

    返回:
        格式化后的对话文本
    """
    lines = []
    for role, content in messages:
        lines.append(f"【{role}】{content}")
    return "\n".join(lines)


# =============================================================================
# ReAct 响应解析
# =============================================================================

def parse_react_response(response_text: str) -> dict:
    """
    解析 ReAct 格式响应（Thought/Action/Observation）。

    支持以下格式：
        Thought: 我需要查询今天的天气
        Action: weather_api
        Action Input: 北京

    或最终回答：
        Thought: 根据以上信息，今天北京的天气是...
        Final Answer: 北京今天晴，气温15°C

    参数:
        response_text: LLM 响应文本

    返回:
        {
            "thought": str,
            "action": str | None,
            "action_input": str | None,
            "final_answer": str | None,
        }
    """
    result = {
        "thought": "",
        "action": None,
        "action_input": None,
        "final_answer": None,
    }

    # 提取 Thought
    thought_match = re.search(r'Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)', response_text, re.DOTALL)
    if thought_match:
        result["thought"] = thought_match.group(1).strip()
    else:
        result["thought"] = response_text.strip()

    # 提取 Action
    action_match = re.search(r'Action:\s*(.+?)(?:\n|$)', response_text)
    if action_match:
        result["action"] = action_match.group(1).strip()

    # 提取 Action Input
    input_match = re.search(r'Action Input:\s*(.+?)(?:\n|$)', response_text)
    if input_match:
        result["action_input"] = input_match.group(1).strip()

    # 提取 Final Answer
    final_match = re.search(r'Final Answer:\s*(.+)', response_text, re.DOTALL)
    if final_match:
        result["final_answer"] = final_match.group(1).strip()

    return result
