"""
ReAct Agent 教学示例 (CoT + Tool Use)
======================================
核心范式: Thought(思考) → Action(行动) → Observation(观察) → 循环

每一轮:
  Thought:  模型思考当前该做什么
  Action:   模型选择调用一个工具 (或决定 finished)
  Obs:      执行工具，将结果反馈给模型
当模型输出 finished 时循环结束，输出最终答案
"""

import logging
import re

from dotenv import load_dotenv

from agent.agent_paradigm.qwen import call_llm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
load_dotenv()


# ──────────────────── 可用工具 ────────────────────

def calculator(expr: str) -> str:
    """安全计算数学表达式 (仅支持 +, -, *, /, 括号)。"""
    if re.search(r"[^0-9+\-*/().%\s]", expr):
        return "Error: 表达式包含不允许的字符"
    try:
        return str(eval(expr))  # noqa: S307 — 已做正则过滤，教学用途
    except Exception as e:
        return f"Error: {e}"

def query_weather(city: str) -> str:
    """查询天气"""
    mock_db = {
        '北京':'晴天',
        '上海':'多云',
        '广州':'雨天',
        '深圳':'晴天',
        '成都':'晴天',
        '杭州':'多云',
        '南京':'晴天',
        '武汉':'多云',
        '重庆':'晴天',
        '天津':'晴天',
        '青岛':'晴天',
        '济南':'晴天',
        '沈阳':'晴天',
        '大连':'晴天',
        '长春':'晴天',
        '哈尔滨':'晴天',
        '呼和浩特':'晴天',
        '乌鲁木齐':'晴天',
        '拉萨':'晴天',
        '西宁':'晴天',
        '银川':'晴天',
        '兰州':'晴天',
        '太原':'晴天',
        '西安':'晴天',
        '石家庄':'晴天',
        '郑州':'晴天',
    }
    return f"{city}的天气是{mock_db[city]}"

TOOLS = {
    "calculator": {
        "description": "计算数学表达式，例如 calculator: 3 * (4 + 2)",
        "fn": calculator,
    },
    "query_weather": {
        "description": "查询天气，例如 query_weather: 北京",
        "fn": query_weather,
    }
}


def react_agent(question: str, max_turns: int = 5) -> str:
    """
    ReAct Agent: Thought → Action → Observation 循环

    参数:
      question:  用户的问题
      max_turns: 最大推理轮次，防止无限循环

    返回:
      最终答案
    """
    # 工具描述写入 system prompt
    tool_desc = "\n".join(
        f"  - {name}: {t['description']}" for name, t in TOOLS.items()
    )

    system_msg = (
        "你是一个会逐步推理并可以使用工具的助手。\n"
        f"你可以使用以下工具:\n{tool_desc}\n\n"
        "每轮请按以下格式输出:\n"
        "Thought: 你的思考\n"
        "Action: tool_name: tool_input  (或 finished: 你的最终答案)\n"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"问题: {question}"},
    ]

    for turn in range(1, max_turns + 1):
        response = call_llm(messages)
        logger.info(f"[ReAct Turn {turn}]\n{response}\n")

        # 用正则表达式从 LLM 的输出中提取 "Action:" 这一行
        # r"Action:\s*(.+)" 含义:
        #   "Action:" — 匹配字面量
        #   \s*      — 匹配 0 个或多个空白字符
        #   (.+)     — 捕获组1: 匹配 1 个或多个任意字符(即 Action 的完整内容)
        # 示例: response 包含 "Action: calculator: 100 - 85.5"
        #       则 action_match.group(1) = "calculator: 100 - 85.5"
        action_match = re.search(r"Action:\s*(.+)", response)
        if not action_match:
            # 没有找到 "Action:" 标记，说明 LLM 没有按约定格式输出
            # 可能是直接回答了问题，或输出格式不规范
            return response

        action_line = action_match.group(1).strip()

        # 判断 LLM 是否认为任务已完成
        # 如果 Action 以 "finished:" 开头，说明模型认为已经可以给出最终答案
        if action_line.lower().startswith("finished"):
            # 去掉 "finished:" 前缀，提取真正的最终答案
            final = re.sub(r"^finished:\s*", "", action_line, flags=re.IGNORECASE)
            return final.strip() or response  # 若提取为空则返回原始 response

        # 解析工具名称和工具输入，格式为: tool_name: tool_input
        if ":" in action_line:
            # 按第一个 ":" 分割(最多一次)，这样 tool_input 中的 ":" 不会被错误分割
            tool_name, tool_input = action_line.split(":", 1)
            tool_name = tool_name.strip()
            tool_input = tool_input.strip()
        else:
            # 只有工具名没有参数
            tool_name = action_line
            tool_input = ""

        # 执行工具: 从 TOOLS 字典中找到对应函数并调用
        if tool_name in TOOLS:
            obs = TOOLS[tool_name]["fn"](tool_input)
        else:
            obs = f"Unknown tool: {tool_name}"

        logger.info(f"  -> Action: {tool_name}({tool_input!r})\n  -> Observation: {obs}\n")

        # 将本轮对话追加到 messages，供下一轮 LLM 调用使用
        messages.append({"role": "assistant", "content": response})
        # 追加 Observation (工具执行结果)，让 LLM 能看到工具输出并决定下一步行动
        messages.append({"role": "user", "content": f"Observation: {obs}"})

    return "达到最大轮次，未能完成推理"


if __name__ == "__main__":
    print("=" * 60)
    print("ReAct Agent 演示 (CoT + 工具)")
    print("=" * 60)
    q = "小明有100元，买了3本书每本28.5元，又买了2支笔每支4.5元，还剩多少钱？深圳的天气如何"
    result = react_agent(q)
    print(f"问题: {q}")
    print(f"最终结果: {result}")
