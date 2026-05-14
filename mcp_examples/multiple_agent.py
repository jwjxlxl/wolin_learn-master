from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph_supervisor import create_supervisor
from langchain_core.tools import tool

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import model_untils


# =============================================================================
# 多 Agent  Supervisor 模式演示
# =============================================================================
#
# 架构：一个主 Agent（Supervisor）负责规划和分发任务，多个子 Agent（Worker）
#       各自负责处理单独的细分业务。
#
# 本案例包含 3 个子 Agent：
#   1. 计算专家（Calculator Agent）—— 负责数学计算
#   2. 翻译专家（Translator Agent）—— 负责中英文翻译
#   3. 写作专家（Writer Agent）—— 负责文案写作和创意生成
#
# 主 Agent 根据用户问题的类型，自动分发给对应的子 Agent 处理。
# =============================================================================

# -----------------------------------------------------------------------------
# 运行前检查
# -----------------------------------------------------------------------------
# 1. 已安装 langgraph-supervisor：pip install langgraph-supervisor
# 2. 已安装 langchain-openai：pip install langchain-openai
# -----------------------------------------------------------------------------

# 检查依赖
try:
    from langgraph_supervisor import create_supervisor
    LANGGRAPH_SUPERVISOR_AVAILABLE = True
except ImportError:
    LANGGRAPH_SUPERVISOR_AVAILABLE = False
    print("提示：运行此示例需要安装 langgraph-supervisor 包")
    print("  pip install langgraph-supervisor")


# =============================================================================
# 第一部分：定义子 Agent 的工具
# =============================================================================

@tool
def calculator(expression: str) -> str:
    """计算数学表达式。输入如 "2+3", "15*7", "100/3", "2**10" 等合法的 Python 数学表达式。"""
    try:
        # 只允许安全的数学字符
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "错误：表达式包含不安全的字符"
        result = eval(expression)  # noqa: S307 — 已做安全过滤
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"


@tool
def translate_chinese_to_english(text: str) -> str:
    """将中文文本翻译成英文。直接返回翻译结果。"""
    # 这里用简单规则模拟，实际项目中应调用翻译 API
    return f"[Translator] 中文 → English: {text}"


@tool
def translate_english_to_chinese(text: str) -> str:
    """将英文文本翻译成中文。直接返回翻译结果。"""
    return f"[Translator] English → 中文: {text}"


@tool
def creative_writer(topic: str, style: str = "正式") -> str:
    """根据给定主题和风格生成文案。返回生成的文案内容。"""
    return f"[Writer] 关于「{topic}」的{style}风格文案已生成。"


# =============================================================================
# 第二部分：创建子 Agent（Worker Agents）
# =============================================================================

def create_worker_agents(model):
    """
    创建三个专业化的子 Agent：
    1. 计算专家 —— 配备计算器工具
    2. 翻译专家 —— 配备中英互译工具
    3. 写作专家 —— 配备文案生成工具
    """

    # 1. 计算专家
    calculator_agent = create_agent(
        model=model,
        tools=[calculator],
        system_prompt=(
            "你是一个专业的数学计算专家，擅长处理各种数学计算问题。"
            "请使用 calculator 工具来精确计算用户提供的数学表达式。"
            "只回答计算相关的问题，对于非计算类问题请拒绝回答。"
        ),
        name="calculator_agent"
    )

    # 2. 翻译专家
    translator_agent = create_agent(
        model=model,
        tools=[translate_chinese_to_english, translate_english_to_chinese],
        system_prompt=(
            "你是一个专业的中英文翻译专家。"
            "如果用户给的是中文内容需要翻译成英文，使用 translate_chinese_to_english 工具。"
            "如果用户给的是英文内容需要翻译成中文，使用 translate_english_to_chinese 工具。"
            "只回答翻译相关的问题，对于非翻译类问题请拒绝回答。"
        ),
        name="translator_agent"
    )

    # 3. 写作专家
    writer_agent = create_agent(
        model=model,
        tools=[creative_writer],
        system_prompt=(
            "你是一个专业的文案写作专家，擅长根据主题生成各种风格的创意文案。"
            "请使用 creative_writer 工具来为用户生成内容。"
            "只回答写作相关的问题，对于非写作类问题请拒绝回答。"
        ),
        name="writer_agent"
    )

    return calculator_agent, translator_agent, writer_agent


# =============================================================================
# 第三部分：创建主 Agent（Supervisor）
# =============================================================================

def create_supervisor_agent(model, calculator_agent, translator_agent, writer_agent):
    """
    创建 Supervisor 主 Agent，负责：
    1. 理解用户意图
    2. 将任务分发给合适的子 Agent
    3. 汇总子 Agent 的结果并返回给用户
    """

    supervisor = create_supervisor(
        agents=[calculator_agent, translator_agent, writer_agent],
        model=model,
        prompt=(
            "你是一个智能任务调度中心，负责管理三个专家 Agent：\n"
            "- calculator_agent：负责数学计算问题（如 2+3 等于多少、15*7 等）\n"
            "- translator_agent：负责中英文翻译（如把中文翻译成英文，或把英文翻译成中文）\n"
            "- writer_agent：负责文案写作和创意内容生成\n\n"
            "请根据用户问题的类型，分发给对应的专家 Agent 处理。\n"
            "如果是计算类问题，交给 calculator_agent。\n"
            "如果是翻译类问题，交给 translator_agent。\n"
            "如果是写作类问题，交给 writer_agent。\n"
            "收到专家的结果后，请用简洁的中文回复用户。"
        )
    )

    return supervisor


# =============================================================================
# 第四部分：运行演示
# =============================================================================

def run_multi_agent_demo():
    """运行多 Agent Supervisor 模式演示"""

    if not LANGGRAPH_SUPERVISOR_AVAILABLE:
        print("【跳过】缺少依赖包，请先安装：pip install langgraph-supervisor")
        return

    # 获取模型
    model = model_untils.get_qwen_client()
    if model is None:
        print("【跳过】未配置阿里云 API Key，无法运行此示例")
        return

    print("=" * 60)
    print("  多 Agent Supervisor 模式演示")
    print("=" * 60)
    print()

    # 创建子 Agent
    print("正在创建子 Agent...")
    calculator_agent, translator_agent, writer_agent = create_worker_agents(model)
    print("  - 计算专家（calculator_agent）：负责数学计算")
    print("  - 翻译专家（translator_agent）：负责中英文翻译")
    print("  - 写作专家（writer_agent）：负责文案写作")
    print()

    # 创建主 Agent
    print("正在创建 Supervisor 主 Agent...")
    supervisor = create_supervisor_agent(
        model, calculator_agent, translator_agent, writer_agent
    )

    # 编译为可执行的工作流
    app = supervisor.compile()
    print("  Supervisor 工作流已编译完成")
    print()

    # 测试问题
    test_questions = [
        "计算一下 25 乘以 17 等于多少？",
        "请把 '今天天气真好' 翻译成英文",
        "帮我写一段关于人工智能的宣传文案，风格要正式一些",
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"{'─' * 60}")
        print(f"【测试 {i}】{question}")
        print(f"{'─' * 60}")

        try:
            result = app.invoke({
                "messages": [{"role": "user", "content": question}]
            })

            # 获取最后一条 AI 消息
            last_msg = result["messages"][-1]
            print(f"【回复】{last_msg.content}")

        except Exception as e:
            print(f"【错误】处理失败：{e}")

        print()

    print("=" * 60)
    print("  演示完成！")
    print("=" * 60)


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    run_multi_agent_demo()
