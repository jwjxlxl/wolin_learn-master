"""
Skill 技能演示 - 理解 Skill 的作用、原理和实现

学习目标：
  1. 理解 Skill 和 Tool 的区别（Skill = 组合 Tool 完成复杂目标）
  2. 掌握 Skill 的两种实现方式（system_prompt 编排 vs 封装函数）
  3. 实际运行 Agent 体验 Skill 的编排能力

依赖：pip install langchain langchain-openai python-dotenv
"""

import sys
import os
import io

# 设置标准输出编码为 UTF-8，避免 Windows GBK 编码错误
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.model_utils import get_qwen_client

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


# =============================================================================
# Skill 是什么？和 Tool 有什么区别？
# =============================================================================
#
# 💡 生活化比喻：
#    Tool = "一把螺丝刀"（单一功能，输入→输出）
#    Skill = "维修技能"（知道什么时候用螺丝刀、扳手、电钻）
#
# 📌 核心定义：
#    Tool  = 原子能力，一个函数，一个输入一个输出
#    Skill = 组合能力，编排多个 Tool 完成一个复杂目标
#
# 🔧 两种实现方式：
#    方式A（推荐）：给 Agent 提供多个 Tool + system_prompt 编排规则
#                   → Agent 自动选择 Tool 组合形成 Skill
#    方式B：        封装一个函数作为 Skill，内部调用多个子 Tool
#                   → 适合需要固定流程的场景
# =============================================================================


# =============================================================================
# 定义基础 Tool（原子能力）
# =============================================================================

@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。"""
    weather_db = {
        "北京": "晴，25°C",
        "上海": "多云，28°C",
        "广州": "小雨，30°C",
        "深圳": "晴，29°C",
        "成都": "阴，22°C",
        "杭州": "多云，26°C",
        "东京": "晴，20°C",
    }
    return weather_db.get(city, f"暂无 {city} 的天气数据")


@tool
def get_current_time() -> str:
    """获取当前日期和时间。"""
    from datetime import datetime
    return datetime.now().strftime("%Y年%m月%d日 %H:%M")


@tool
def calculator(expression: str) -> str:
    """计算数学表达式。"""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"计算错误: {e}"


@tool
def search_knowledge(query: str) -> str:
    """搜索知识库，回答退货、发货、发票、营业时间、地址等常见问题。"""
    knowledge_db = {
        "退货": "支持7天无理由退货，需保持商品完好。",
        "发货": "订单确认后24小时内发货。",
        "发票": "可在订单详情页申请电子发票。",
        "营业时间": "周一至周日 9:00-21:00。",
        "地址": "深圳市龙岗区富通海智科技园。",
    }
    synonyms = {"哪里": "地址", "位置": "地址", "在哪": "地址"}
    for syn, key in synonyms.items():
        if syn in query:
            query = query.replace(syn, key)
    for key, value in knowledge_db.items():
        if key in query:
            return value
    return "抱歉，未找到相关信息。"


@tool
def get_exchange_rate(currency: str) -> str:
    """查询汇率（模拟数据）。"""
    rates = {
        "美元": "7.25",
        "欧元": "7.89",
        "日元": "0.048",
        "英镑": "9.18",
        "港币": "0.93",
    }
    rate = rates.get(currency)
    if rate:
        return f"1{currency} = {rate}人民币"
    return f"暂无 {currency} 的汇率数据"


# =============================================================================
# 示例 1: Tool vs Skill 对比 - 理解区别
# =============================================================================

def demo1_tool_vs_skill():
    """
    Tool vs Skill 对比：先理解区别

    Tool：单一输入→输出的原子能力
    Skill：组合多个 Tool 完成复杂目标
    """
    print("=" * 70)
    print("  示例 1: Tool vs Skill 对比")
    print("=" * 70)
    print()

    print("""
┌─────────────────────────────────────────────────────────┐
│  Tool（原子能力）                                         │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐        │
│  │ get_weather│    │ calculator│    │search_know│        │
│  │  城市→天气 │    │ 算式→结果 │    │ 问题→答案 │        │
│  └──────────┘     └──────────┘     └──────────┘        │
│  特点：单一功能，一个输入一个输出                           │
│                                                         │
│  Skill（组合能力）                                        │
│  ┌──────────────────────────────────────────────┐       │
│  │        智能旅行助手 Skill                      │       │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐     │       │
│  │  │get_weather│ │get_exchange│ │calculator│     │       │
│  │  │  查天气   │ │  查汇率   │ │  算费用  │      │       │
│  │  └──────────┘ └──────────┘ └──────────┘     │       │
│  └──────────────────────────────────────────────┘       │
│  特点：编排多个 Tool，完成复杂目标                         │
└─────────────────────────────────────────────────────────┘

一句话总结：
  Tool = 你会做什么（能力）
  Skill = 你知道什么时候做、怎么做（技能）
""")

    # 直接调用 Tool
    print("【直接调用 Tool】")
    print(f"  get_weather('北京') → {get_weather.invoke({'city': '北京'})}")
    print(f"  calculator('100*0.8') → {calculator.invoke({'expression': '100*0.8'})}")
    print(f"  get_exchange_rate('美元') → {get_exchange_rate.invoke({'currency': '美元'})}")
    print()


# =============================================================================
# 示例 2: 方式A - 通过 system_prompt 编排 Skill（推荐方式）
# =============================================================================

def demo2_skill_via_system_prompt():
    """
    方式A：通过 system_prompt 编排 Skill

    这是推荐方式：给 Agent 提供多个 Tool + system_prompt，
    Agent 会根据用户需求自动选择合适的 Tool 组合。
    """
    print("=" * 70)
    print("  示例 2: 方式A - 通过 system_prompt 编排 Skill（推荐）")
    print("=" * 70)
    print()

    model = get_qwen_client()
    if model is None:
        print("【跳过】未配置阿里云 API Key，无法运行此示例")
        return

    print("""
原理：Tool + system_prompt 编排规则 = Skill

  ┌──────────────────────────────────────────┐
  │  system_prompt 告诉 Agent：               │
  │  "当用户问旅行时，先查天气，再查汇率，     │
  │   最后算费用"                             │
  │                                          │
  │  Agent 自动选择 Tool 组合形成 Skill       │
  └──────────────────────────────────────────┘

  优点：灵活，Agent 根据上下文自动编排
  缺点：不保证每次都用相同的 Tool 组合
""")

    agent = create_agent(
        model,
        tools=[get_weather, get_exchange_rate, calculator, search_knowledge, get_current_time],
        system_prompt=(
            "你是一个智能助手，拥有以下技能：\n"
            "1. 【天气查询】查询城市天气 - 使用 get_weather\n"
            "2. 【汇率查询】查询货币汇率 - 使用 get_exchange_rate\n"
            "3. 【数学计算】计算数学表达式 - 使用 calculator\n"
            "4. 【知识搜索】回答退货、发货等客服问题 - 使用 search_knowledge\n"
            "5. 【时间查询】获取当前时间 - 使用 get_current_time\n"
            "\n"
            "当用户的问题涉及多个信息时，请组合使用以上工具。"
            "例如：用户问'去东京旅行要花多少钱'，你应该先查东京天气，"
            "再查日元汇率，再算换算金额。"
        ),
    )

    # 测试：需要 Agent 自动编排多个 Tool
    questions = [
        # 单 Tool 调用
        "北京今天天气怎么样？",
        # 自动组合 2 个 Tool：天气 + 汇率
        "我想去东京旅行，那边天气怎么样？日元汇率是多少？",
        # 自动组合 3 个 Tool：知识搜索 + 时间 + 计算
        "你们退货政策是什么？现在几点？100减37等于多少？",
    ]

    for q in questions:
        print(f"{'─' * 50}")
        print(f"【用户】{q}")
        r = agent.invoke({"messages": [HumanMessage(content=q)]})
        print(f"【回答】{r['messages'][-1].content}")
        print()


# =============================================================================
# 示例 3: 方式B - 封装函数作为 Skill
# =============================================================================

def demo3_skill_via_function():
    """
    方式B：封装函数作为 Skill

    将多个 Tool 的调用封装到一个函数中，
    形成一个固定流程的 Skill。
    """
    print("=" * 70)
    print("  示例 3: 方式B - 封装函数作为 Skill")
    print("=" * 70)
    print()

    print("""
原理：把多个 Tool 的调用流程封装成一个函数

  ┌──────────────────────────────────────────┐
  │  @tool                                   │
  │  def travel_skill(city, budget):         │
  │      weather = get_weather(city)  ← Tool1│
  │      rate = get_exchange_rate(...)← Tool2│
  │      cost = calculator(...)      ← Tool3│
  │      return 综合结果                      │
  └──────────────────────────────────────────┘

  优点：流程固定，输出确定
  缺点：不够灵活，每次都走相同的流程
""")

    # 定义封装 Skill
    @tool
    def travel_skill(city: str, budget_cny: int) -> str:
        """旅行规划技能：查询目的地天气、汇率，并估算预算。

        Args:
            city: 目的地城市
            budget_cny: 人民币预算（元）
        """
        # 内部调用多个 Tool
        weather = get_weather.invoke({"city": city})
        rate = get_exchange_rate.invoke({"currency": "美元"})
        cost = calculator.invoke({"expression": f"{budget_cny} / 7.25"})

        return (
            f"旅行规划 - {city}\n"
            f"  天气: {weather}\n"
            f"  汇率: {rate}\n"
            f"  预算: {budget_cny}元人民币 ≈ {cost}美元\n"
            f"  建议: 根据天气准备衣物，注意汇率波动"
        )

    @tool
    def customer_service_skill(query: str) -> str:
        """智能客服技能：搜索知识库 + 判断营业状态。

        Args:
            query: 用户问题
        """
        info = search_knowledge.invoke({"query": query})
        time = get_current_time.invoke({})

        return f"客服回复: {info}\n当前时间: {time}"

    # 直接调用 Skill
    print("【直接调用封装 Skill】")
    r1 = travel_skill.invoke({"city": "东京", "budget_cny": 10000})
    print(f"  travel_skill('东京', 10000):\n{r1}")
    print()
    r2 = customer_service_skill.invoke({"query": "退货政策是什么"})
    print(f"  customer_service_skill('退货政策'):\n{r2}")
    print()

    # 在 Agent 中使用
    model = get_qwen_client()
    if model is None:
        print("【跳过】未配置阿里云 API Key，无法运行此示例")
        return

    agent = create_agent(
        model,
        tools=[travel_skill, customer_service_skill, get_weather],
        system_prompt=(
            "你是一个智能助手，拥有以下技能：\n"
            "1. 【旅行规划】travel_skill - 查天气、汇率、算预算\n"
            "2. 【智能客服】customer_service_skill - 搜索知识库+判断营业状态\n"
            "3. 【天气查询】get_weather - 单独查天气\n"
            "\n"
            "根据用户需求选择合适的技能。"
        ),
    )

    questions = [
        "帮我规划一下去东京的旅行，预算1万元",
        "你们退货政策是什么？",
    ]

    for q in questions:
        print(f"{'─' * 50}")
        print(f"【用户】{q}")
        r = agent.invoke({"messages": [HumanMessage(content=q)]})
        print(f"【回答】{r['messages'][-1].content}")
        print()


# =============================================================================
# 示例 4: Skill 的核心价值 - 没有 Skill 的 Agent vs 有 Skill 的 Agent
# =============================================================================

def demo4_skill_value():
    """
    对比演示：没有 Skill 编排 vs 有 Skill 编排的 Agent

    直观展示 Skill 的价值：让 Agent 从"只会单点操作"变成"能完成复杂任务"
    """
    print("=" * 70)
    print("  示例 4: Skill 的核心价值 - 对比演示")
    print("=" * 70)
    print()

    model = get_qwen_client()
    if model is None:
        print("【跳过】未配置阿里云 API Key，无法运行此示例")
        return

    question = "我想去东京旅行，帮我看看天气、汇率，算一下5000元能换多少美元"

    # ---- 没有 Skill 编排的 Agent（只有工具，没有编排提示）----
    print("【没有 Skill 编排的 Agent】（只给 Tool，不给编排提示）")
    agent_no_skill = create_agent(
        model,
        tools=[get_weather, get_exchange_rate, calculator],
        system_prompt="你是一个助手，请使用工具来回答问题。",
    )
    print(f"  用户: {question}")
    r = agent_no_skill.invoke({"messages": [HumanMessage(content=question)]})
    print(f"  回答: {r['messages'][-1].content}")
    print()

    # ---- 有 Skill 编排的 Agent（给 Tool + 编排提示）----
    print("【有 Skill 编排的 Agent】（给 Tool + system_prompt 编排规则）")
    agent_with_skill = create_agent(
        model,
        tools=[get_weather, get_exchange_rate, calculator],
        system_prompt=(
            "你是一个旅行助手，拥有以下技能：\n"
            "【旅行规划 Skill】：当用户询问旅行相关问题时，请依次：\n"
            "  1. 调用 get_weather 查询目的地天气\n"
            "  2. 调用 get_exchange_rate 查询汇率\n"
            "  3. 调用 calculator 计算换算金额\n"
            "  4. 综合以上信息给出完整的旅行建议\n"
        ),
    )
    print(f"  用户: {question}")
    r = agent_with_skill.invoke({"messages": [HumanMessage(content=question)]})
    print(f"  回答: {r['messages'][-1].content}")
    print()

    print("""
💡 对比总结：

  没有 Skill 编排：
    Agent 拿到工具后，可能只用部分工具，回答不够完整

  有 Skill 编排：
    Agent 知道应该按什么顺序组合工具，回答更完整、更有条理

  这就是 Skill 的核心价值：
    Skill = Tool + 编排规则
    让 Agent 从"有工具但不知道怎么组合"变成"有技能，知道什么时候用什么工具"
""")


# =============================================================================
# 示例 5: Skill 原理总结
# =============================================================================

def demo5_summary():
    """Skill 原理总结"""
    print("=" * 70)
    print("  示例 5: Skill 原理总结")
    print("=" * 70)
    print()

    print("""
┌──────────────────────────────────────────────────────────┐
│  Skill 的三个核心理解                                     │
│                                                          │
│  1️⃣  Skill 是什么？                                      │
│     Skill = 组合多个 Tool 完成一个复杂目标                  │
│     就像人类的"技能"不是单一动作，而是一系列动作的组合       │
│                                                          │
│  2️⃣  Skill 怎么实现？                                     │
│     方式A（推荐）: 多个 Tool + system_prompt 编排规则       │
│        → Agent 自动选择 Tool 组合，最灵活                  │
│     方式B: 封装函数作为 Skill，内部调用多个子 Tool          │
│        → 适合需要固定流程、确定性输出的场景                 │
│                                                          │
│  3️⃣  Skill vs Tool 怎么选？                               │
│     用 Tool: 功能简单、输入输出明确（查天气、算数学）       │
│     用 Skill: 需要多步骤、条件判断、组合多种能力            │
│              （旅行规划、智能客服、数据分析）                │
│                                                          │
│  🔧 实际项目最佳实践：                                     │
│     1. 先定义好基础 Tool（原子能力）                        │
│     2. 用 system_prompt 告诉 Agent 怎么组合 Tool           │
│     3. 复杂场景可以封装函数作为 Skill                      │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  Function Calling / MCP / Skill 的关系（自底向上）         │
│                                                          │
│  Function Calling（底层协议：让模型能调用函数）             │
│        ↑                                                 │
│  MCP（标准化连接层：让 Tool 可跨平台复用）                  │
│        ↑                                                 │
│  Skill（应用层编排：组合多个 Tool 完成复杂任务）            │
└──────────────────────────────────────────────────────────┘
""")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  Skill 技能演示 - 理解 Skill 的作用、原理和实现")
    print("=" * 70)

    # 示例 1: Tool vs Skill 对比（无需 API Key，直接运行）
    # demo1_tool_vs_skill()

    # 示例 2: 方式A - system_prompt 编排（需要 API Key）
    demo2_skill_via_system_prompt()

    # 示例 3: 方式B - 封装函数（需要 API Key）
    # demo3_skill_via_function()

    # 示例 4: Skill 价值对比（需要 API Key）
    # demo4_skill_value()

    # 示例 5: 原理总结（无需 API Key，直接运行）
    # demo5_summary()
