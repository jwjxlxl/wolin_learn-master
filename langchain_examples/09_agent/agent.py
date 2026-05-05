from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from utils import model_untils


# =============================================================================
# Agent 智能体
# =============================================================================
#  
# 用途：教学演示 - 使用 LangChain Agent 实现自主推理和工具调用
#
# 核心概念：
#   - Agent = "会思考的助手" = AI + 工具 + 决策循环
#   - ReAct 模式 = Reason（推理）+ Act（行动）+ Observe（观察）
# =============================================================================

# -----------------------------------------------------------------------------
# 运行前检查
# -----------------------------------------------------------------------------
# 1. 已安装 Ollama 服务
# 2. 已下载模型：ollama pull qwen3.5:2b
# -----------------------------------------------------------------------------


# =============================================================================
# 第一部分：理解 Agent
# =============================================================================
"""
什么是 Agent（智能体）？

🤖 定义
   Agent = "智能体" = 让 AI 不仅能回答问题，还能主动调用工具完成任务
   Agent 将语言模型与工具结合，创建能够推理、决策、迭代的系统

🔄 核心流程（ReAct 模式）
   思考(Reason) → 行动(Action) → 观察(Observe) → 再思考 → ...
   Agent 会一直运行，直到满足停止条件

🎯 为什么需要 Agent？

   普通的 LLM 只能生成文字：
   ❌ 无法主动获取外部信息
   ❌ 无法执行具体操作
   ❌ 遇到复杂问题无法分步解决

   有了 Agent：
   ✅ 可以调用工具获取实时数据
   ✅ 可以多步骤迭代解决问题
   ✅ 可以根据结果调整策略

💡 生活化比喻
   Agent = "餐厅服务员"
   - 顾客点餐（用户提问）
   - 服务员判断需要什么（推理）
   - 去厨房/吧台取东西（调用工具）
   - 把东西端给顾客（给出答案）
   - 复杂订单？多跑几趟！（迭代）
"""

# =============================================================================
# 示例 1: 创建agent
# =============================================================================

def bind_tools_demo():
    """

    这是 Agent 的基础能力：模型能知道有哪些工具可用
    """

    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage

    # 定义工具
    @tool
    def get_weather(city: str) -> str:
        """查询指定城市的天气情况。

        Args:
            city: 城市名称，如"北京"、"上海"
        """
        weather_db = {
            "北京": "晴，25°C",
            "上海": "多云，28°C",
            "广州": "小雨，30°C",
        }
        return weather_db.get(city, f"暂无 {city} 的天气数据")

    #实例化大模型的客户端
    model = ChatOllama(model="qwen3.5:2b")

    # 模型收到问题后，可以决定是否调用工具
    print("【发送问题给模型】")
    # 创建 Agent 并绑定工具
    agent = create_agent(
        model=model,
        tools=[get_weather],
        system_prompt=SystemMessage("你是一个有用的助手")
    )

    # 运行代理
    response = agent.invoke(
        {"messages": [HumanMessage("东京的天气怎么样")]}
    )

    print(response["messages"][-1].content)



# =============================================================================
# 示例 2: 使用 create_agent 创建 Agent
# =============================================================================

def create_agent_demo():
    """
    使用 create_agent 创建可调用天气和计算工具的 Agent

    最简洁的 Agent 创建方式，使用 LangChain 官方推荐的 create_agent
    """

    from langchain.tools import tool
    from langchain.agents import create_agent

    # 定义天气查询工具
    @tool
    def get_weather(location: str) -> str:
        """获取指定位置的天气信息。"""
        weather_db = {
            "北京": "晴，25°C",
            "上海": "多云，28°C",
            "广州": "小雨，30°C",
            "深圳": "晴，29°C",
        }
        return weather_db.get(location, f"{location} 的天气：暂无数据")

    # 定义计算工具
    @tool
    def calculator(expression: str) -> str:
        """计算数学表达式的结果。"""
        try:
            return f"计算结果：{eval(expression)}"
        except Exception as e:
            return f"计算错误：{e}"

    # 获取模型
    model = model_untils.get_qwen_client()
    if model is None:
        print("【跳过】未配置阿里云 API Key，无法运行此示例")
        return

    # 使用 create_agent 创建 Agent
    agent = create_agent(
        model,
        tools=[get_weather, calculator],
    )

    print("【Agent 创建成功】")
    print(f"  工具 1: get_weather - {get_weather.description}")
    print(f"  工具 2: calculator - {calculator.description}")
    print()

    # 调用 Agent
    questions = [
        "北京今天天气怎么样？",
        "23 加 45 等于多少？",
        "上海和广州哪个城市更热？",
    ]

    for question in questions:
        print(f"【用户提问】{question}")
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": question}]}
            )
            # 从返回的消息中提取最终回答
            last_msg = result["messages"][-1]
            print(f"【Agent 回答】{last_msg.content}")
        except Exception as e:
            print(f"【执行出错】{e}")
        print()


# =============================================================================
# 示例 3: 动态选择模型
# =============================================================================

def dynamic_model_selection_demo():
    """
    根据对话复杂性动态选择模型

    核心思路：
      - 简单问题 / 短对话 → 使用轻量模型（本地 Ollama），速度快、成本低
      - 复杂问题 / 长对话 → 使用高级模型（云端 Qwen），能力更强

    通过 middleware（中间件）机制，在每次模型调用前自动判断并切换模型
    """

    from langchain_core.tools import tool

    # 定义工具
    @tool
    def get_weather(location: str) -> str:
        """获取指定位置的天气信息。"""
        weather_db = {
            "北京": "晴，25°C",
            "上海": "多云，28°C",
            "广州": "小雨，30°C",
            "深圳": "晴，29°C",
        }
        return weather_db.get(location, f"{location} 的天气：暂无数据")

    @tool
    def calculator(expression: str) -> str:
        """计算数学表达式的结果。"""
        try:
            return f"计算结果：{eval(expression)}"
        except Exception as e:
            return f"计算错误：{e}"

    # -------------------------------------------------------------------------
    # 准备两个模型：基础模型 + 高级模型
    # -------------------------------------------------------------------------
    basic_model = ChatOllama(model="qwen3.5:2b")           # 本地轻量模型
    advanced_model = model_untils.get_qwen_client()        # 云端高级模型

    if advanced_model is None:
        print("【跳过】未配置阿里云 API Key，无法运行此示例")
        return

    # 用于跨 invoke 累积消息数（闭包状态）
    total_messages = [0]

    # -------------------------------------------------------------------------
    # 定义中间件：根据消息数量动态选择模型
    # -------------------------------------------------------------------------
    @wrap_model_call
    def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
        """根据对话复杂性选择模型。
        可以自定义策略（可以使用指定的关键字判断，也可以使用模型判断）

        策略：
          - 累积消息数 > 10 → 切换到高级模型（长对话需要更强的推理能力）
          - 累积消息数 ≤ 10 → 使用基础模型（简单任务不需要大模型）
        """
        # 累加当前请求的消息数（每次 invoke 的消息会持续增加）
        total_messages[0] += len(request.messages)
        msg_count = total_messages[0]

        if msg_count > 10:
            # 对较长的对话使用高级模型（使用 override API 替代已废弃的直接赋值）
            print(f"  [动态切换] 累积消息数={msg_count} -> 使用高级模型 (Qwen)")
            return handler(request.override(model=advanced_model))
        else:
            print(f"  [动态切换] 累积消息数={msg_count} -> 使用基础模型 (Ollama)")
            return handler(request)

    # -------------------------------------------------------------------------
    # 创建带中间件的 Agent
    # -------------------------------------------------------------------------
    agent = create_agent(
        model=basic_model,                        # 默认模型
        tools=[get_weather, calculator],
        middleware=[dynamic_model_selection]       # 注入动态选择中间件
    )

    # 调用 Agent
    questions = [
        "北京今天天气怎么样？",
        "23 加 45 等于多少？",
        "北京今天天气怎么样？适合户外活动吗？",
        "帮我算一下 (15 + 27) * 3 等于多少",
        "我想了解你们公司的地址和营业时间",
        "机器学习的应用领域有哪些？",
        "人工智能的应用领域有哪些？"
    ]

    for question in questions:
        print(f"【用户提问】{question}")
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": question}]}
            )
            last_msg = result["messages"][-1]
            print(f"【Agent 回答】{last_msg.content}")
        except Exception as e:
            print(f"【执行出错】{e}")
        print()


# =============================================================================
# 示例 4: 基于 ReAct 模式的通用 Agent
# =============================================================================

def react_agent_demo():
    """
    基于 ReAct（Reasoning + Acting）模式的通用 Agent

    ReAct 核心循环：
      Thought（思考）→ Action（行动/调用工具）→ Observation（观察结果）→ 循环或结束

    与普通 LLM 的区别：
      - 普通 LLM：直接回答，无法获取外部信息
      - ReAct Agent：先推理需要什么信息，再调用工具获取，最后综合回答

    本示例使用 LangChain 1.0+ 统一的 create_agent 方法实现，
    它内部自动构建 ReAct 循环图，是官方推荐的 Agent 创建方式。
    """

    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage

    # -------------------------------------------------------------------------
    # 第一步：定义工具（Agent 的"手脚"）
    # -------------------------------------------------------------------------
    # ReAct 模式中，工具就是 Action 的具体执行者
    # Agent 会根据推理结果，选择合适的工具来执行

    @tool
    def search_knowledge(query: str) -> str:
        """搜索知识库，获取相关信息。

        Args:
            query: 搜索关键词，如"公司地址"、"产品介绍"
        """
        knowledge_db = {
            "公司": "沃林数智 - 专注于 AI 教育和智能解决方案",
            "地址": "深圳市龙岗区富通海智科技园",
            "产品": "AI 学习平台、智能助手、企业 AI 培训",
            "电话": "400-888-9999",
            "营业时间": "周一至周五 9:00-18:00",
        }
        for key, value in knowledge_db.items():
            if key in query:
                return value
        return f"未找到关于'{query}'的信息，请尝试其他关键词。"

    @tool
    def calculator(expression: str) -> str:
        """计算数学表达式的结果。

        Args:
            expression: 数学表达式，如 "2+3"、"10*5"
        """
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"计算错误: {e}"

    @tool
    def get_weather(city: str) -> str:
        """查询指定城市的天气信息。

        Args:
            city: 城市名称，如"北京"、"上海"
        """
        weather_db = {
            "北京": "晴，25°C，空气质量良好",
            "上海": "多云，28°C，有轻微雾霾",
            "广州": "小雨，30°C，注意携带雨具",
            "深圳": "晴，29°C，紫外线较强",
        }
        return weather_db.get(city, f"暂无 {city} 的天气数据")

    @tool
    def get_current_time() -> str:
        """获取当前的日期和时间。"""
        from datetime import datetime
        return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

    tools = [search_knowledge, calculator, get_weather, get_current_time]

    # -------------------------------------------------------------------------
    # 第二步：获取模型
    # -------------------------------------------------------------------------
    model = model_untils.get_qwen_client()
    if model is None:
        print("【跳过】未配置阿里云 API Key，无法运行此示例")
        return

    # -------------------------------------------------------------------------
    # 第三步：创建 ReAct Agent
    # -------------------------------------------------------------------------
    # create_agent 内部自动构建了 ReAct 循环图：
    #   LLM思考 → 是否需要调用工具？ → 是 → 执行工具 → 回到LLM思考
    #                                   → 否 → 返回最终答案
    #
    # LangChain 1.0+ 统一使用 create_agent，无需区分 ReAct/Plan-and-Execute 等

    system_prompt = (
        "你是一个智能助手，能够通过推理和工具调用来回答用户的问题。\n"
        "请遵循 ReAct 模式：\n"
        "1. 先思考(Thought)：分析问题，判断需要什么信息\n"
        "2. 再行动(Action)：调用合适的工具获取信息\n"
        "3. 后观察(Observation)：根据工具返回的结果继续推理\n"
        "4. 重复以上步骤直到可以给出最终答案\n\n"
        "注意：对于需要实时数据的问题（如天气、时间），务必调用工具获取；"
        "对于数学计算，请使用计算器工具。"
    )

    agent = create_agent(
        model,
        tools=tools,
        system_prompt=system_prompt,
    )

    print("【ReAct Agent 创建成功】")
    print(f"  模式: ReAct (Reasoning + Acting)")
    print(f"  工具列表:")
    for t in tools:
        print(f"    - {t.name}: {t.description.strip()}")
    print()

    # -------------------------------------------------------------------------
    # 第四步：运行 Agent，观察 ReAct 循环过程
    # -------------------------------------------------------------------------
    # 使用 stream() 可以逐步观察 Agent 的思考和行动过程

    questions = [
        "北京今天天气怎么样？适合户外活动吗？",
        "帮我算一下 (15 + 27) * 3 等于多少",
        "我想了解你们公司的地址和营业时间",
    ]

    for question in questions:
        print(f"{'─' * 50}")
        print(f"【用户提问】{question}")
        print(f"{'─' * 50}")

        inputs = {"messages": [HumanMessage(content=question)]}

        # 使用 stream 逐步观察 ReAct 过程
        step_count = 0
        for event in agent.stream(inputs, stream_mode="values"):
            messages = event.get("messages", [])
            if not messages:
                continue

            last_msg = messages[-1]
            step_count += 1

            # 识别消息类型，展示 ReAct 各阶段
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                # Thought + Action 阶段：模型决定调用工具
                for tc in last_msg.tool_calls:
                    print(f"  [Thought {step_count}] 需要调用工具获取信息")
                    print(f"  [Action  {step_count}] 调用 {tc['name']}({tc['args']})")
            elif hasattr(last_msg, "name") and last_msg.name:
                # Observation 阶段：工具返回结果
                print(f"  [Observation {step_count}] {last_msg.name} 返回: {last_msg.content}")
            elif hasattr(last_msg, "content") and last_msg.content:
                # Final Answer 阶段：模型给出最终回答
                print(f"  [Final Answer] {last_msg.content}")

        print(f"  (共 {step_count} 步完成)")
        print()



# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':

    # 运行示例
    # bind_tools_demo()
    # create_agent_demo()
    dynamic_model_selection_demo()
    # react_agent_demo()
