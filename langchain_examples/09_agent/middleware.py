import sys
import os

from langchain_ollama import ChatOllama

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import model_untils

from langchain.agents import create_agent
from langchain.agents.middleware import (
    wrap_model_call,
    before_model,
    after_model,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage


# =============================================================================
# 中间件 Middleware
# =============================================================================
#
# 用途：教学演示 - 使用 LangChain 中间件控制并定制 Agent 执行流程
#
# 核心概念：
#   - 中间件 = 在 Agent 执行流程中插入自定义逻辑的"拦截器"
#   - 可以在模型调用、工具调用等步骤的前后添加钩子(hooks)
#
# 中间件能做什么？
#   - 监控(Monitor)：追踪 Agent 行为，日志记录、调试
#   - 修改(Modify)：转换提示词、工具选择和输出格式
#   - 控制(Control)：添加重试、回退和提前终止逻辑
#   - 强制执行(Enforce)：应用速率限制、安全防护、PII 检测
#
# 创建中间件的方式：
#   1. 基于装饰器(Decorator-based) - 适用于单钩子中间件，快速简单
#   2. 基于类(Class-based) - 适用于多钩子复杂中间件，功能更强
# =============================================================================

# -----------------------------------------------------------------------------
# 运行前检查
# -----------------------------------------------------------------------------
# 1. 已配置阿里云 API Key（.env 文件）
# -----------------------------------------------------------------------------


# =============================================================================
# 第一部分：理解中间件
# =============================================================================
"""
什么是中间件（Middleware）？

🔧 定义
   中间件 = Agent 执行流程中的"拦截器"
   在模型调用、工具调用等关键步骤的前后，插入自定义逻辑

📊 核心循环
   普通 Agent 循环：
     调用模型 → 模型选择工具 → 执行工具 → 循环或结束

   加入中间件后：
     [中间件] → 调用模型 → [中间件] → 模型选择工具 → [中间件] → 执行工具 → ...

🎯 两种钩子风格

   节点式(Node-style) - 在特定执行点顺序运行：
     @before_agent  - Agent 启动前（每次调用一次）
     @before_model  - 每次模型调用前
     @after_model   - 每次模型响应后
     @after_agent   - Agent 完成时

   包装式(Wrap-style) - 拦截执行并完全控制：
     @wrap_model_call - 每次模型调用周围（可重试、修改、短路）
     @wrap_tool_call  - 每次工具调用周围

💡 生活化比喻
   中间件 = "安检站"
   - 旅客（请求/响应）必须经过安检站
   - 安检站可以：记录日志、检查违禁品、修改行李、拒绝通行
   - 多个安检站按顺序工作，各司其职
"""


# =============================================================================
# 示例 1: 基于装饰器的中间件 - 监控与日志
# =============================================================================

def decorator_middleware_demo():
    """
    使用装饰器创建中间件：监控 Agent 的执行过程

    装饰器方式适合单钩子中间件，快速简单。

    演示三种装饰器：
      - @before_model：模型调用前记录状态
      - @after_model：模型响应后检查内容
      - @wrap_model_call：包装模型调用，添加计时功能
    """

    # -------------------------------------------------------------------------
    # 定义装饰器中间件
    # -------------------------------------------------------------------------

    # 1. @before_model - 节点式：模型调用前执行
    #    用途：日志记录、状态检查、前置验证
    @before_model
    def log_before_model(state: AgentState, runtime) -> dict | None:
        """在每次模型调用前打印当前消息数量。"""
        msg_count = len(state.get("messages", []))
        print(f"  [before_model] 即将调用模型，当前消息数: {msg_count}")
        return None  # 返回 None 表示不修改状态

    # 2. @after_model - 节点式：模型响应后执行
    #    用途：输出验证、内容过滤、提前终止
    @after_model(can_jump_to=["end"])
    def check_response(state: AgentState, runtime) -> dict | None:
        """检查模型响应，如果包含敏感词则提前终止。"""
        last_msg = state["messages"][-1]
        # 检测敏感词（示例）
        sensitive_words = ["BLOCKED", "禁止回答"]
        if hasattr(last_msg, "content") and last_msg.content:
            for word in sensitive_words:
                if word in last_msg.content:
                    print(f"  [after_model] 检测到敏感词 '{word}'，提前终止")
                    return {
                        "messages": [AIMessage("抱歉，我无法回答这个问题。")],
                        "jump_to": "end"  # 跳转到结束节点
                    }
        print(f"  [after_model] 模型响应检查通过")
        return None

    # 3. @wrap_model_call - 包装式：围绕模型调用执行
    #    用途：重试逻辑、计时、缓存、模型切换
    @wrap_model_call
    def timing_middleware(request: ModelRequest, handler) -> ModelResponse:
        """记录每次模型调用的耗时。"""
        import time
        start = time.time()
        # 调用实际的模型处理
        response = handler(request)
        elapsed = time.time() - start
        print(f"  [wrap_model_call] 模型调用耗时: {elapsed:.2f}秒")
        return response

    # -------------------------------------------------------------------------
    # 定义工具和模型
    # -------------------------------------------------------------------------

    @tool
    def get_weather(city: str) -> str:
        """查询指定城市的天气信息。"""
        weather_db = {
            "北京": "晴，25°C",
            "上海": "多云，28°C",
            "深圳": "晴，29°C",
        }
        return weather_db.get(city, f"暂无 {city} 的天气数据")

    model = model_untils.get_qwen_client()
    if model is None:
        print("【跳过】未配置阿里云 API Key，无法运行此示例")
        return

    # -------------------------------------------------------------------------
    # 创建带中间件的 Agent
    # -------------------------------------------------------------------------

    agent = create_agent(
        model,
        tools=[get_weather],
        system_prompt="你是一个有用的助手，请简洁回答。",
        middleware=[log_before_model, check_response, timing_middleware],
    )

    print("【Agent 创建成功 - 带监控中间件】")
    print(f"  中间件列表:")
    print(f"    - log_before_model  (@before_model - 调用前日志)")
    print(f"    - check_response    (@after_model - 响应检查)")
    print(f"    - timing_middleware (@wrap_model_call - 调用计时)")
    print()

    # 调用 Agent
    question = "北京今天天气怎么样？"
    print(f"【用户提问】{question}")
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    print(f"【Agent 回答】{result['messages'][-1].content}")
    print()


# =============================================================================
# 示例 2: 基于类的中间件 - 重试与模型回退
# =============================================================================

def class_middleware_demo():
    """
    使用类创建中间件：实现重试逻辑和模型回退

    基于类的方式适合具有多个钩子或需要复杂配置的中间件。

    演示两个类中间件：
      - RetryMiddleware：模型调用失败时自动重试
      - FallbackMiddleware：主模型失败时自动切换到备用模型
    """

    from langchain.agents.middleware import AgentMiddleware
    from typing import Callable

    # -------------------------------------------------------------------------
    # 自定义类中间件 1：重试中间件
    # -------------------------------------------------------------------------
    # 包装式(wrap-style)：拦截模型调用，失败时自动重试

    class RetryMiddleware(AgentMiddleware):
        """模型调用重试中间件。

        当模型调用抛出异常时，自动重试指定次数。
        使用指数退避策略，避免频繁重试导致服务器压力过大。

        Args:
            max_retries: 最大重试次数（默认 3）
            backoff_factor: 退避因子，每次重试等待时间翻倍（默认 2.0）
        """

        def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0):
            super().__init__()
            self.max_retries = max_retries
            self.backoff_factor = backoff_factor

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
            """拦截模型调用，失败时自动重试。"""
            import time

            for attempt in range(self.max_retries):
                try:
                    response = handler(request)
                    if attempt > 0:
                        print(f"  [RetryMiddleware] 第 {attempt + 1} 次尝试成功")
                    return response
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        print(f"  [RetryMiddleware] 重试 {self.max_retries} 次后仍失败: {e}")
                        raise
                    # 指数退避：等待时间逐渐增加
                    wait_time = self.backoff_factor ** attempt
                    print(f"  [RetryMiddleware] 第 {attempt + 1} 次失败，{wait_time:.1f}秒后重试: {e}")
                    time.sleep(wait_time)

    # -------------------------------------------------------------------------
    # 自定义类中间件 2：模型回退中间件
    # -------------------------------------------------------------------------
    # 包装式(wrap-style)：主模型失败时切换到备用模型

    class FallbackMiddleware(AgentMiddleware):
        """模型回退中间件。

        当主模型调用失败时，自动切换到备用模型继续执行。

        Args:
            fallback_model: 备用模型实例
        """

        def __init__(self, fallback_model):
            super().__init__()
            self.fallback_model = fallback_model

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
            """主模型失败时切换到备用模型。"""
            try:
                return handler(request)
            except Exception as e:
                print(f"  [FallbackMiddleware] 主模型失败: {e}")
                print(f"  [FallbackMiddleware] 切换到备用模型...")
                request.model = self.fallback_model
                return handler(request)

    # -------------------------------------------------------------------------
    # 定义工具和模型
    # -------------------------------------------------------------------------

    @tool
    def calculator(expression: str) -> str:
        """计算数学表达式的结果。"""
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {e}"

    # primary_model = model_untils.get_qwen_client()
    primary_model = ChatOllama(model="qwen3.5:2b")
    if primary_model is None:
        print("【跳过】未配置阿里云 API Key，无法运行此示例")
        return

    # 备用模型（使用不同的模型名称）
    fallback_model = model_untils.get_qwen_client(model_name="qwen-turbo")

    # -------------------------------------------------------------------------
    # 创建带中间件的 Agent
    # -------------------------------------------------------------------------

    agent = create_agent(
        primary_model,
        tools=[calculator],
        system_prompt="你是一个计算助手，请使用计算器工具完成计算。",
        middleware=[
            RetryMiddleware(max_retries=3, backoff_factor=2.0),
            FallbackMiddleware(fallback_model=fallback_model),
        ],
    )

    print("【Agent 创建成功 - 带重试和回退中间件】")
    print(f"  中间件列表:")
    print(f"    - RetryMiddleware    (类 - 重试3次, 指数退避)")
    print(f"    - FallbackMiddleware (类 - 主模型失败时切换到 qwen-turbo)")
    print()

    # 调用 Agent
    question = "计算 (15 + 27) * 3 等于多少？"
    print(f"【用户提问】{question}")
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    print(f"【Agent 回答】{result['messages'][-1].content}")
    print()


# =============================================================================
# 示例 3: 使用预置中间件 - 摘要与调用限制
# =============================================================================

def builtin_middleware_demo():
    """
    使用 LangChain 预置中间件：摘要与调用限制

    LangChain 为常见用例提供了预构建的中间件，无需自己实现。

    演示两个预置中间件：
      - SummarizationMiddleware：对话过长时自动摘要
      - ModelCallLimitMiddleware：限制模型调用次数，防止无限循环
    """

    from langchain.agents.middleware import (
        SummarizationMiddleware,
        ModelCallLimitMiddleware,
    )

    # -------------------------------------------------------------------------
    # 定义工具
    # -------------------------------------------------------------------------

    @tool
    def get_weather(city: str) -> str:
        """查询指定城市的天气信息。"""
        weather_db = {
            "北京": "晴，25°C",
            "上海": "多云，28°C",
            "广州": "小雨，30°C",
            "深圳": "晴，29°C",
        }
        return weather_db.get(city, f"暂无 {city} 的天气数据")

    @tool
    def calculator(expression: str) -> str:
        """计算数学表达式的结果。"""
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {e}"

    model = model_untils.get_qwen_client()
    if model is None:
        print("【跳过】未配置阿里云 API Key，无法运行此示例")
        return

    # -------------------------------------------------------------------------
    # 预置中间件 1：摘要中间件
    # -------------------------------------------------------------------------
    # 当对话接近 token 限制时，自动对历史消息进行摘要
    # 适用于：长期对话、多轮对话、需要保留完整上下文的应用

    summarization = SummarizationMiddleware(
        model="openai:gpt-4o-mini",           # 用于生成摘要的模型（轻量模型即可）
        max_tokens_before_summary=4000,        # 超过 4000 token 时触发摘要
        messages_to_keep=10,                   # 摘要后保留最近 10 条消息
    )

    # -------------------------------------------------------------------------
    # 预置中间件 2：模型调用限制中间件
    # -------------------------------------------------------------------------
    # 限制模型调用次数，防止无限循环或成本过高
    # 适用于：生产部署中的成本控制、防止失控 Agent

    call_limit = ModelCallLimitMiddleware(
        run_limit=10,          # 每次运行最多 10 次模型调用
        thread_limit=50,       # 每个线程最多 50 次模型调用
        exit_behavior="end",   # 达到限制时优雅终止（而非抛出异常）
    )

    # -------------------------------------------------------------------------
    # 创建带预置中间件的 Agent
    # -------------------------------------------------------------------------

    agent = create_agent(
        model,
        tools=[get_weather, calculator],
        system_prompt="你是一个有用的助手，请简洁回答。",
        middleware=[summarization, call_limit],
    )


    # 调用 Agent
    questions = [
        "北京今天天气怎么样？",
        "计算 100 + 200 等于多少？",
    ]

    for question in questions:
        print(f"【用户提问】{question}")
        result = agent.invoke({"messages": [HumanMessage(content=question)]})
        print(f"【Agent 回答】{result['messages'][-1].content}")
        print()


# =============================================================================
# 示例 4: 人在回路 (Human-in-the-Loop) - 人类审批中间件
# =============================================================================

def human_in_the_loop_demo():
    """
    人在回路中间件：使用内置 HumanInTheLoopMiddleware 实现人类审批

    人在回路（Human-in-the-Loop, HITL）是指 AI 系统在执行关键操作前
    需要人类确认、审批或提供反馈，确保输出符合人类意图和安全要求。

    HumanInTheLoopMiddleware 允许你：
      - 对指定工具调用设置 interrupt（中断），等待人类决策
      - 支持三种决策：approve（批准）、edit（编辑）、reject（拒绝）
      - 对不需要审批的工具设为 False 自动放行

    应用场景：
      - 邮件发送：AI 起草，人类审核后才发出
      - 数据库操作：AI 建议 SQL，人类确认后才执行
      - API 调用：AI 准备请求参数，人类检查后才发送

    💡 生活化比喻
       人在回路 = "双人复核制"
       - 飞行员起飞前需要塔台批准
       - 财务大额转账需要主管签字
       - Agent 调用敏感工具需要人类点头
    """
    print("=" * 60)
    print("示例 4: 人在回路中间件 - HumanInTheLoopMiddleware")
    print("=" * 60)

    from langchain.agents.middleware import HumanInTheLoopMiddleware
    from langgraph.checkpoint.memory import InMemorySaver

    # -------------------------------------------------------------------------
    # 定义工具：模拟 "读取邮件" 和 "发送邮件"
    # -------------------------------------------------------------------------
    # 实际生产中，这些工具会连接真实的邮件 API

    @tool
    def read_email(email_id: str) -> str:
        """读取指定邮件的内容。"""
        email_db = {
            "001": "发件人: 老板 | 主题: 本周工作汇报 | 内容: 请于周五前提交本周工作总结...",
            "002": "发件人: HR | 主题: 年会通知 | 内容: 公司年会定于12月30日举行...",
        }
        return email_db.get(email_id, f"未找到邮件 {email_id}")

    @tool
    def send_email(to: str, subject: str, body: str) -> str:
        """发送一封邮件给指定收件人。"""
        print(f"\n  ✉️ [发送邮件] 正在发送至: {to}")
        print(f"     主题: {subject}")
        print(f"     正文: {body[:100]}...")
        return f"邮件已成功发送至 {to}"

    model = model_untils.get_qwen_client()
    if model is None:
        print("【跳过】未配置阿里云 API Key，无法运行此示例")
        return

    # -------------------------------------------------------------------------
    # 人在回路中间件配置
    # -------------------------------------------------------------------------
    # interrupt_on 字典定义每个工具的审批策略：
    #   - 设为 False = 自动批准，不需要人类干预
    #   - 设为字典 = 需要人类审批，allowed_decisions 定义允许的决策类型

    hitl = HumanInTheLoopMiddleware(
        interrupt_on={
            # 发送邮件需要审批：人类可以批准、编辑后批准、或拒绝
            "send_email": {
                "allowed_decisions": ["approve", "edit", "reject"],
            },
            # 读取邮件自动批准：不需要人类介入
            "read_email": False,
        }
    )

    # -------------------------------------------------------------------------
    # 创建带人在回路中间件的 Agent
    # -------------------------------------------------------------------------
    # 注意：HITL 需要 checkpointer 来保存中间状态，
    # 以便人类审批后 Agent 能从断点恢复继续执行

    agent = create_agent(
        model,
        tools=[read_email, send_email],
        system_prompt="你是一个邮件助手，可以帮用户读取邮件和发送邮件。",
        middleware=[hitl],
        checkpointer=InMemorySaver(),
    )

    print("【Agent 创建成功 - 带人在回路中间件】")
    print(f"  工具审批策略:")
    print(f"    - read_email  → 自动批准（无需审批）")
    print(f"    - send_email  → 需要审批（支持 approve / edit / reject）")
    print()

    # -------------------------------------------------------------------------
    # 场景 1：读取邮件 - 不需要审批，直接执行
    # -------------------------------------------------------------------------
    print("【场景 1：读取邮件（自动批准）】")
    result1 = agent.invoke({
        "messages": [HumanMessage(content="帮我读一下邮件 001")]
    }, config={"configurable": {"thread_id": "hitl_thread_1"}})
    print(f"【Agent 回答】{result1['messages'][-1].content}")
    print()

    # -------------------------------------------------------------------------
    # 场景 2：发送邮件 - 触发人类审批中断
    # -------------------------------------------------------------------------
    print("【场景 2：发送邮件（需要审批）】")
    result2 = agent.invoke({
        "messages": [HumanMessage(
            content="帮我给老板发一封邮件，主题写'工作进度'，内容写'本周完成了所有任务'"
        )]
    }, config={"configurable": {"thread_id": "hitl_thread_2"}})
    print(f"【Agent 回答】{result2['messages'][-1].content}")
    print()

    # -------------------------------------------------------------------------
    # 场景 3：演示如何处理审批中断 - 模拟人类批准
    # -------------------------------------------------------------------------
    print("【场景 3：模拟人类审批流程】")
    print("  说明：在实际应用中，当 Agent 触发中断时，你需要：")
    print("  1. 检查是否有待审批的 interrupt")
    print("  2. 人类做出决策（approve/edit/reject）")
    print("  3. 将决策结果传给 Agent，Agent 从断点继续")
    print()

    # 以下代码演示了审批流程的核心逻辑（不实际运行）：
    demo_code = '''
    # 完整的人在回路审批流程示例：
    from langgraph.types import Command

    config = {"configurable": {"thread_id": "my_thread"}}

    # 步骤 1：Agent 运行到需要审批的工具调用时会被中断
    result = agent.invoke({
        "messages": [HumanMessage(content="帮我给张三发邮件...")]
    }, config=config)

    # 步骤 2：检查是否有待处理的中断
    if result.get("__interrupt__"):
        for interrupt in result["__interrupt__"]:
            # interrupt.value 中包含 action_requests 和 review_configs
            action = interrupt.value["action_requests"][0]
            print(f"待审批工具: {action['name']}")
            print(f"调用参数: {action['args']}")

    # 步骤 3a：人类批准 → 继续执行
    result = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config
    )

    # 步骤 3b：人类拒绝 → 返回拒绝结果
    result = agent.invoke(
        Command(resume={"decisions": [{"type": "reject", "message": "内容不合适"}]}),
        config=config
    )

    # 步骤 3c：人类编辑后批准 → 修改参数再批准
    result = agent.invoke(
        Command(resume={"decisions": [{
            "type": "edit",
            "edited_action": {
                "name": "send_email",
                "args": {"to": "boss@company.com", "subject": "修正后的主题", "body": "修改后的内容"}
            }
        }]}),
        config=config
    )
    '''
    print("  审批流程代码示例：")
    print(demo_code)
    print()


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':


    # 运行示例
    # decorator_middleware_demo()
    # class_middleware_demo()
    # builtin_middleware_demo()
    human_in_the_loop_demo()
