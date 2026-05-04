import sys
import os
import io
# 设置标准输出编码为 UTF-8，避免 Windows 控制台乱码
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import model_untils


# =============================================================================
# LangGraph 工作流与智能体 - 官方文档重点归纳与演示
# =============================================================================
# 参考：https://langchain-doc.cn/v1/python/langgraph/workflows-agents.html
#
# 核心概念总结：
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  工作流（Workflow）vs 智能体（Agent）                                     │
# ├─────────────────────────────────────────────────────────────────────────┤
# │                                                                         │
# │  工作流 = 预定义的执行路径                                               │
#    问题和解决方案可预测，按特定顺序执行                                     │
#    包含：条件分支、循环、并行                                               │
#                                                                         │
#  智能体 = 自主决策的执行体                                                 │
#    Agent 自己决定使用哪些工具、何时结束                                     │
#    核心循环：思考 → 选择工具 → 执行 → 观察 → 再思考                         │
#                                                                         │
# └─────────────────────────────────────────────────────────────────────────┘
#
# 五大常见工作流模式：
#   1. 提示链（Prompt Chain）— 顺序执行 + 条件分支
#   2. 并行化（Parallelization）— 多个节点同时执行后聚合
#   3. 路由（Routing）— LLM 判断类型后选择不同路径
#   4. 协调器-工作者（Orchestrator-Worker）— 先规划，再分配任务
#   5. 评估器-优化器（Evaluator-Optimizer）— 生成 → 评估 → 不满意则重试
#
# 智能体核心模式（ReAct Agent）：
#   LLM 节点 → 检查是否有 tool_calls → 有则执行工具 → 回到 LLM → 直到无 tool_calls
# =============================================================================

# -----------------------------------------------------------------------------
# 运行前检查
# -----------------------------------------------------------------------------
# 1. 已安装依赖：pip install langgraph langchain-core
# 2. 已配置 .env 文件中的 ALIYUN_API_KEY
# -----------------------------------------------------------------------------


# =============================================================================
# 重点一：提示链（Prompt Chain）+ 条件分支
# =============================================================================
"""
模式说明：
  工作流中的某些步骤按顺序执行，但中间可以包含条件判断，
  根据中间结果决定是否走"改进"路径。

  官方例子：写笑话 → 评估笑点 → 需要改进则重写 → 最终润色 → 结束

  流程图：
    START → generate_joke → check_punchline → (需要改进?) → improve_joke → polish_joke → END
                                                          ↘ (不需要) ──────────┘

  关键点：
    - 每个节点函数返回的是"状态更新"（只更新部分字段）
    - 条件路由函数返回字符串，映射到对应的节点名
    - 这是"确定性流程 + 条件分支"的典型模式
"""


def prompt_chain_demo():
    """
    提示链：生成笑话并评估质量，必要时改进

    演示点：
      - 顺序执行 + 条件分支
      - 结构化输出（用 Pydantic 模型让 LLM 返回固定格式）
      - add_conditional_edges 的用法
    """
    print("=" * 60)
    print("重点一：提示链 + 条件分支")
    print("=" * 60)

    from typing import TypedDict
    from typing_extensions import Literal
    from pydantic import BaseModel, Field
    from langgraph.graph import StateGraph, START, END
    from langchain_core.messages import HumanMessage

    # -------------------------------------------------------------------------
    # 1. 结构化输出：让 LLM 返回固定格式的 JSON
    # -------------------------------------------------------------------------
    # .with_structured_output() 的作用：
    #   要求 LLM 按照 Pydantic 模型的格式返回数据，而不是自由文本
    #   底层原理：自动在 prompt 中加入格式说明，并解析返回的 JSON

    class JokeEvaluation(BaseModel):
        """判断笑话是否有趣的结构化输出"""
        needs_improvement: bool = Field(description="如果笑话需要改进返回 true")
        reason: str = Field(description="评价原因")

    model = model_untils.get_qwen_client()
    if model is None:
        print("【跳过】未配置阿里云 API Key，无法运行此示例")
        return

    evaluator = model.with_structured_output(JokeEvaluation)

    # -------------------------------------------------------------------------
    # 2. 定义状态
    # -------------------------------------------------------------------------
    class JokeState(TypedDict):
        topic: str          # 笑话主题
        joke: str           # 当前笑话内容
        needs_improvement: bool  # 是否需要改进
        reason: str         # 评估原因
        final_joke: str     # 最终笑话

    # -------------------------------------------------------------------------
    # 3. 定义节点函数
    # -------------------------------------------------------------------------

    def generate_joke(state: JokeState):
        """生成笑话节点"""
        print(f"  [节点: generate_joke] 为主题 '{state['topic']}' 生成笑话")
        response = model.invoke(f"写一个关于 {state['topic']} 的简短中文笑话")
        return {"joke": response.content}

    def evaluate_joke(state: JokeState):
        """评估节点：使用结构化输出判断笑话质量"""
        print(f"  [节点: evaluate_joke] 评估笑话质量")
        evaluation: JokeEvaluation = evaluator.invoke(f"评价这个笑话是否有趣：{state['joke']}")
        print(f"    需要改进: {evaluation.needs_improvement}")
        print(f"    原因: {evaluation.reason}")
        return {
            "needs_improvement": evaluation.needs_improvement,
            "reason": evaluation.reason,
        }

    def improve_joke(state: JokeState):
        """改进节点：根据反馈重新生成"""
        print(f"  [节点: improve_joke] 根据反馈改进笑话")
        response = model.invoke(
            f"改进这个笑话，使其更有趣。原笑话：{state['joke']}，反馈：{state['reason']}"
        )
        return {"joke": response.content}

    def polish_joke(state: JokeState):
        """润色节点：最终优化"""
        print(f"  [节点: polish_joke] 润色最终笑话")
        response = model.invoke(f"润色这个笑话，使其更加完美：{state['joke']}")
        return {"final_joke": response.content}

    def should_improve(state: JokeState):
        """路由函数：根据评估结果决定下一步"""
        if state["needs_improvement"]:
            print(f"  [路由] 需要改进 → 走 improve_joke 路径")
            return "improve_joke"
        else:
            print(f"  [路由] 无需改进 → 直接走 polish_joke")
            return "polish_joke"

    # -------------------------------------------------------------------------
    # 4. 构建图
    # -------------------------------------------------------------------------
    workflow = (
        StateGraph(JokeState)
        .add_node("generate_joke", generate_joke)
        .add_node("evaluate_joke", evaluate_joke)
        .add_node("improve_joke", improve_joke)
        .add_node("polish_joke", polish_joke)
        .add_edge(START, "generate_joke")
        .add_edge("generate_joke", "evaluate_joke")
        # 条件边：根据 should_improve 的返回值决定走哪条路
        .add_conditional_edges("evaluate_joke", should_improve, {
            "improve_joke": "improve_joke",
            "polish_joke": "polish_joke",
        })
        .add_edge("improve_joke", "polish_joke")  # 改进后也必须润色
        .add_edge("polish_joke", END)
        .compile()
    )

    # -------------------------------------------------------------------------
    # 5. 执行
    # -------------------------------------------------------------------------
    result = workflow.invoke({"topic": "程序员", "joke": "", "needs_improvement": False, "reason": "", "final_joke": ""})
    print(f"\n【最终笑话】{result['final_joke']}")
    print()


# =============================================================================
# 重点二：路由（Routing）— 基于 LLM 判断选择不同路径
# =============================================================================
"""
模式说明：
  先让 LLM 判断输入的"类型"，然后根据类型走不同的处理路径。

  官方例子：接收问题 → LLM 判断是技术问题/哲学问题/创意问题 → 对应角色回答

  流程图：
    START → router → (判断问题类型)
                         ├── "technical"   → technical_answer → END
                         ├── "philosophical" → philosophical_answer → END
                         └── "creative"    → creative_answer → END

  关键点：
    - 路由节点使用结构化输出判断类型
    - 条件边的映射字典把字符串映射到不同的回答节点
    - 所有分支最终都走到 END（不走回头路）
    - 这和 01_quickstart.py 中示例 2 的情感分类类似，但这里用 LLM 做判断
"""


def routing_demo():
    """
    路由：LLM 判断问题类型后选择对应回答路径

    演示点：
      - 结构化输出 + Literal 类型约束
      - 条件路由的三种分支
      - 路由决策由 LLM 完成（非硬编码关键词匹配）
    """
    print("=" * 60)
    print("重点二：路由 — 基于 LLM 判断选择路径")
    print("=" * 60)

    from typing import TypedDict
    from typing_extensions import Literal
    from pydantic import BaseModel, Field
    from langgraph.graph import StateGraph, START, END
    from langchain_core.messages import HumanMessage

    model = model_untils.get_qwen_client()
    if model is None:
        print("【跳过】未配置阿里云 API Key，无法运行此示例")
        return

    # -------------------------------------------------------------------------
    # 1. 结构化路由：让 LLM 判断问题类型
    # -------------------------------------------------------------------------
    # Literal["technical", "philosophical", "creative"] 约束 LLM 只能从这三个值中选择
    # 这是结构化输出的关键作用 — 把 LLM 的自由文本输出变成"枚举选择"

    class RouteDecision(BaseModel):
        """路由决策：判断问题属于哪种类型"""
        question_type: Literal["technical", "philosophical", "creative"] = Field(
            description="问题类型：technical(技术问题)、philosophical(哲学问题)、creative(创意问题)"
        )

    router = model.with_structured_output(RouteDecision)

    # -------------------------------------------------------------------------
    # 2. 定义状态
    # -------------------------------------------------------------------------
    class RoutingState(TypedDict):
        question: str       # 用户的问题
        question_type: str  # LLM 判断的类型
        answer: str         # 最终回答

    # -------------------------------------------------------------------------
    # 3. 定义节点
    # -------------------------------------------------------------------------

    def router_node(state: RoutingState):
        """路由节点：让 LLM 判断问题类型"""
        print(f"  [节点: router] 判断问题类型: '{state['question']}'")
        decision: RouteDecision = router.invoke(f"判断这个问题的类型：{state['question']}")
        q_type = decision.question_type
        print(f"    判断结果: {q_type}")
        return {"question_type": q_type}

    def technical_answer(state: RoutingState):
        """技术专家回答"""
        print(f"  [节点: technical_answer] 以技术专家身份回答")
        response = model.invoke(f"以技术专家的身份，简洁专业地回答：{state['question']}")
        return {"answer": response.content}

    def philosophical_answer(state: RoutingState):
        """哲学家回答"""
        print(f"  [节点: philosophical_answer] 以哲学家身份回答")
        response = model.invoke(f"以哲学家的身份，深入思辨地回答：{state['question']}")
        return {"answer": response.content}

    def creative_answer(state: RoutingState):
        """创意专家回答"""
        print(f"  [节点: creative_answer] 以创意专家身份回答")
        response = model.invoke(f"以创意专家的身份，富有想象力地回答：{state['question']}")
        return {"answer": response.content}

    def route_decision(state: RoutingState):
        """路由函数：根据 question_type 决定走哪个回答节点"""
        return state["question_type"]

    # -------------------------------------------------------------------------
    # 4. 构建图
    # -------------------------------------------------------------------------
    workflow = (
        StateGraph(RoutingState)
        .add_node("router", router_node)
        .add_node("technical_answer", technical_answer)
        .add_node("philosophical_answer", philosophical_answer)
        .add_node("creative_answer", creative_answer)
        .add_edge(START, "router")
        # 条件边：LLM 判断的问题类型 → 对应的回答节点
        .add_conditional_edges("router", route_decision, {
            "technical": "technical_answer",
            "philosophical": "philosophical_answer",
            "creative": "creative_answer",
        })
        .add_edge("technical_answer", END)
        .add_edge("philosophical_answer", END)
        .add_edge("creative_answer", END)
        .compile()
    )

    # -------------------------------------------------------------------------
    # 5. 测试不同类型的问题
    # -------------------------------------------------------------------------
    questions = [
        "Python 中的 GIL 是什么？",                # → technical
        "人生的意义是什么？",                       # → philosophical
        "如果月亮是一块巨大的奶酪，世界会怎样？",  # → creative
    ]

    for q in questions:
        print(f"【问题】{q}")
        result = workflow.invoke({"question": q, "question_type": "", "answer": ""})
        print(f"【回答】{result['answer']}")
        print()


# =============================================================================
# 重点三：智能体（Agent）— ReAct 循环模式
# =============================================================================
"""
模式说明：
  这是 LangGraph 最核心的 Agent 模式，也是和 01_quickstart.py 示例 3 一致的架构。

  流程：
    START → llm_call → (有 tool_calls?) → tool_node → 回到 llm_call
                          ↘ (无) ──────────────────────→ END

  官方例子：数学计算 Agent（加法、乘法、除法工具）

  关键组件：
    1. bind_tools() — 把工具"告诉"模型，让它知道可以用什么
    2. llm_call 节点 — 调用带工具的模型，返回 AIMessage（可能含 tool_calls）
    3. tool_node 节点 — 执行 AIMessage 中的 tool_calls，返回 ToolMessage
    4. should_continue 路由 — 检查是否有 tool_calls，决定继续还是结束

  消息流转：
    用户输入 → LLM → AIMessage(tool_calls=[...]) → 执行工具 → ToolMessage(...) →
    LLM 看到 ToolMessage → 再次推理 → 最终 AIMessage(content="...") → END
"""


def agent_react_demo():
    """
    ReAct Agent：能使用工具的智能体

    演示点：
      - bind_tools 绑定工具到模型
      - AIMessage.tool_calls 的结构和含义
      - 工具执行 + ToolMessage 返回结果给模型
      - 条件路由实现循环（ReAct 循环）
    """
    print("=" * 60)
    print("重点三：智能体 — ReAct 循环模式")
    print("=" * 60)

    from typing import TypedDict
    from langchain_core.messages import HumanMessage, ToolMessage
    from langchain_core.tools import tool
    from langgraph.graph import StateGraph, START, END

    # -------------------------------------------------------------------------
    # 1. 定义工具
    # -------------------------------------------------------------------------

    @tool
    def add(a: int, b: int) -> int:
        """计算两个数的和"""
        result = a + b
        print(f"  [工具: add] {a} + {b} = {result}")
        return result

    @tool
    def multiply(a: int, b: int) -> int:
        """计算两个数的乘积"""
        result = a * b
        print(f"  [工具: multiply] {a} × {b} = {result}")
        return result

    @tool
    def divide(a: int, b: int) -> float:
        """计算 a 除以 b 的商"""
        if b == 0:
            return "错误：除数不能为 0"
        result = a / b
        print(f"  [工具: divide] {a} ÷ {b} = {result}")
        return result

    # 工具列表和快速查找字典
    tools = [add, multiply, divide]
    tools_by_name = {t.name: t for t in tools}

    model = model_untils.get_qwen_client()
    if model is None:
        print("【跳过】未配置阿里云 API Key，无法运行此示例")
        return

    # bind_tools 的作用：告诉模型"你可以使用这些工具"
    # 模型在回复时会在 AIMessage.tool_calls 字段中列出想调用的工具
    model_with_tools = model.bind_tools(tools)

    # -------------------------------------------------------------------------
    # 2. 定义状态
    # -------------------------------------------------------------------------
    # 使用 LangGraph 内置的 MessagesState 来管理消息历史
    # MessagesState 内部结构：{"messages: list[BaseMessage]}
    from langgraph.graph import MessagesState

    # -------------------------------------------------------------------------
    # 3. 定义节点
    # -------------------------------------------------------------------------

    def llm_call(state: MessagesState):
        """LLM 节点：调用带工具的模型
        工作原理：
          1. 从状态中取出消息历史（包含用户之前的输入和工具的结果）
          2. 调用模型，让它决定是直接回答还是调用工具
          3. 把模型的回复（AIMessage）追加到消息列表
        """
        messages = state["messages"]
        print(f"  [节点: llm_call] 调用模型，消息数: {len(messages)}")

        response = model_with_tools.invoke(messages)

        # 检查模型是直接回答还是要调用工具
        if response.content:
            print(f"    模型直接回答: {response.content[:50]}...")
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"    模型要调用工具: {[tc['name'] for tc in response.tool_calls]}")

        return {"messages": [response]}

    def tool_node(state: MessagesState):
        """工具节点：执行模型请求的工具调用
        工作原理：
          1. 从消息列表最后一条（AIMessage）中取出 tool_calls
          2. 逐个执行对应的工具函数
          3. 把每个工具结果包装成 ToolMessage
          4. ToolMessage 会告诉模型"你的工具调用得到了这个结果"
        """
        messages = state["messages"]
        last_message = messages[-1]  # 最后一条是模型的 AIMessage
        results = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]    # 工具名
            tool_args = tool_call["args"]    # 工具参数
            tool_id = tool_call["id"]        # 调用 ID，用于匹配 ToolMessage

            tool_func = tools_by_name[tool_name]
            result = tool_func.invoke(tool_args)

            # ToolMessage 把工具执行结果返回给模型
            results.append(ToolMessage(content=str(result), tool_call_id=tool_id))

        return {"messages": results}

    def should_continue(state: MessagesState):
        """路由函数：决定是继续调用工具还是结束
        检查最后一条消息是否有 tool_calls：
          - 有 → 走 tool_node（执行工具后回到 llm_call）
          - 没有 → 走 END（模型已经给出了最终回答）
        """
        last_message = state["messages"][-1]
        has_tool_calls = hasattr(last_message, 'tool_calls') and last_message.tool_calls

        if has_tool_calls:
            tool_names = [tc["name"] for tc in last_message.tool_calls]
            print(f"  [路由] 需要调用工具: {tool_names} → 走 tool_node")
            return "tool_node"
        else:
            print(f"  [路由] 无需工具 → 结束")
            return END

    # -------------------------------------------------------------------------
    # 4. 构建图 — 经典的 ReAct Agent 循环
    # -------------------------------------------------------------------------
    # 循环的关键：tool_node → llm_call 这条边
    # 流程：llm_call → (有工具?) → tool_node → 回到 llm_call → (还有工具?) → ...
    # 直到某次 llm_call 的回复没有 tool_calls，路由返回 END，循环终止
    agent = (
        StateGraph(MessagesState)
        .add_node("llm_call", llm_call)
        .add_node("tool_node", tool_node)
        .add_edge(START, "llm_call")
        .add_conditional_edges("llm_call", should_continue, ["tool_node", END])
        .add_edge("tool_node", "llm_call")  # 工具执行后回到 LLM — 形成循环！
        .compile()
    )

    # -------------------------------------------------------------------------
    # 5. 测试
    # -------------------------------------------------------------------------
    questions = [
        "3 加 4 等于多少？",
        "5 乘以 6 等于多少？",
        "20 除以 4 等于多少？",
    ]

    for q in questions:
        print(f"【用户提问】{q}")
        result = agent.invoke({"messages": [HumanMessage(content=q)]})
        final_msg = result["messages"][-1]
        if hasattr(final_msg, 'content') and final_msg.content:
            print(f"【Agent 回答】{final_msg.content}")
        print()


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  LangGraph 工作流与智能体 - 官方文档重点归纳")
    print("  参考：https://langchain-doc.cn/v1/python/langgraph/workflows-agents.html")
    print("=" * 70 + "\n")

    print("【核心概念总结】")
    print("  工作流（Workflow）= 预定义的、可预测的执行路径")
    print("  智能体（Agent）  = 自主决策、反复思考-行动-观察的循环")
    print()

    print("【五大工作流模式】")
    print("  1. 提示链  — 顺序执行 + 条件分支（示例 1）")
    print("  2. 路由    — LLM 判断类型后选择路径（示例 2）")
    print("  3. 智能体  — ReAct 循环：思考→工具→观察→再思考（示例 3）")
    print()

    print("【运行前检查】")
    print("  1. 已安装依赖：pip install langgraph langchain-core pydantic")
    print("  2. 已配置 .env 文件中的 ALIYUN_API_KEY")
    print()

    # 运行所有示例
    # prompt_chain_demo()
    # routing_demo()
    agent_react_demo()

    print("=" * 70)
    print("  完整工作流模式还包括：")
    print("    - 并行化（Parallelization）— 多个节点从 START 并行出发后聚合")
    print("    - 协调器-工作者（Orchestrator-Worker）— 先规划再分配任务（Send API）")
    print("    - 评估器-优化器（Evaluator-Optimizer）— 生成→评估→不满意则重试")
    print("    - 人在回路（Human-in-the-Loop）— 关键操作需人类审批（见 human_in_the_loop.py）")
    print("=" * 70 + "\n")
