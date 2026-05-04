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
# LangGraph 入门课件
# =============================================================================
#
# 用途：教学演示 - 使用 LangGraph 构建 AI 智能体工作流
#
# 核心概念：
#   - StateGraph = "有状态的工作流图"
#   - Node = 图中的处理节点（函数）
#   - Edge = 节点之间的连接边
#   - State = 贯穿整个图的状态数据
#   - 支持循环、条件分支等复杂流程
# =============================================================================

# -----------------------------------------------------------------------------
# 运行前检查
# -----------------------------------------------------------------------------
# 1. 已安装依赖：pip install langgraph langchain-core langchain-openai
# 2. 已配置 .env 文件中的 ALIYUN_API_KEY
# -----------------------------------------------------------------------------


# =============================================================================
# 第一部分：理解 LangGraph 核心概念
# =============================================================================
"""
什么是 LangGraph？

📊 定义
   LangGraph = 基于图结构的状态机框架
   用于构建有状态的多参与者（Actor）AI 应用

🎯 为什么需要 LangGraph？

   普通的 LangChain 链（Chain）是线性的：
   A → B → C（单向、无循环）

   但真实的 AI 应用往往是循环的：
   - Agent 思考 → 调用工具 → 观察结果 → 再思考 → 再调用 → ...
   - 需要记忆历史状态
   - 需要根据条件走不同分支

   LangGraph 就能解决：
   ✅ 循环工作流（Agent 反复思考-行动）
   ✅ 条件分支（根据结果决定下一步）
   ✅ 状态管理（全程共享数据）
   ✅ 可视化调试（生成流程图）

💡 生活化比喻
   LangGraph = "地铁线路图"
   - 站点（Node）= 每个处理步骤
   - 线路（Edge）= 站点间的连接
   - 乘客（State）= 在图中传递的数据
   - 换乘站（Conditional Edge）= 根据条件选择不同线路
"""


# =============================================================================
# 示例 1: 最简图 - 两个节点的顺序执行
# =============================================================================

def simple_two_node_graph():
    """
    最简单的 LangGraph：两个节点顺序执行

    START → 问候 → 回应 → END
    """
    print("=" * 60)
    print("示例 1: 最简图 - 两个节点顺序执行")
    print("=" * 60)

    from typing import TypedDict
    from langgraph.graph import StateGraph, START, END

    # 1. 定义状态：图中传递的数据结构
    #    TypedDict 是 Python 的类型提示工具，LangGraph 用它来定义"贯穿整张图的状态"
    #    每个节点函数都可以读取和修改这个状态
    class GraphState(TypedDict):
        """图的状态：存储消息和步骤计数"""
        messages: list[str]
        step_count: int

    # 2. 定义节点函数：每个节点接收状态，返回状态更新
    #    ⚠️ 关键：节点函数的返回值不是"替换"整个状态，而是"合并"到状态中
    #    例如 return {"messages": [...]} 只会更新 messages 字段，step_count 不受影响
    def greet(state: GraphState):
        """节点 1：添加问候消息"""
        print("  [节点: greet] 添加问候语")
        return {
            "messages": ["你好！很高兴见到你！"],
            "step_count": 1
        }

    def respond(state: GraphState):
        """节点 2：添加回应消息
        这里演示了如何读取之前的状态（state["step_count"]）并在此基础上更新
        """
        print("  [节点: respond] 添加回应")
        return {
            "messages": ["你好！我也很高兴！"],
            "step_count": state["step_count"] + 1  # 读取上一步的值，+1
        }

    # 3. 构建图：连接节点
    #    StateGraph(GraphState)  — 创建一个带状态的图，状态类型是 GraphState
    #    .add_node("名字", 函数)  — 注册一个节点，"名字" 用于后续连线时引用
    #    .add_edge(A, B)         — 添加一条从 A 到 B 的固定边（无条件，一定执行）
    #    START                   — 图的入口常量，等价于"从这里开始"
    #    END                     — 图的出口常量，等价到这里结束
    #    .compile()              — 把图编译成可执行的对象
    graph = (
        StateGraph(GraphState)
        .add_node("greet", greet)       # 添加节点
        .add_node("respond", respond)
        .add_edge(START, "greet")       # 起点 → greet
        .add_edge("greet", "respond")   # greet → respond
        .add_edge("respond", END)       # respond → 终点
        .compile()
    )

    # 4. 执行图
    #    invoke() 的参数是初始状态，图会从这个初始状态开始按边依次执行节点
    #    返回值是最终的状态（所有节点执行完毕后的状态）
    print("【执行图】")
    result = graph.invoke({"messages": [], "step_count": 0})
    print(f"  最终消息: {result['messages']}")
    print(f"  总步骤数: {result['step_count']}")
    print()


# =============================================================================
# 示例 2: 条件分支 - 根据内容选择不同路径
# =============================================================================

def conditional_branch_graph():
    """
    带条件分支的图：根据消息内容走不同路径

    START → 分类 → (正面路径 / 负面路径) → END
    """
    print("=" * 60)
    print("示例 2: 条件分支 - 根据内容选择路径")
    print("=" * 60)

    from typing import TypedDict
    from langgraph.graph import StateGraph, START, END

    class SentimentState(TypedDict):
        text: str
        category: str
        reply: str

    def classify(state: SentimentState):
        """分类节点：简单判断情感倾向"""
        positive_words = ["好", "棒", "开心", "喜欢", "谢谢", "不错"]
        negative_words = ["差", "糟", "难过", "讨厌", "失望", "不好"]

        text = state["text"]
        if any(w in text for w in positive_words):
            category = "positive"
        elif any(w in text for w in negative_words):
            category = "negative"
        else:
            category = "neutral"

        print(f"  [节点: classify] 文本 '{text}' → 分类: {category}")
        return {"category": category}

    def positive_reply(state: SentimentState):
        """正面回复节点"""
        print("  [节点: positive_reply] 生成正面回复")
        return {"reply": "太好了！听到你这么说我很开心！😊"}

    def negative_reply(state: SentimentState):
        """负面回复节点"""
        print("  [节点: negative_reply] 生成安慰回复")
        return {"reply": "别担心，一切都会好起来的！有什么我可以帮你的吗？"}

    def neutral_reply(state: SentimentState):
        """中性回复节点"""
        print("  [节点: neutral_reply] 生成中性回复")
        return {"reply": "好的，我明白了。"}

    def route(state: SentimentState):
        """条件路由函数：根据分类结果决定下一步
        ⚠️ 这个函数的返回值是字符串，必须和下面 add_conditional_edges 映射中的 key 对应
        """
        cat = state["category"]
        if cat == "positive":
            return "positive_reply"
        elif cat == "negative":
            return "negative_reply"
        else:
            return "neutral_reply"

    # 构建图
    graph = (
        StateGraph(SentimentState)
        .add_node("classify", classify)
        .add_node("positive_reply", positive_reply)
        .add_node("negative_reply", negative_reply)
        .add_node("neutral_reply", neutral_reply)
        .add_edge(START, "classify")
        # 条件边（Conditional Edge）：这是 LangGraph 的核心概念之一
        # .add_conditional_edges(源节点, 路由函数, 映射字典)
        #
        # 执行过程：
        #   1. 当流程走到 "classify" 节点后，调用 route(state) 函数
        #   2. route 返回一个字符串，比如 "positive_reply"
        #   3. 在映射字典中查找这个字符串对应的目标节点
        #   4. 流程走向那个目标节点
        #
        # 映射字典的值也可以是 END，表示直接结束
        .add_conditional_edges("classify", route, {
            "positive_reply": "positive_reply",
            "negative_reply": "negative_reply",
            "neutral_reply": "neutral_reply",
        })
        .add_edge("positive_reply", END)
        .add_edge("negative_reply", END)
        .add_edge("neutral_reply", END)
        .compile()
    )

    # 测试不同输入
    print("【测试 1: 正面输入】")
    r1 = graph.invoke({"text": "今天天气真好，心情不错！", "category": "", "reply": ""})
    print(f"  回复: {r1['reply']}\n")

    print("【测试 2: 负面输入】")
    r2 = graph.invoke({"text": "今天的体验太差了，很失望。", "category": "", "reply": ""})
    print(f"  回复: {r2['reply']}\n")

    print("【测试 3: 中性输入】")
    r3 = graph.invoke({"text": "我明天要去开会。", "category": "", "reply": ""})
    print(f"  回复: {r3['reply']}\n")


# =============================================================================
# 示例 3: 循环图 - Agent 思考-行动-观察循环
# =============================================================================

def agent_loop_graph():
    """
    带循环的图：模拟 Agent 的思考-行动-观察循环

    START → LLM调用 → 是否有工具调用？
                        ├── 是 → 执行工具 → 回到LLM调用
                        └── 否 → END
    """
    print("=" * 60)
    print("示例 3: 循环图 - Agent 思考-行动-观察")
    print("=" * 60)

    from typing import TypedDict
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    from langchain_core.tools import tool
    from langgraph.graph import StateGraph, START, END

    # 定义工具
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

    tools = [add, multiply]
    # 把工具列表转成字典，方便通过工具名快速查找
    # 例如：tools_by_name["add"] → 拿到 add 这个工具对象
    tools_by_name = {t.name: t for t in tools}

    # 获取模型并绑定工具
    # .bind_tools(tools) 的作用：告诉模型"你有这些工具可以用"
    # 绑定后，模型在回复时会在 tool_calls 字段中列出它想调用的工具
    # 但这只是"列出"，实际执行需要我们在代码中处理
    model = model_untils.get_qwen_client()
    model_with_tools = model.bind_tools(tools)

    # 定义状态
    # 这里用 list 存消息历史，格式是 LangChain 的 Message 对象列表
    # 包括 HumanMessage（用户输入）、AIMessage（AI 回复）、ToolMessage（工具结果）
    class AgentState(TypedDict):
        messages: list

    def llm_node(state: AgentState):
        """LLM 节点：调用模型，让它决定是回复还是调用工具
        ⚠️ 关键：模型返回的是 AIMessage 对象
        - 如果模型直接回答，AIMessage.content 有文本内容
        - 如果模型要调工具，AIMessage.tool_calls 有工具调用列表
        """
        messages = state["messages"]
        print(f"  [节点: llm_node] 调用模型，当前消息数: {len(messages)}")

        response = model_with_tools.invoke(messages)
        print(f"  [节点: llm_node] 模型响应: {response.content if response.content else '(调用工具)'}")

        # 返回 {"messages": [response]} — 把模型的回复追加到消息列表中
        return {"messages": [response]}

    def tool_node(state: AgentState):
        """工具节点：执行模型请求的工具调用
        工作原理：
          1. 从消息列表最后一条（模型的 AIMessage）中取出 tool_calls
          2. 逐个执行对应的工具函数
          3. 把每个工具的执行结果包装成 ToolMessage
          4. 返回 ToolMessage 列表，追加到消息历史中
        """
        messages = state["messages"]
        last_message = messages[-1]  # 最后一条是模型的 AIMessage
        results = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]    # 工具名，如 "add"
            tool_args = tool_call["args"]    # 工具参数，如 {"a": 3, "b": 4}
            tool_id = tool_call["id"]        # 工具调用 ID，用于匹配哪个 ToolMessage 属于哪个调用

            # 通过工具名找到对应的工具函数并执行
            tool_func = tools_by_name[tool_name]
            result = tool_func.invoke(tool_args)

            # 创建工具响应消息 — ToolMessage 会告诉模型"你的工具调用得到了这个结果"
            results.append(ToolMessage(content=str(result), tool_call_id=tool_id))

        return {"messages": results}

    def should_use_tool(state: AgentState):
        """条件路由：检查模型是否要调用工具
        这是循环的关键 — 决定"继续循环"还是"跳出循环"
        - 有 tool_calls → 走 tool_node（执行工具后回到 llm_node）
        - 没有 tool_calls → 走 END（直接结束）
        """
        last_message = state["messages"][-1]
        has_tool_calls = hasattr(last_message, 'tool_calls') and last_message.tool_calls
        if has_tool_calls:
            print(f"  [路由] 模型要调用工具: {[tc['name'] for tc in last_message.tool_calls]}")
            return "tool_node"
        else:
            print(f"  [路由] 模型回复完成，无工具调用")
            return END

    # 构建图
    # 🔁 循环的关键：tool_node → llm_node 这条边形成了循环
    # 流程：llm_node → (有工具?) → tool_node → 回到 llm_node → (还有工具?) → ...
    # 直到某次 llm_node 的模型回复没有 tool_calls，should_use_tool 返回 END，循环终止
    graph = (
        StateGraph(AgentState)
        .add_node("llm_node", llm_node)
        .add_node("tool_node", tool_node)
        .add_edge(START, "llm_node")
        # 条件边：LLM 决定是调用工具还是结束
        .add_conditional_edges("llm_node", should_use_tool, ["tool_node", END])
        # 工具执行后回到 LLM — 这条边形成了循环！
        .add_edge("tool_node", "llm_node")
        .compile()
    )

    # 运行 Agent
    print("【运行 Agent: 3加4等于多少，再乘以2】")
    result = graph.invoke({
        "messages": [HumanMessage(content="3加4等于多少？然后结果乘以2。")]
    })


    final_msg = result["messages"][-1]
    print(f"\n  【最终回复】{final_msg.content}")
    print(f"  【消息总数】{len(result['messages'])}")
    print()



# =============================================================================
# 示例 5: 实用案例 - 搜索问答 Agent
# =============================================================================

def search_qa_agent():
    """
    实用案例：构建一个能搜索和计算的问答 Agent

    工具：
    - search_knowledge: 模拟知识库搜索
    - calculate: 数学计算
    - get_date: 获取当前日期
    """
    print("=" * 60)
    print("示例 5: 实用案例 - 搜索问答 Agent")
    print("=" * 60)

    from typing import TypedDict
    from langchain_core.messages import HumanMessage, ToolMessage
    from langchain_core.tools import tool
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langgraph.graph import StateGraph, START, END
    from datetime import datetime

    # 知识库搜索工具
    @tool
    def search_knowledge(query: str) -> str:
        """在知识库中搜索相关信息"""
        kb = {
            "公司": "我们公司是一家专注于 AI 技术的高科技企业。",
            "产品": "我们的核心产品是智能助手和自动化解决方案。",
            "福利": "公司提供五险一金、带薪年假、弹性工作制等福利。",
            "地址": "公司位于北京市海淀区中关村软件园。",
        }
        for key, value in kb.items():
            if key in query:
                print(f"  [工具: search_knowledge] 匹配关键词 '{key}'")
                return value
        print(f"  [工具: search_knowledge] 未找到匹配")
        return f"知识库中未找到关于'{query}'的信息。"

    # 计算工具
    @tool
    def calculate(expression: str) -> str:
        """执行数学计算"""
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            print(f"  [工具: calculate] {expression} = {result}")
            return str(result)
        except Exception as e:
            return f"计算错误: {e}"

    # 日期工具
    @tool
    def get_date() -> str:
        """获取当前日期和时间"""
        now = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
        print(f"  [工具: get_date] {now}")
        return now

    tools = [search_knowledge, calculate, get_date]
    # 工具字典：通过工具名快速查找
    tools_by_name = {t.name: t for t in tools}

    # 获取模型并绑定工具
    model = model_untils.get_qwen_client()
    model_with_tools = model.bind_tools(tools)

    # 添加系统提示词
    # ChatPromptTemplate 的作用：把系统提示 + 用户消息组合成完整的 prompt
    # MessagesPlaceholder("messages") 是占位符，运行时会被实际的 messages 列表替换
    system_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个智能问答助手。你可以搜索知识库、执行计算和查询日期。请根据用户的问题合理使用工具。"),
        MessagesPlaceholder("messages"),
    ])

    class QAState(TypedDict):
        messages: list

    def llm_call(state: QAState):
        """LLM 节点：拼接 prompt 后调用模型
        1. 从状态中取出消息历史
        2. 用 ChatPromptTemplate 把系统提示 + 消息历史合并
        3. 调用模型（带工具）
        4. 把模型的回复追加到消息列表
        """
        messages = state["messages"]
        # format_messages 会把 MessagesPlaceholder 替换成实际的 messages 列表
        full_messages = system_prompt.format_messages(messages=messages)
        response = model_with_tools.invoke(full_messages)
        return {"messages": [response]}

    def tool_executor(state: QAState):
        """工具执行节点：和示例 3 的 tool_node 逻辑相同
        遍历模型的 tool_calls，执行对应工具，返回 ToolMessage 列表
        """
        last_msg = state["messages"][-1]
        results = []
        for tc in last_msg.tool_calls:
            tool_func = tools_by_name[tc["name"]]
            result = tool_func.invoke(tc["args"])
            results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        return {"messages": results}

    def route_check(state: QAState):
        """路由检查：和示例 3 的 should_use_tool 逻辑相同
        有工具调用 → 走 tool；没有 → 走 END
        """
        last_msg = state["messages"][-1]
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            return "tool"
        return END

    # 图结构：和示例 3 完全一致的 ReAct Agent 模式
    # llm → (需要工具?) → tool → 回到 llm → ... → END
    graph = (
        StateGraph(QAState)
        .add_node("llm", llm_call)
        .add_node("tool", tool_executor)
        .add_edge(START, "llm")
        .add_conditional_edges("llm", route_check, ["tool", END])
        .add_edge("tool", "llm")
        .compile()
    )

    # 测试不同问题
    tests = [
        "公司的福利有哪些？",
        "计算 123 乘以 456 等于多少",
        "今天几号？",
    ]

    for q in tests:
        print(f"【问题】{q}")
        result = graph.invoke({"messages": [HumanMessage(content=q)]})
        final = result["messages"][-1]
        if hasattr(final, 'content') and final.content:
            print(f"【回答】{final.content}")
        print()


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  LangGraph 入门 - 构建 AI 智能体工作流")
    print("  说明：使用图结构构建有状态的 AI 应用")
    print("=" * 70 + "\n")

    print("【运行前检查】")
    print("  1. 已安装依赖：pip install langgraph langchain-core langchain-openai")
    print("  2. 已配置 .env 文件中的 ALIYUN_API_KEY")
    print()

    # 取消注释以运行不同示例
    # simple_two_node_graph()
    # conditional_branch_graph()
    # agent_loop_graph()
    search_qa_agent()

    print("=" * 70)
    print("  接下来探索：将 langgraph_examples 目录扩展为完整的 Agent 项目")
    print("=" * 70 + "\n")
