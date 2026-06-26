# LangGraph 练习与测试试卷

> 覆盖模块：01_introduction → 02_state_and_branching → 03_agent_loop → 04_workflows
> 
> 试卷分为三大部分：
> - **A 卷：基础练习**（学完即练，巩固概念）
> - **B 卷：编程实操**（动手写代码，检验应用能力）
> - **C 卷：综合测试**（闭卷考试，检验整体理解）

---

## A 卷：基础练习（学完即练）

### 练习一：核心概念填空（对应 01_introduction）

1. LangGraph 最核心的四个概念是：________（整张地铁图）、________（地铁站）、________（轨道）、________（乘客/数据）。

2. `StateGraph(GraphState).add_node("greet", greet).add_edge(START, "greet")` 中：
   - `add_node` 的第一个参数 `"greet"` 是节点的 _______
   - `add_node` 的第二个参数 `greet` 是节点的 _______
   - `add_edge(START, "greet")` 表示流程从 _______ 流向 _______

3. `compile()` 的比喻是 _______，它的作用是让线路图 _______。

4. 节点函数返回的是"状态 _______"，不是"状态 _______"——只会更新你返回的字段。

5. `graph.invoke({"messages": [], "step_count": 0})` 中，传入的字典是图的 _______。

### 练习二：概念匹配（对应 01_introduction / what_is_langgraph.py）

将左侧概念与右侧描述连线：

| 概念 | 描述 |
|------|------|
| StateGraph | A. 根据条件选择不同线路的换乘站 |
| Node | B. 数据在图中传递时携带的结构 |
| Edge | C. 让线路图开通运营的编译方法 |
| State | D. 每个处理步骤，像地铁站 |
| Conditional Edge | E. 站点间的连接，像轨道 |
| compile() | F. 整张地铁线路图 |

### 练习三：判断题（对应全部模块）

判断以下说法是否正确，错误的请说明原因：

1. ( ) `add_edge("A", "B")` 表示只有满足特定条件时才会从 A 走到 B。
2. ( ) 节点函数可以只返回状态中的部分字段，未返回的字段保持不变。
3. ( ) `add_conditional_edges` 的路由函数返回的是目标节点的名称字符串。
4. ( ) ReAct Agent 循环中，`tool_node → llm_call` 这条边形成了循环。
5. ( ) `create_react_agent()` 是 LangGraph 中完全独立于 StateGraph 的另一种架构。
6. ( ) 工作流（Workflow）和智能体（Agent）的区别在于：工作流是预定义路径，Agent 自主决策。
7. ( ) `Send()` API 可以实现 1 对 N 的并行扇出。
8. ( ) 评估器-优化器模式和提示链模式完全相同。
9. ( ) `with_structured_output()` 配合 Pydantic 模型可以让 LLM 返回结构化 JSON。
10. ( ) `Annotated[list, add_messages]` 的作用是自动追加消息到列表中。

### 练习四：流程排序（对应各模块）

以下代码片段的执行顺序是什么？请按 1→2→3→... 排序：

**题目 A（条件分支）：**
```python
.add_edge(START, "classify")
.add_conditional_edges("classify", route, {...})
.add_edge("positive_reply", END)
.add_node("classify", classify)
.add_node("positive_reply", positive_reply)
```
正确顺序：____ → ____ → ____ → ____ → ____

**题目 B（ReAct Agent）：**
```
用户提问 → LLM 推理 → （需要工具?）→ 执行工具 → 观察结果 → ?
```
如果 LLM 不再需要工具，流程走向 ______；如果还需要工具，流程走向 ______。

### 练习五：代码阅读（对应 02_state_and_branching）

阅读以下代码，回答问题：

```python
class SentimentState(TypedDict):
    text: str
    category: str
    reply: str

def classify(state: SentimentState):
    positive_words = ["好", "棒", "开心"]
    text = state["text"]
    if any(w in text for w in positive_words):
        category = "positive"
    else:
        category = "neutral"
    return {"category": category}

def route(state: SentimentState):
    cat = state["category"]
    if cat == "positive":
        return "positive_reply"
    else:
        return "neutral_reply"

graph = (
    StateGraph(SentimentState)
    .add_node("classify", classify)
    .add_node("positive_reply", lambda s: {"reply": "开心！"})
    .add_node("neutral_reply", lambda s: {"reply": "好的。"})
    .add_edge(START, "classify")
    .add_conditional_edges("classify", route, {
        "positive_reply": "positive_reply",
        "neutral_reply": "neutral_reply",
    })
    .add_edge("positive_reply", END)
    .add_edge("neutral_reply", END)
    .compile()
)

result = graph.invoke({"text": "今天真棒！", "category": "", "reply": ""})
```

1. `classify` 节点返回 `{"category": "positive"}` 后，`category` 字段的值变为 ______，`text` 字段的值变为 ______。
2. 路由函数 `route` 返回的字符串是 ______。
3. 最终 `result["reply"]` 的值是 ______。
4. 如果输入 `{"text": "我明天去开会。", ...}`，流程会经过哪些节点？______

---

## B 卷：编程实操（动手写代码）

> 要求：在 `langgraph_examples/practice/` 目录下完成以下编程任务。

### 实操一：最简三节点图（难度 ★）

**任务：** 创建一个三节点的顺序执行图，实现"问候 → 翻译 → 计数"的流程。

**要求：**
- 定义 State 包含 `text: str` 和 `count: int`
- 节点1（greet）：生成一句中文问候语
- 节点2（translate）：在问候语后追加 "（Hello!）"
- 节点3（count）：计算消息的字符数
- 绘制图为 PNG 保存到 `images/practice_simple.png`

### 实操二：智能客服路由（难度 ★★）

**任务：** 创建一个条件分支图，模拟智能客服系统根据用户问题的情感倾向返回不同回复。

**要求：**
- State 包含 `question: str`、`sentiment: str`、`response: str`
- 分类节点：判断问题的情感（正面/负面/中性），规则：
  - 包含"谢谢"、"好的"、"没问题" → positive
  - 包含"不行"、"太差"、"投诉" → negative
  - 其他 → neutral
- 三个回答节点：分别生成正面感谢、负面安抚、中性回复
- 路由函数：根据 sentiment 分发到对应回答节点
- 至少用 3 个不同输入测试

### 实操三：数学计算 Agent（难度 ★★★）

**任务：** 创建一个 ReAct Agent，能够使用加减乘除工具回答数学问题。

**要求：**
- 定义 4 个工具：add、subtract、multiply、divide
- 使用 `@tool` 装饰器
- 用 `bind_tools()` 绑定工具到模型
- 手动构建 StateGraph（不使用 `create_react_agent()`）
- 实现经典的 ReAct 循环：llm_call → should_continue → tool_node → 回到 llm_call
- 测试用例：
  - `"10 减 3 等于多少？"`
  - `"2 乘以 8 再加 5 等于多少？"`
- 打印消息流转过程（每条消息的类型和内容）

### 实操四：create_react_agent() 对比实验（难度 ★★）

**任务：** 对同一个数学问题，分别用 `create_react_agent()` 和手动构建两种方式创建 Agent，对比结果。

**要求：**
- 定义相同的 3 个工具（add、multiply、divide）
- 方式 A：用 `create_react_agent(model, tools)` 一行创建
- 方式 B：手动 StateGraph 构建（如实操三）
- 用同一个输入 `"(5 + 3) × 2 - 4"` 测试两种方式
- 打印对比：消息数量、工具调用次数、最终答案

### 实操五：提示链 — 文章摘要生成（难度 ★★★）

**任务：** 创建一个提示链工作流：生成摘要 → 评估摘要质量 → 需要改进则重新生成 → 输出最终版。

**要求：**
- 使用 `with_structured_output()` 让 LLM 返回 Pydantic 模型（包含 `score: int`、`feedback: str`）
- State 包含 `topic: str`、`summary: str`、`score: int`、`feedback: str`、`final_summary: str`
- 四个节点：generate_summary → evaluate_summary → improve_summary（条件分支）→ polish_summary
- 如果评分 >= 7，跳过改进直接润色
- 如果评分 < 7，先改进再润色

### 实操六：并行搜索聚合（难度 ★★★）

**任务：** 使用 Send() API 实现一个并行搜索系统：对一个查询词，同时搜索 3 个数据源后聚合结果。

**要求：**
- 使用 `Send()` API 实现扇出（Fan-out）
- 三个搜索节点模拟不同数据源（可用预设列表模拟）
- 聚合节点汇总所有搜索结果，生成最终回答
- State 中 `results` 字段使用 `Annotated[list, operator.add]` 自动追加
- 用查询词 `"LangGraph Agent"` 测试
- 对比：如果改为顺序执行，需要几次 LLM 调用？并行执行节省了多少时间？

### 实操七：评估器-优化器 — 写作改进（难度 ★★★★）

**任务：** 创建一个评估器-优化器循环，让 AI 写一段文字（如自我介绍），然后自动评估和改进。

**要求：**
- 循环结构：generate → evaluate → (不达标?) → optimize → 回到 generate
- 评估器使用 `with_structured_output()` 返回 Pydantic 模型（score 1-10、is_good_enough、issues）
- 设置最大迭代次数为 3，评分 >= 7 达标则结束
- State 包含 `task`、`content`、`score`、`issues`、`iteration`、`max_iterations`、`history`
- 记录每次迭代的评分历史
- 额外：用纯逻辑（无 LLM）实现一个"猜数字"的评估器-优化器作为对照

---

## C 卷：综合测试（闭卷考试）

### 一、单选题（每题 2 分，共 20 分）

1. LangGraph 中，定义图状态的数据结构通常使用：
   - A. dataclass
   - B. TypedDict
   - C. namedtuple
   - D. dict

2. 以下哪个方法让 StateGraph 进入可执行状态？
   - A. `.build()`
   - B. `.run()`
   - C. `.compile()`
   - D. `.start()`

3. `add_conditional_edges` 的路由函数应该返回什么类型？
   - A. bool
   - B. 目标节点名称的字符串
   - C. 节点对象
   - D. int 索引

4. ReAct Agent 循环中，形成循环的关键边是：
   - A. START → llm_call
   - B. llm_call → tool_node
   - C. tool_node → llm_call
   - D. llm_call → END

5. `@tool` 装饰器的作用是：
   - A. 将函数注册为 LLM 可调用的工具
   - B. 将函数设为图的节点
   - C. 将函数绑定到状态
   - D. 将函数标记为路由

6. `create_react_agent()` 和手动构建 StateGraph 的关系是：
   - A. 完全不同的两种架构
   - B. create_react_agent() 是手动构建的封装快捷方式
   - C. 手动构建已弃用，只能用 create_react_agent()
   - D. create_react_agent() 不支持工具调用

7. 五大工作流模式中，"评估器-优化器"的特点是：
   - A. 线性执行，不走回头路
   - B. LLM 判断类型后选择路径
   - C. 循环迭代直到质量达标
   - D. 多个任务同时执行

8. `Send()` API 的主要用途是：
   - A. 发送邮件通知
   - B. 实现 1 对 N 的并行扇出
   - C. 在节点间传递错误信息
   - D. 序列化图结构

9. `with_structured_output()` 配合什么使用可以让 LLM 返回固定格式？
   - A. JSON 字符串
   - B. Pydantic BaseModel
   - C. dict 对象
   - D. XML Schema

10. 工作流（Workflow）和智能体（Agent）的核心区别是：
    - A. 工作流更快，Agent 更慢
    - B. 工作流用 LangChain，Agent 用 LangGraph
    - C. 工作流是预定义路径，Agent 自主决策
    - D. 没有区别

### 二、多选题（每题 3 分，共 15 分）

1. 以下哪些是 LangGraph 的核心元素？（多选）
   - A. StateGraph
   - B. Node
   - C. Edge
   - D. Prompt
   - E. State

2. 构建一个完整的 LangGraph 图，必须包含的步骤有：（多选）
   - A. 定义 State（TypedDict）
   - B. 创建 StateGraph 实例
   - C. 添加节点（add_node）
   - D. 添加边（add_edge）
   - E. 编译（compile）

3. ReAct Agent 中，消息流转涉及的消息类型有：（多选）
   - A. HumanMessage
   - B. AIMessage
   - C. ToolMessage
   - D. SystemMessage

4. 以下哪些场景适合用 Send() API？（多选）
   - A. 对多篇文档同时做摘要
   - B. 根据问题类型路由到不同专家
   - C. 同时查询多个数据源后合并
   - D. 对一个答案不满意则重新生成

5. 评估器-优化器模式中，防止无限循环的机制有：（多选）
   - A. 设置最大迭代次数
   - B. 设置质量阈值（分数达标即停止）
   - C. 使用随机数决定是否继续
   - D. 每次评估给出具体反馈

### 三、填空题（每空 2 分，共 20 分）

1. LangGraph 的核心比喻是"________"，其中站点比喻为 Node，线路比喻为 Edge，乘客比喻为 ________。

2. `Annotated[list, add_messages]` 中，`add_messages` 是一个 ________ 函数，作用是自动将新消息 ________ 到列表中。

3. `bind_tools()` 让模型知道有哪些 ________ 可用，模型会在 ________ 字段列出想调用的工具。

4. ReAct 的全称是 Reasoning + ________，流程是：思考 → ________ → 观察 → 再思考。

5. 五大工作流模式分别是：提示链、________、路由、________、评估器-优化器。

6. `Send("generate_joke", {"topics": [t]})` 中，`"generate_joke"` 是目标 ________，第二个参数是传给该分支的 ________。

7. 评估器-优化器模式和提示链模式的区别：提示链走 ________ 路径（只走一次），评估器-优化器走 ________ 路径（循环迭代）。

### 四、简答题（每题 5 分，共 25 分）

1. **简述 LangGraph 和 LangChain 的关系。** 它们是替代关系还是互补关系？各自适合什么场景？

2. **解释 ReAct Agent 的消息流转过程。** 从用户提问到最终回答，消息经历了哪些类型？每种类型代表什么？

3. **对比 `create_react_agent()` 和手动构建 StateGraph 两种 Agent 构建方式。** 各有什么优缺点？何时用哪种？

4. **解释 Send() API 的工作原理。** 它和固定边（Edge）、条件边（Conditional Edge）有什么区别？举一个实际使用场景。

5. **描述评估器-优化器模式的完整流程。** 画出流程图（用文字描述），并说明为什么要设置最大迭代次数和质量阈值。

### 五、代码分析题（每题 10 分，共 20 分）

**题目一：** 以下代码有什么问题？请指出并修正。

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class MyState(TypedDict):
    msg: str
    count: int

def step_a(state):
    return {"msg": "Hello", "count": 1}

def step_b(state):
    return {"msg": state["msg"] + " World"}

graph = StateGraph(MyState)
graph.add_node("a", step_a)
graph.add_node("b", step_b)
graph.add_edge("a", "b")
result = graph.invoke({"msg": "", "count": 0})
```

**题目二：** 分析以下 Agent 图的执行流程，回答后面问题。

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def llm_node(state):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def tool_node(state):
    last_msg = state["messages"][-1]
    results = []
    for tc in last_msg.tool_calls:
        result = tools_by_name[tc["name"]].invoke(tc["args"])
        results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": results}

def router(state):
    last_msg = state["messages"][-1]
    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        return "tool_node"
    return END

agent = (
    StateGraph(AgentState)
    .add_node("llm", llm_node)
    .add_node("tools", tool_node)
    .add_edge(START, "llm")
    .add_conditional_edges("llm", router, ["tools", END])
    .add_edge("tools", "llm")
    .compile()
)

result = agent.invoke({"messages": [HumanMessage("3+4=?")]})
```

1. 请画出该图的完整执行流程图（节点和边的关系）。
2. 如果 LLM 第一次回复包含 `tool_calls=[{"name": "add", "args": {"a":3, "b":4}}]`，流程怎么走？
3. 工具执行后返回 `ToolMessage(content="7")`，接下来流程怎么走？
4. 如果 LLM 第二次回复不再有 `tool_calls`，流程走向哪里？
5. `add_edge("tools", "llm")` 这条边在整个流程中起到了什么作用？

---

## 参考答案

### A 卷参考答案

**练习一：**
1. StateGraph、Node、Edge、State
2. 名称、处理函数、起点站、greet 节点
3. 开通运营、可以运行（被 invoke）
4. 更新、替换
5. 初始状态

**练习二：** F-D-E-B-A-C

**练习三：**
1. ×（固定边是无条件执行，条件边才是根据条件判断）
2. √
3. √
4. √
5. ×（create_react_agent() 底层就是 StateGraph，只是封装了）
6. √
7. √
8. ×（提示链是确定性路径走一次，评估器-优化器是循环迭代直到达标）
9. √
10. √

**练习四：**
- 题目 A：`.add_node("classify")` → `.add_node("positive_reply")` → `.add_edge(START, "classify")` → `.add_conditional_edges(...)` → `.add_edge("positive_reply", END)`
- 题目 B：不再需要工具 → END；还需要工具 → tool_node

**练习五：**
1. `"positive"`、`"今天真棒！"`（未被修改，保留原值）
2. `"positive_reply"`
3. `"开心！"`
4. classify → neutral_reply → END

### C 卷参考答案

**一、单选题：** 1-B  2-C  3-B  4-C  5-A  6-B  7-C  8-B  9-B  10-C

**二、多选题：** 
1. ABCE
2. ABCDE
3. ABC
4. AC
5. ABD

**三、填空题：**
1. 地铁线路图、State（状态/数据）
2. reducer（归并）、追加
3. 工具、tool_calls
4. Acting（行动）、行动
5. 并行化、协调器-工作者
6. 节点名、初始状态（参数）
7. 确定性、循环迭代

**四、简答题要点：**
1. 互补关系。LangChain 适合线性流水线（翻译、摘要），LangGraph 适合复杂路由和循环（Agent、条件分支）。两者可组合使用。
2. HumanMessage（用户输入）→ AIMessage（含 tool_calls）→ ToolMessage（工具执行结果）→ AIMessage（最终回答）。
3. create_react_agent() 一行搞定，适合快速原型；手动构建灵活可控，适合自定义逻辑。
4. Send() 返回 Send 列表实现 1 对 N 并行。Edge 是 1 对 1 固定，Conditional 是 1 对 1 动态选择，Send 是 1 对 N 并行。
5. generate → evaluate → 达标? → END / 不达标 → optimize → generate 循环。需要最大迭代次数防死循环，质量阈值提供提前退出。

**五、代码分析题要点：**

题目一：
- 问题：`step_b` 只返回 `msg` 字段，没有返回 `count` 字段。虽然不会报错（LangGraph 是增量更新），但如果意图是保留 count，这是正确的。
- 真正问题：图缺少 `START` 和 `END` 的边连接，应该加 `add_edge(START, "a")` 和 `add_edge("b", END)`。

题目二：
1. START → llm → (条件判断: 有 tool_calls → tools / 没有 → END) → tools → llm（循环）
2. llm 收到问题 → 发现有 tool_calls → router 返回 "tool_node" → 走向 tools 节点
3. tools 执行 add(3,4)=7 → 返回 ToolMessage → 流程回到 llm 节点（因为 `add_edge("tools", "llm")`）
4. llm 看到 ToolMessage 后生成最终回答，不再有 tool_calls → router 返回 END → 流程结束
5. 形成了 ReAct 循环：工具执行后回到 LLM 再次推理，直到 LLM 不再需要工具为止。
