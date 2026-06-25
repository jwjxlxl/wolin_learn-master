# 03_agent_loop — ReAct Agent 循环

## 概述

本目录讲解 LangGraph 中最核心的概念：**ReAct Agent 循环**，以及两种构建 Agent 的方式对比。

### ReAct Agent 是什么？

**ReAct = Reasoning（推理）+ Acting（行动）**

这是 AI Agent 的核心工作模式：模型不只是"聊天"，而是能**自主决定调用外部工具**来完成任务。

| | 普通聊天 LLM | ReAct Agent |
|---|---|---|
| **能力** | 只能根据训练知识回答 | 可以调用外部工具获取实时信息 |
| **决策** | 无法决定做什么 | 自己决定何时调用工具、调用哪个 |
| **循环** | 一次问答结束 | 思考 → 行动 → 观察 → 再思考...（循环） |
| **比喻** | 一个博学但断网的顾问 | 一个能上网、能查数据库、能执行代码的助手 |

### ReAct 循环流程图

```
START → LLM 思考 → (需要工具?) ──是──→ 执行工具 → 观察结果 ─┐
                            ↘ (不需要)                    │
                                     └→ END ←─────────────┘
```

### 消息流转（理解 Agent 的关键）

```
用户输入 → LLM → AIMessage(tool_calls=[...]) → 执行工具 → ToolMessage(...)
  → LLM 看到 ToolMessage → 再次推理 → 最终 AIMessage(content="回答") → END
```

三大组件：
1. **`bind_tools()`** — 把工具"告诉"模型，模型会在 `tool_calls` 字段列出想调用的工具
2. **`llm_call` 节点** — 调用带工具的模型，返回 AIMessage（可能含 `tool_calls`）
3. **`tool_node` 节点** — 执行 AIMessage 中的 `tool_calls`，返回 ToolMessage

**生活化比喻：**
- LLM = "老板"（做决策，但不亲自干活）
- Tool = "员工"（听老板指挥，执行具体任务）
- ReAct 循环 = "老板想 → 指派任务 → 员工干活 → 汇报结果 → 老板再想..."

---

## 文件说明

### 1. `agent_react.py` — 手动构建 ReAct Agent

**学什么：**
- 用 `@tool` 装饰器定义 AI 可调用的工具
- 用 `bind_tools()` 把工具绑定到模型
- 构建带循环的图（`tool_node → llm_call` 形成闭环）
- 理解 `AIMessage.tool_calls` 和 `ToolMessage` 的消息流转

#### 示例 1：最简 Agent — 数学计算

**流程图：**
```
START → llm_call → (有 tool_calls?) → tool_node → 回到 llm_call
                    ↘ (无) ──────────────────────→ END
```

**代码案例：** 定义 `add`、`multiply`、`divide` 三个工具，让 AI 能用它们回答数学问题。

**关键代码片段：**
```python
# 1. 用 @tool 装饰器定义工具
@tool
def add(a: int, b: int) -> int:
    """计算两个数的和"""
    return a + b

# 2. 绑定工具到模型
model_with_tools = model.bind_tools(tools)

# 3. 定义状态（消息列表自动追加）
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# 4. 构建带循环的图
graph = (
    StateGraph(AgentState)
    .add_node("llm_call", llm_call)
    .add_node("tool_node", tool_node)
    .add_edge(START, "llm_call")
    .add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    .add_edge("tool_node", "llm_call")  # ★ 循环的关键
    .compile()
)
```

**消息流示例（用户问 "3 加 4 等于多少？"）：**
```
[0] HumanMessage: "3 加 4 等于多少？"
[1] AIMessage: 调用工具 add(a=3, b=4)
[2] ToolMessage: 结果: 7
[3] AIMessage: "3 加 4 等于 7"
```

#### 示例 2：多工具组合 — 单次对话使用多个工具

演示一个复杂问题触发**多次工具调用**：`(3 + 4) × 5 等于多少？`

Agent 自动决定工具调用顺序：
1. 先调用 `add(3, 4)` → 得到 7
2. 再调用 `multiply(7, 5)` → 得到 35
3. 最终回答 35

**关键学习点：** Agent 不需要你写死工具调用顺序，它**自己决定**先做什么、后做什么。

**实际应用场景：**
- 🧮 **智能计算器**：用户用自然语言提问数学/统计问题 → 自动选择公式计算
- 📞 **客服 Agent**：用户说"查订单 + 改地址" → 依次调用订单查询 API + 地址修改 API
- 🔬 **科研助手**：分析数据 → 调用统计工具 → 生成图表 → 解释结果
- 🛒 **电商 Agent**：搜索商品 → 比价 → 查库存 → 下单（多工具串联）
- 🏦 **金融分析**：查股价 → 算技术指标 → 生成报告 → 发送通知

---

### 2. `create_agent_demo.py` — 两种 Agent 构建方式对比

**学什么：**
- 使用 `create_react_agent()` 一行创建 Agent（官方推荐方式）
- 理解 `create_react_agent()` 底层做的事（就是 `agent_react.py` 的内容！）
- 知道何时用 `create_react()`，何时手动构建 StateGraph

#### 示例 1：用 `create_react_agent()` 创建 Agent（推荐方式）

**代码案例：** 用一行代码创建和 `agent_react.py` 完全相同的 Agent。

```python
# ★ 一行搞定！底层就是 agent_react.py 中手写的所有逻辑
agent = create_react_agent(model, tools)
result = agent.invoke({"messages": [HumanMessage("(3 + 4) × 5 等于多少？")]})
```

同时还展示了**消息流追踪**，打印每条消息的类型和内容，帮助调试理解：
```
【消息流追踪】（共 X 条消息）
    [0] HumanMessage: (3 + 4) × 5 等于多少？
    [1] AIMessage: 调用工具: ['add']
    [2] ToolMessage: 结果: 7
    [3] AIMessage: 调用工具: ['multiply']
    [4] ToolMessage: 结果: 35
    [5] AIMessage: (3 + 4) × 5 等于 35
```

#### 示例 2：手动构建相同 Agent（对比理解）

手动构建 ~20 行代码，展示 `create_react_agent()` 底层做了什么：

```python
# 这就是 create_react_agent() 内部做的事
graph = (
    StateGraph(AgentState)
    .add_node("llm_call", llm_call)
    .add_node("tool_node", tool_executor)
    .add_edge(START, "llm_call")
    .add_conditional_edges("llm_call", router, ["tool_node", END])
    .add_edge("tool_node", "llm_call")
    .compile()
)
```

**对比总结：**
| | `create_react_agent()` | 手动 StateGraph |
|---|---|---|
| **代码量** | 1 行 | ~20 行 |
| **效果** | 完全相同 | 完全相同 |
| **自定义** | 标准 ReAct 模式 | 完全控制每个节点 |

**何时用哪种？**

| 场景 | 推荐方式 |
|------|---------|
| 快速原型开发 | `create_react_agent()` |
| 标准 ReAct Agent（不需要自定义逻辑） | `create_react_agent()` |
| 只想关注 Tool 定义，不想管图结构 | `create_react_agent()` |
| 需要自定义状态（不只是 messages） | 手动 StateGraph |
| 需要多个 LLM 节点 | 手动 StateGraph |
| 需要在工具执行前后加自定义逻辑 | 手动 StateGraph |
| 需要混合工作流模式（路由 + Agent + 汇总） | 手动 StateGraph |

**实际应用场景：**
- 🚀 **项目起步**：先用 `create_react_agent()` 快速验证想法，确认方案可行后再按需重构为手动 StateGraph
- 📦 **标准化产品**：如果只需要标准的工具调用 Agent，用 `create_react_agent()` 减少维护成本
- 🏗️ **企业级应用**：需要在 Agent 前后加入审批、日志、缓存等自定义节点时，手动构建
- 🎓 **教学理解**：先学手动构建（理解原理），再用 `create_react_agent()`（提高生产力）

---

## 关键概念详解

### `@tool` 装饰器

普通函数变成 AI 可调用的工具：

```python
@tool
def search_database(query: str) -> str:
    """搜索数据库中的相关信息"""
    # ... 执行搜索 ...
    return results
```

- 函数名 → 工具名（`search_database`）
- 函数 docstring → **LLM 理解工具用途的关键**，要写清楚
- 参数类型注解 → LLM 知道传什么参数
- 返回值 → 自动转为 `ToolMessage` 的内容

### `bind_tools()`

把工具列表"告诉"模型：

```python
model_with_tools = model.bind_tools(tools)
```

- 模型收到工具列表后，会在回复中用 `tool_calls` 字段列出想调用的工具
- 模型**不会直接执行工具**，只是"说要调用"
- 执行工具是 `tool_node` 节点的工作

### 循环的退出条件

`should_continue` 路由函数检查最后一条消息：

```python
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    has_tool_calls = hasattr(last_message, 'tool_calls') and last_message.tool_calls
    if has_tool_calls:
        return "tool_node"  # 还有工具要调用，继续循环
    else:
        return END  # 不再需要工具，结束
```

> ⚠️ **注意**：如果模型反复调用工具不结束（罕见情况），可能形成无限循环。实际项目中需要设置 `recursion_limit` 防止此问题。

---

## 与 04_workflows 的关系

理解了本目录的 Agent 循环后，再学 04_workflows 的工作流模式，你就可以**组合使用**：

```
用户提问
  ↓
[03_agent_loop] ReAct Agent：自主调用工具获取数据
  ↓
[04_workflows] 工作流：对获取的数据进行评估-优化
  ↓
[03_agent_loop] Agent 再次调用工具：发送结果给用户
```

- **Agent（03）**：解决"什么时候用什么工具"的自主决策问题
- **Workflow（04）**：解决"流程怎么编排"的确定性编排问题

---

## 运行方式

```bash
# 确保 Ollama 正在运行
ollama serve

# 下载模型（首次运行）
ollama pull qwen3.5:2b

# 运行各个示例
python agent_react.py          # 手动构建 ReAct Agent
python create_agent_demo.py    # 两种构建方式对比
```

建议学习顺序：先理解 `agent_react.py`（手动构建，理解原理），再看 `create_agent_demo.py`（对比封装方式）。
