# 02_state_and_branching — 状态管理与条件分支

## 概述

本目录讲解 LangGraph 的两个基础概念：**状态（State）管理**和**条件分支（Conditional Branching）**。

这是 LangGraph 的核心机制——理解了它们，才能进一步学习 Agent 循环和工作流模式。

### 状态（State）是什么？

**State = 图执行过程中的"全局变量"**

```python
class MyState(TypedDict):
    text: str      # 输入文本
    category: str  # 分类结果
    reply: str     # 最终回复
```

- 每个节点**读取状态**作为输入
- 每个节点**返回状态更新**（只更新部分字段，不是覆盖全部）
- 状态在节点之间**自动传递**

**生活化比喻：**
- State = 流水线上的"产品托盘"
- 每个节点 = 一个工位
- 工位从托盘上取信息，加工后把结果放回托盘
- 托盘带着所有信息流向下一个工位

### 条件分支是什么？

```
固定边（Edge）：A → B（无论什么情况，都走这条路）
条件边（Conditional Edge）：A → (判断) → B 或 C 或 D（根据条件选择）
```

```python
.add_conditional_edges("classify", route, {
    "positive_reply": "positive_reply",
    "negative_reply": "negative_reply",
    "neutral_reply": "neutral_reply",
})
```

执行过程：
1. 流程走到 `classify` 节点后，调用路由函数 `route(state)`
2. `route` 返回一个字符串，比如 `"positive_reply"`
3. 在映射字典中查找对应的目标节点
4. 流程走向那个目标节点

**生活化比喻：**
- 固定边 = 单行道（只能直走）
- 条件边 = 十字路口（根据红绿灯选择方向）
- 路由函数 = 交警（判断当前该往哪走）

---

## 文件说明

### `conditional_branch.py` — 条件分支实战

**学什么：**
- 理解条件边（Conditional Edge）的工作原理
- 用 `add_conditional_edges` 实现分支路由
- 写路由函数：根据状态返回下一步节点名

#### 代码案例：情感分类与回复

**流程图：**
```
START → classify → (判断情感)
                     ├── positive → positive_reply → END
                     ├── negative → negative_reply → END
                     └── neutral  → neutral_reply  → END
```

**场景：** 用户输入一段文本，AI 判断情感倾向（正面/负面/中性），然后生成对应的回复。

**关键代码片段：**
```python
# 1. 定义状态
class SentimentState(TypedDict):
    text: str       # 用户输入
    category: str   # 分类结果（positive/negative/neutral）
    reply: str      # 最终回复

# 2. 分类节点：关键词匹配判断情感
def classify(state: SentimentState):
    positive_words = ["好", "棒", "开心", "喜欢", "谢谢", "不错"]
    negative_words = ["差", "糟", "难过", "讨厌", "失望", "不好"]
    
    text = state["text"]
    if any(w in text for w in positive_words):
        category = "positive"
    elif any(w in text for w in negative_words):
        category = "negative"
    else:
        category = "neutral"
    return {"category": category}

# 3. 路由函数：根据分类结果决定下一步
def route(state: SentimentState):
    cat = state["category"]
    if cat == "positive":
        return "positive_reply"
    elif cat == "negative":
        return "negative_reply"
    else:
        return "neutral_reply"

# 4. 构建图
graph = (
    StateGraph(SentimentState)
    .add_node("classify", classify)
    .add_node("positive_reply", positive_reply)
    .add_node("negative_reply", negative_reply)
    .add_node("neutral_reply", neutral_reply)
    .add_edge(START, "classify")
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
```

**测试用例：**
| 输入 | 分类 | 回复 |
|------|------|------|
| "今天天气真好，心情不错！" | positive | "太好了！听到你这么说我很开心！😊" |
| "今天的体验太差了，很失望。" | negative | "别担心，一切都会好起来的！" |
| "我明天要去开会。" | neutral | "好的，我明白了。" |

**特点：** 本示例**不调用 LLM**，纯本地执行，无需 API Key，适合理解条件分支的基础原理。

**实际应用场景：**
- 💬 **智能客服路由**：用户描述问题 → 关键词/LLM 分类 → 技术/售后/账单 → 不同处理流程
- 📋 **表单验证**：填写表单 → 验证字段 → 合法则提交 / 不合法则返回错误提示
- 🎯 **内容审核**：用户发帖 → 敏感词检测 → 通过/打回/标记 → 不同处理
- 🌡️ **IoT 设备控制**：传感器读数 → 温度过高/正常/过低 → 开空调/不动/开暖气
- 📧 **邮件自动处理**：收到邮件 → 判断类型（紧急/普通/广告）→ 通知/归档/删除
- 🏥 **健康问诊**：患者描述症状 → 初步分诊 → 内科/外科/急诊 → 不同科室
- 🎮 **游戏 AI**：玩家行为 → 判断状态（攻击/防御/逃跑）→ 不同 NPC 反应

---

## 关键概念详解

### 状态更新是"部分合并"，不是"覆盖"

节点返回的字典只更新指定字段，其他字段保持不变：

```python
def classify(state: SentimentState):
    # 只返回 category，text 和 reply 不受影响
    return {"category": "positive"}
```

这得益于 `TypedDict` + LangGraph 的状态合并机制，类似于 Python 的 `dict.update()`。

### 路由函数的返回值

路由函数返回**字符串**（节点名），然后通过映射字典找到目标节点：

```python
def route(state):
    return "positive_reply"  # 返回节点名
```

映射字典的格式：`{"返回值": "目标节点名"}`

也可以直接映射到 `END`：
```python
.add_conditional_edges("node", route, {
    "done": END,          # 直接结束
    "continue": "next",   # 继续下一个节点
})
```

### 固定边 vs 条件边

| | 固定边 `.add_edge()` | 条件边 `.add_conditional_edges()` |
|---|---|---|
| **目标** | 固定（A → B） | 动态选择（A → B 或 C 或 D） |
| **用途** | 确定性的下一步 | 需要分支判断的场景 |
| **示例** | `classify → positive_reply` | `classify → route → 选择之一` |

### 状态定义的最佳实践

```python
class MyState(TypedDict):
    # 输入字段（在 invoke 时传入初始值）
    input_text: str
    
    # 中间字段（在流程中逐步填充）
    processed_data: dict
    
    # 输出字段（最终结果）
    result: str
```

初始调用时，所有字段需要提供默认值（空字符串、空字典、0 等）：

```python
graph.invoke({
    "input_text": "用户输入",
    "processed_data": {},
    "result": ""
})
```

---

## 在 LangGraph 学习路径中的位置

```
01_introduction          → 了解 LangGraph 是什么、StateGraph 的基础用法
02_state_and_branching   → 状态管理 + 条件分支（← 你在这里）
03_agent_loop            → ReAct Agent 循环（条件分支 + 循环 = Agent）
04_workflows             → 五种工作流模式（组合使用分支、并行、循环）
05_practical             → 综合实战
```

**承上启下：**
- 本目录的**条件分支**是 03_agent_loop 中 `should_continue` 路由函数的基础
- 03 的 Agent 循环 = 本目录的条件分支 + 一条回环边（`tool_node → llm_call`）
- 04 的工作流模式 = 条件分支 + 并行（Send）+ 循环的组合

---

## 运行方式

```bash
# 本示例无需 API Key，直接运行
python conditional_branch.py
```

运行后会自动生成分支流程图到 `langgraph_examples/images/conditional_branch.png`，并测试三种情感输入。
