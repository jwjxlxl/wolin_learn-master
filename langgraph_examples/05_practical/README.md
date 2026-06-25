# 05_practical — 智能问答 Agent 综合实战

## 概述

本目录是 LangGraph 课程的**综合实战**，将前面所学的 Agent 循环、工具定义、条件分支等知识组合起来，构建一个能完成真实任务的智能问答 Agent。

### 什么是综合实战？

前面几个目录分别学习了独立的概念：

| 目录 | 学了什么 |
|------|---------|
| 01_introduction | LangGraph 是什么、StateGraph 基础 |
| 02_state_and_branching | 状态管理 + 条件分支 |
| 03_agent_loop | ReAct Agent 循环（`llm_call → tool_node → llm_call`） |
| 04_workflows | 提示链、路由、并行化、评估器-优化器 |

本目录将这些概念**组合成一个完整的应用**：

```
用户提问
  ↓
[系统提示词] 定义 Agent 角色和能力
  ↓
[ReAct 循环] llm → 判断需要哪些工具 → 执行 → 再 llm → ...
  ↓
知识库搜索 / 数学计算 / 日期查询（多种工具混用）
  ↓
返回最终答案
```

---

## 文件说明

### `search_qa_agent.py` — 智能问答 Agent

**学什么：**
- 构建一个能搜索知识库、执行计算、查询日期的完整 Agent
- 用 `ChatPromptTemplate` 添加系统提示词
- 理解 `MessagesPlaceholder` 的作用
- 单次对话触发多种不同类型的工具

#### 代码案例：综合问答 Agent

**流程图：**
```
START → llm → (有 tool_calls?) → tool → 回到 llm
                ↘ (无) ─────────────→ END
```

标准 ReAct 循环结构，但加入了**系统提示词**和**多种工具**。

**三大工具：**

| 工具 | 功能 | 示例 |
|------|------|------|
| `search_knowledge` | 搜索知识库（公司、产品、福利、地址） | "公司的福利有哪些？" |
| `calculate` | 安全的数学计算 | "123 乘以 456 等于多少" |
| `get_date` | 获取当前日期和时间 | "今天几号？" |

**关键代码片段：**

```python
# 1. 定义工具
@tool
def search_knowledge(query: str) -> str:
    """在知识库中搜索相关信息（公司、产品、福利、地址等）"""
    kb = {
        "公司": "我们公司是一家专注于 AI 技术的高科技企业。",
        "产品": "我们的核心产品是智能助手和自动化解决方案。",
        "福利": "公司提供五险一金、带薪年假、弹性工作制等福利。",
        "地址": "公司位于北京市海淀区中关村软件园。",
    }
    for key, value in kb.items():
        if key in query:
            return value
    return f"知识库中未找到关于'{query}'的信息。"

@tool
def calculate(expression: str) -> str:
    """执行数学计算（安全的 eval 环境）"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"

@tool
def get_date() -> str:
    """获取当前日期和时间"""
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

# 2. 添加系统提示词（给 Agent 定义角色）
system_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能问答助手。你可以搜索知识库、执行计算和查询日期。"),
    MessagesPlaceholder("messages"),  # 运行时替换为实际消息列表
])

# 3. LLM 节点：拼接提示词后调用
def llm_call(state: QAState):
    messages = state["messages"]
    full_messages = system_prompt.format_messages(messages=messages)
    response = model_with_tools.invoke(full_messages)
    return {"messages": [response]}

# 4. 构建标准 ReAct Agent 图
graph = (
    StateGraph(QAState)
    .add_node("llm", llm_call)
    .add_node("tool", tool_executor)
    .add_edge(START, "llm")
    .add_conditional_edges("llm", route_check, ["tool", END])
    .add_edge("tool", "llm")  # ★ 循环
    .compile()
)
```

**测试用例：**

| 问题 | 触发工具 | 预期行为 |
|------|---------|---------|
| "公司的福利有哪些？" | `search_knowledge` | 从知识库匹配"福利"关键词，返回福利信息 |
| "计算 123 乘以 456 等于多少" | `calculate` | 执行 `123 * 456`，返回结果 `56088` |
| "今天几号？" | `get_date` | 返回当前日期时间 |

**消息流示例（用户问 "公司的福利有哪些？"）：**
```
[0] HumanMessage: "公司的福利有哪些？"
[1] AIMessage: 调用工具 search_knowledge(query="公司福利")
[2] ToolMessage: "公司提供五险一金、带薪年假、弹性工作制等福利。"
[3] AIMessage: "公司的福利包括：五险一金、带薪年假、弹性工作制等。"
```

---

## 关键概念详解

### `ChatPromptTemplate` — 系统提示词

```python
system_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能问答助手..."),
    MessagesPlaceholder("messages"),
])
```

- `("system", "...")` — **角色定义**，告诉 LLM "你是谁"、"你能做什么"
- `MessagesPlaceholder("messages")` — **占位符**，运行时会被实际的对话消息列表替换

**为什么需要系统提示词？**
- 没有系统提示词的 Agent 是"空白状态"，回答风格不可控
- 有了系统提示词，可以定义 Agent 的角色、能力边界、行为准则
- 实际产品中，系统提示词通常包含：角色描述 + 可用工具说明 + 回复格式要求

### `MessagesPlaceholder` 的作用

```python
MessagesPlaceholder("messages")  # ← 占位符，等运行时填充
```

它在 Prompt 模板中占一个位置，调用 `format_messages(messages=实际消息列表)` 时，这个占位符会被替换为真实的对话历史。

**类比：** 就像写邮件时的 `{{收件人姓名}}` 占位符，发送时才替换为真实姓名。

### 安全的 `eval` 用法

`calculate` 工具中使用了安全的 `eval`：

```python
eval(expression, {"__builtins__": {}}, {})
```

- `{"__builtins__": {}}` — 清空内置函数，防止执行危险操作（如 `os.system("rm -rf /")`）
- `{}` — 空的命名空间，没有额外变量
- 这是**受限的安全计算环境**，只允许基本数学运算

> ⚠️ 注意：即使是这样也不是 100% 安全的。生产环境中应使用专门的数学表达式解析器（如 `asteval`、`simpleeval`）。

---

## 实际应用场景

本文件演示的是一个"企业问答 Agent"，同样的架构可以应用到各种场景：

- 🏢 **企业内部助手**：员工问"年假怎么算？" → 查 HR 知识库；"今天星期几？" → 查日期；"帮我算绩效" → 执行计算
- 🛒 **电商客服 Agent**：查订单状态 → 查物流 → 计算优惠金额 → 修改收货地址（多工具串联）
- 🏦 **银行客服 Agent**：查余额 → 转账 → 计算利息 → 查询汇率（金融类多工具组合）
- 🏥 **医疗问诊 Agent**：查询药品信息 → 症状匹配 → 计算 BMI → 推荐科室
- 🎓 **教育辅导 Agent**：查知识点 → 数学题计算 → 查询考试时间 → 生成学习计划
- 📊 **数据分析 Agent**：查询数据库 → 计算统计指标 → 生成图表 → 解释结果
- 🏠 **智能家居 Agent**：查询室温 → 计算能耗 → 调整空调温度 → 查询天气预报

---

## 在 LangGraph 学习路径中的位置

```
01_introduction          → 了解 LangGraph 是什么、StateGraph 的基础用法
02_state_and_branching   → 状态管理 + 条件分支
03_agent_loop            → ReAct Agent 循环（核心概念）
04_workflows             → 五种工作流模式（高级编排）
05_practical             → 综合实战（← 你在这里）
```

**本文件的定位：**
- 是前面所有知识的**综合应用**
- 展示了如何用 `ChatPromptTemplate` 给 Agent 添加角色定义
- 体现了**多工具混用**的实际场景（搜索 + 计算 + 日期查询）

---

## 运行方式

```bash
# 确保 Ollama 正在运行
ollama serve

# 下载模型（首次运行）
ollama pull qwen3.5:2b

# 运行示例
python search_qa_agent.py
```

> 🎉 运行完成后，LangGraph 课程的核心内容就全部学完了！
