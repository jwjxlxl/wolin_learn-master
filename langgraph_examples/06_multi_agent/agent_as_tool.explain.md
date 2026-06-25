# agent_as_tool.py — 函数解释与流程图

## 文件概览

本文件演示 **Agent-as-Tool 模式**：主 Agent（总台接待员）把子 Agent（各部门专员）当作可调用的工具。控制权始终在主 Agent 手里，子 Agent 执行完就返回。

---

## 函数一：`reception_desk_demo()` — 无 LLM 版客服总台

### 函数解释

**目标：** 不用 LLM，用关键词匹配模拟完整的 Agent-as-Tool 控制流，帮助理解模式结构本身。

**架构：**

```
员工提问 → 主 Agent（关键词判断）→ 调用子 Agent 工具 → 子 Agent 内部执行完整 ReAct 循环 → 返回最终回答 → 主 Agent 汇总 → END
```

**三个子 Agent，每个都是独立的 mini StateGraph：**

| 子 Agent | 工具 | 处理范围 |
|---------|------|---------|
| **IT 支持** (`it_graph`) | `reset_password` / `check_system_status` | 密码重置、系统状态查询 |
| **HR 支持** (`hr_graph`) | `check_leave_balance` / `lookup_policy` | 年假查询、人事政策 |
| **财务支持** (`finance_graph`) | `check_budget` / `calculate_reimbursement` | 部门预算、报销计算 |

**关键步骤：**

1. **构建子 Agent 图** — 每个子 Agent 是一个完整的 `StateGraph`，有 `llm` 节点、`tool` 节点、条件路由和循环边。用关键词匹配代替 LLM 决策。

2. **子 Agent 打包为 `@tool`** — 用 `@tool` 装饰器把子图包装成普通工具函数：
   - `call_it_support(query)` → `it_graph.invoke(...)`
   - `call_hr_support(query)` → `hr_graph.invoke(...)`
   - `call_finance(query)` → `finance_graph.invoke(...)`

3. **构建主 Agent 图** — 主 Agent 用关键词判断问题归属，选择对应的子 Agent 工具调用。

4. **执行流程** — 主 Agent 收到问题 → 判断归属 → 调用子工具 → 子 Agent 内部循环执行 → 返回结果 → 主 Agent 汇总回答。

### 流程图

```
                          agent_as_tool.py — reception_desk_demo() 流程图

START
  │
  ▼
┌─────────────────────────┐
│     main_llm 节点        │  总台接待员：判断问题归属（关键词匹配）
│  ┌───────────────────┐  │
│  │ 含"密码/系统/IT"   │──┼─→ 调用 call_it_support
│  │ 含"年假/假期/HR"   │──┼─→ 调用 call_hr_support
│  │ 含"预算/报销/财务" │──┼─→ 调用 call_finance
│  │ 都不匹配           │──┼─→ 直接回复"请问需要什么帮助？"
│  └───────────────────┘  │
└───────────┬─────────────┘
            │
            ▼
  ┌─────────────────────────────────────────────────────┐
  │                   子 Agent 工具（@tool 包装）          │
  │  call_it_support(query)                             │
  │    │                                                 │
  │    ▼                                                 │
  │  ┌─────────────────────────────────────────────┐    │
  │  │           it_graph (独立 StateGraph)          │    │
  │  │                                               │    │
  │  │  START → it_llm → (有tool_calls?) → it_tool   │    │
  │  │              ↘ (无) ───────────→ END          │    │
  │  │                                               │    │
  │  │  工具: reset_password / check_system_status    │    │
  │  └─────────────────────────────────────────────┘    │
  │    │                                                 │
  │    ▼ 同样适用于                                      │
  │  hr_graph (check_leave_balance / lookup_policy)      │
  │  finance_graph (check_budget / calc_reimbursement)   │
  └────────────────────┬────────────────────────────────┘
                       │ 返回最终回答字符串
                       ▼
┌─────────────────────────┐
│    main_tool_node 节点   │  执行子 Agent 工具，获取返回结果
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│     main_llm 节点        │  收到子 Agent 结果 → 汇总回答
│  （检测到有 ToolMessage） │
└───────────┬─────────────┘
            │ 无 tool_calls → END
            ▼
          END
```

---

## 函数二：`agent_as_tool_with_llm()` — LLM 版 Agent-as-Tool

### 函数解释

**目标：** 用真正的 LLM 做决策，架构与示例 1 完全相同，只是"关键词匹配"换成"LLM 语义理解"。

**架构：**

```
员工提问 → 主 Agent（LLM 判断问题归属）→ 调用子 Agent 工具 → 子 Agent 用 LLM + 工具处理 → 返回结果 → 主 Agent 汇总 → END
```

**与示例 1 的核心区别：**

| 维度 | 示例 1（无 LLM） | 示例 2（LLM 版） |
|------|-----------------|-----------------|
| **主 Agent 决策** | `if "密码" in query` | LLM 语义理解后判断调用哪个工具 |
| **子 Agent 构建** | 手写 StateGraph（~50 行） | `build_react_agent()` 一行创建 |
| **子 Agent 能力** | 关键词匹配路由工具 | LLM 自主决定调用哪个工具 |
| **适合场景** | 教学理解控制流 | 真实生产环境 |

**关键步骤：**

1. **子 Agent 一行创建** — 用 `build_react_agent(it_model, it_tools, system_prompt)` 一行创建每个子 Agent，替代示例 1 中手写的 StateGraph。

2. **子 Agent 打包为 `@tool`** — 与示例 1 完全相同，调用子 Agent 的 `invoke()` 方法。

3. **主 Agent 用 LLM** — `model.bind_tools(all_tools)` 让 LLM 知道有三个工具可用，LLM 自主判断调用哪个。

4. **主 Agent 图结构** — 与示例 1 同构：`main_llm → main_router → main_tool_node → main_llm` 循环。

### 流程图

```
                          agent_as_tool.py — agent_as_tool_with_llm() 流程图

START
  │
  ▼
┌─────────────────────────────────┐
│        main_llm 节点             │  总台接待员（LLM 版）
│  model.bind_tools([              │
│    call_it_support,              │  LLM 自主判断问题归属
│    call_hr_support,              │  决定调用哪个子 Agent 工具
│    call_finance                  │
│  ])                              │
└───────────┬─────────────────────┘
            │ 有 tool_calls?
            ▼
  ┌─────────────────────────────────────────────────────┐
  │                   子 Agent 工具（@tool 包装）          │
  │  call_it_support(query)                             │
  │    │                                                 │
  │    ▼                                                 │
  │  ┌─────────────────────────────────────────────┐    │
  │  │         it_sub_agent (build_react_agent)      │    │
  │  │                                               │    │
  │  │  LLM + tools: reset_password                  │    │
  │  │              check_system_status               │    │
  │  │  系统提示: "你是IT支持专员..."                  │    │
  │  └─────────────────────────────────────────────┘    │
  │    │                                                 │
  │    ▼ 同样适用于                                      │
  │  hr_sub_agent (LLM + check_leave_balance/lookup)     │
  │  finance_sub_agent (LLM + check_budget/calc_reimb)   │
  └────────────────────┬────────────────────────────────┘
                       │ 返回最终回答字符串
                       ▼
┌─────────────────────────┐
│    main_tool_node 节点   │  执行子 Agent 工具，获取返回结果
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│     main_llm 节点        │  LLM 看到子 Agent 返回 → 汇总回答
│  (模型不再需要工具)       │
└───────────┬─────────────┘
            │ 无 tool_calls → END
            ▼
          END
```

---

## 两个函数对比总结

| 对比项 | `reception_desk_demo()` | `agent_as_tool_with_llm()` |
|-------|------------------------|---------------------------|
| **决策方式** | 关键词匹配（`if "密码" in query`） | LLM 语义理解（`model.bind_tools(...)`） |
| **子 Agent 构建** | 手写 StateGraph（~50 行/个） | `build_react_agent()` 一行 |
| **依赖 LLM** | 否 | 是（Ollama `qwen3.5:2b`） |
| **教学目的** | 理解 Agent-as-Tool 的**控制流结构** | 理解**真实生产环境**的做法 |
| **消息流转** | 完全相同（都是 ReAct 循环） | 完全相同 |
| **控制权归属** | 始终在主 Agent | 始终在主 Agent |

## 核心模式一图总结

```
Agent-as-Tool 模式本质：

主 Agent (StateGraph)
  ├── main_llm: 判断需要哪个子 Agent
  └── main_tool_node: 调用子 Agent 工具
        │
        ├── call_it_support() ──→ it_graph.invoke(...) ──→ 返回回答
        ├── call_hr_support() ──→ hr_graph.invoke(...) ──→ 返回回答
        └── call_finance()  ──→ finance_graph.invoke(...) ──→ 返回回答
              │
              ▼
        回到 main_llm → 汇总回答 → END

关键特征：
  ✅ 子 Agent 是完整的 StateGraph（有自己的 LLM + 工具 + 循环）
  ✅ 子 Agent 被 @tool 包装，对主 Agent 来说就是一个普通工具
  ✅ 子 Agent 执行完必须返回，控制权始终在主 Agent
  ✅ 与 Handoff 的区别：Handoff 交出控制权就不回来了
```
