# agent_as_tool.py — 函数解释与流程图

## 文件概览

本文件演示 **Agent-as-Tool 模式**：主 Agent（总台接待员）把子 Agent（各部门专员）当作可调用的工具。控制权始终在主 Agent 手里，子 Agent 执行完就返回。

---

## 函数：`agent_as_tool_with_llm()` — LLM 版 Agent-as-Tool

### 函数解释

**目标：** 用真正的 LLM 做决策，主 Agent 通过 LLM 语义理解判断问题归属。

**架构：**

```
员工提问 → 主 Agent（LLM 判断问题归属）→ 调用子 Agent 工具 → 子 Agent 用 LLM + 工具处理 → 返回结果 → 主 Agent 汇总 → END
```

**关键步骤：**

1. **子 Agent 一行创建** — 用 `build_react_agent(it_model, it_tools, system_prompt)` 一行创建每个子 Agent。

2. **子 Agent 打包为 `@tool`** — 用 `@tool` 装饰器把子 Agent 包装成普通工具，调用其 `invoke()` 方法。

3. **主 Agent 用 LLM** — `model.bind_tools(all_tools)` 让 LLM 知道有三个工具可用，LLM 自主判断调用哪个。

4. **主 Agent 图结构** — `main_llm → main_router → main_tool_node → main_llm` 循环。

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

## 核心模式一图总结

```
Agent-as-Tool 模式本质：

主 Agent (StateGraph)
  ├── main_llm: 判断需要哪个子 Agent
  └── main_tool_node: 调用子 Agent 工具
        │
        ├── call_it_support() ──→ it_sub_agent.invoke(...) ──→ 返回回答
        ├── call_hr_support() ──→ hr_sub_agent.invoke(...) ──→ 返回回答
        └── call_finance()  ──→ finance_sub_agent.invoke(...) ──→ 返回回答
              │
              ▼
        回到 main_llm → 汇总回答 → END

关键特征：
  ✅ 子 Agent 是完整的 StateGraph（有自己的 LLM + 工具 + 循环）
  ✅ 子 Agent 被 @tool 包装，对主 Agent 来说就是一个普通工具
  ✅ 子 Agent 执行完必须返回，控制权始终在主 Agent
  ✅ 与 Handoff 的区别：Handoff 交出控制权就不回来了
```
