# 06_multi_agent — 多智能体协作模式

## 概述

本目录讲解 LangGraph 中的**多智能体（Multi-Agent）协作模式**。

前面所学的 Agent 和工作流都是"单 Agent"或"确定性编排"，而多智能体模式让**多个 Agent 协作完成复杂任务**。

### 三种多 Agent 模式

| 模式 | 文件 | 控制权 | 比喻 |
|------|------|--------|------|
| **Agent-as-Tool** | `agent_as_tool.py` | 主 Agent 始终掌控 | 前台打电话给某部门，等回复后继续对话 |
| **Handoff** | `handoffs.py` | 完全移交 | 接力赛交棒，接棒的人跑你就不跑了 |
| **Supervisor** | `supervisor.py` | 主管分配+审查 | 包工头派活，工人干完回来汇报，包工头再派下一个 |

### 三种模式对比详解

```
Agent-as-Tool:
  主 Agent ──调用──→ 子 Agent（IT 支持）──返回──→ 主 Agent 继续决策
                    子 Agent（HR 支持）──返回──→
                    子 Agent（财务）  ──返回──→
  特点：子 Agent 是"工具式"，问完就回，主 Agent 掌控全程

Handoff:
  分诊护士 ──交棒──→ 心内科医生 ──交棒──→ 放射科 ──结束
  特点：控制权完全转移，前一个人交出后就不管了，下一个独立决策

Supervisor:
                ├──→ 研究员 ──汇报──→
  项目经理 ──分配──→ 程序员 ──汇报──→ 项目经理审查 ──决定下一步
                ├──→ 评审员 ──汇报──→
  特点：主管是唯一决策者，工人干完必回主管，主管可多次循环审查
```

---

## 文件说明

### 1. `agent_as_tool.py` — Agent 作为工具

**学什么：**
- 理解 Agent-as-Tool：主 Agent 把子 Agent 当作可调用的工具
- 用编译好的子图包装为 `@tool`
- 掌握"控制权始终在主 Agent"的架构特点

#### 示例 1：无 LLM 版 — 关键词路由的客服总台（理解模式结构）

**场景：** 公司智能客服总台，前台接待员（主 Agent）接到员工电话后判断问题归属，转接对应部门专员（子 Agent）。

**架构图：**
```
员工提问 → 主 Agent（前台接待员）
              ├── IT 问题 → 调用 IT 支持子 Agent（密码/系统状态）→ 返回结果
              ├── 人事问题 → 调用 HR 支持子 Agent（年假/政策）→ 返回结果
              └── 财务问题 → 调用财务支持子 Agent（预算/报销）→ 返回结果
           主 Agent 汇总结果 → 回答员工
```

**三个子 Agent，每个都是独立的 mini StateGraph：**

| 子 Agent | 工具 | 处理范围 |
|---------|------|---------|
| **IT 支持** | `reset_password` / `check_system_status` | 密码重置、系统状态查询 |
| **HR 支持** | `check_leave_balance` / `lookup_policy` | 年假查询、人事政策 |
| **财务支持** | `check_budget` / `calculate_reimbursement` | 部门预算、报销计算 |

**关键代码片段：**
```python
# 子 Agent 是独立的 StateGraph（完整的 ReAct Agent）
it_graph = (
    StateGraph(ITState)
    .add_node("it_llm", it_llm)
    .add_node("it_tool", it_tool_node)
    .add_edge(START, "it_llm")
    .add_conditional_edges("it_llm", it_router, ["it_tool", END])
    .add_edge("it_tool", "it_llm")
    .compile()
)

# 子 Agent 打包为 @tool
@tool
def call_it_support(query: str) -> str:
    """将 IT 相关问题转接给 IT 支持专员"""
    result = it_graph.invoke({"messages": [HumanMessage(content=query)]})
    return extract_final_answer(result)

# 主 Agent 把子 Agent 当作普通工具调用
all_tools = [call_it_support, call_hr_support, call_finance]
main_model = model.bind_tools(all_tools)
```

**测试用例：**
| 员工提问 | 转接部门 | 最终回答 |
|---------|---------|---------|
| "我的电脑密码忘记了，怎么办？" | IT 支持 | 重置密码，初始密码 123456 |
| "我想查一下年假还剩几天？" | HR 支持 | 年假余额 5 天，已使用 7 天 |
| "技术部今年的预算用得怎么样了？" | 财务支持 | 预算 200 万，已使用 72.5% |
| "OA 系统今天怎么这么慢？" | IT 支持 | OA 系统运行正常，响应时间 120ms |

#### 示例 2：LLM 版 — 真正的 Agent-as-Tool

架构与示例 1 完全相同，但"关键词匹配"换成了"LLM 理解"。子 Agent 使用 `build_react_agent()` 一行创建，主 Agent 用 LLM 判断问题归属。

**实际应用场景：**
- 🏢 **企业智能助手**：员工问各种问题 → 主 Agent 判断领域 → 转接 HR/IT/财务/行政子 Agent
- 🛒 **电商平台客服**：用户咨询 → 主 Agent 识别意图 → 转接订单/物流/售后/商品子 Agent
- 🏦 **银行综合客服**：客户来电 → 主 Agent 判断业务类型 → 转接信用卡/贷款/理财/账户子 Agent
- 🏥 **医院导诊系统**：患者描述症状 → 主 Agent 初步判断 → 转接内科/外科/儿科子 Agent
- 📚 **在线教育平台**：学生提问 → 主 Agent 识别学科 → 转接数学/英语/物理/化学子 Agent

---

### 2. `handoffs.py` — 控制权接力传递

**学什么：**
- 理解 Handoff：控制权从一个 Agent 完全转移到下一个 Agent
- 使用 `Command(goto=...)` 实现节点间的接力传递
- 掌握与 Agent-as-Tool 的关键区别（控制权归属）

#### 核心概念：Handoff（交接）

```
场景：医院分诊系统
患者到院后：
  分诊护士评估症状 → 转给心内科医生 → 医生看后需拍片 → 转给放射科 → 出报告 → 结束
```

**关键特征：**
1. **控制权完全转移**：分诊护士把患者"交"给心内科后就不管了
2. **每个 Agent 独立决策**：心内科医生接手后自己决定下一步（处理 or 转科 or 结束）
3. **不需要条件边**：`Command(goto=...)` 自己决定去哪，不用 `add_conditional_edges`

**核心 API：**
```python
Command(goto="目标节点", update={状态更新})
```
- `goto`：指定下一个要执行的节点名（或 `END`）
- `update`：更新 StateGraph 的状态（类似节点返回的 dict）

**与 Agent-as-Tool 的区别：**
| | Agent-as-Tool | Handoff |
|---|---|---|
| **控制权** | 主 Agent 始终掌控 | 完全移交，交出就不管了 |
| **比喻** | 打电话问同事，挂了继续工作 | 接力赛交棒，交了你就不跑了 |
| **子 Agent 返回后** | 回到主 Agent 继续决策 | 不会回来（除非显式 goto 回来） |
| **适合** | 需要统一入口和汇总的场景 | 每个环节独立处理、无需回传的场景 |

#### 示例 1：无 LLM 版 — 关键词路由的医院分诊

**流程图：**
```
START → 分诊护士 →（症状判断）
            ├── 心脏相关 → 心内科 →（需拍片？）→ 放射科 → END
            ├── 神经相关 → 神经内科 → END
            └── 常见症状 → END（直接处理）
```

**关键代码片段：**
```python
def triage_nurse(state: HandoffState):
    symptom = state["symptom"]
    if any(k in symptom for k in ["胸闷", "心悸", "心脏", "胸痛"]):
        return Command(goto="cardiology", update={
            "path": ["分诊"],
            "diagnosis": "疑似心脏问题，请心内科接诊",
        })
    elif any(k in symptom for k in ["头痛", "眩晕", "麻木"]):
        return Command(goto="neurology", update={
            "path": ["分诊"],
            "diagnosis": "疑似神经问题，请神经内科接诊",
        })
    # ...

def cardiology(state: HandoffState):
    # 心内科独立决策：需要拍片 → 转放射科；不需要 → 直接结束
    if need_xray:
        return Command(goto="radiology", update={...})
    else:
        return Command(goto=END, update={...})

# 构建图：不需要条件边！Command(goto=...) 自己决定去哪
graph = (
    StateGraph(HandoffState)
    .add_node("triage", triage_nurse)
    .add_node("cardiology", cardiology)
    .add_node("neurology", neurology)
    .add_node("radiology", radiology)
    .add_node("general", general)
    .add_edge(START, "triage")
    .compile()
)
```

**测试用例：**
| 患者症状 | 就诊路径 | 诊断结果 |
|---------|---------|---------|
| "胸闷、心悸一周" | 分诊 → 心内科 → 放射科 → END | 心脏彩超显示轻度二尖瓣反流 |
| "头痛、眩晕" | 分诊 → 神经内科 → END | 建议做脑部 CT 进一步检查 |
| "胸痛加剧" | 分诊 → 心内科 → 放射科 → END | 需拍片确认 |
| "发烧 38 度" | 分诊 → END | 多休息多喝水，必要时服药 |

#### 示例 2：LLM 版 — 真正的 Handoff

架构与示例 1 相同，关键词匹配换成了 LLM 判断。分诊护士用结构化输出决定转哪个科室，各科室用 LLM 做专业诊断。

**实际应用场景：**
- 🏥 **医疗分诊系统**：患者症状 → 分诊 → 专科医生 → 检查科室 → 出报告
- 📞 **技术支持升级**：一线客服 → 无法解决 → 转二线技术 → 需要研发 → 转研发团队
- 📋 **审批流转**：员工申请 → 部门主管审批 → 财务审核 → 总经理批准 → 完成
- 🔄 **工单处理链**：用户报障 → 客服记录 → 技术排查 → 外部供应商 → 解决反馈
- 📝 **内容审核链**：用户投稿 → 初审编辑 → 二审主编 → 排版发布

---

### 3. `supervisor.py` — 主管分配+审查

**学什么：**
- 理解 Supervisor：一个主管 Agent 管理多个工人 Agent 的工作流
- 用 `Command(goto=...)` 实现工人完成后回到主管的循环
- 掌握 Supervisor 模式的迭代审查流程

#### 核心概念：Supervisor（监督者）

```
场景：软件开发团队
产品经理提需求后：
  项目经理（Supervisor）拆解任务 → 分配研究员调研 → 审查结果 →
  分配程序员编码 → 审查代码 → 分配评审员验收 → 最终交付 → 结束
```

**关键特征：**
1. **Supervisor 是唯一决策者**：分配任务、审查结果、决定何时结束
2. **所有 Worker 完成后必须回到 Supervisor**（不能自行结束）
3. **Supervisor 可多次循环审查**（不满意 → 再研究/换人研究）
4. **用 `iteration` 字段防止无限循环**

**与 Handoff 的区别：**
| | Handoff | Supervisor |
|---|---------|------------|
| **控制权** | 接力传递，交出就不管 | 主管始终掌控，工人汇报后再分配 |
| **工人能否自行结束** | 能（自己决定 goto=END） | 不能（必须回到主管） |
| **比喻** | 接力赛 | 包工头派活 |
| **适合** | 每个环节独立处理的场景 | 需要统一审查和质量把控的场景 |

**与 Agent-as-Tool 的区别：**
| | Agent-as-Tool | Supervisor |
|---|---|---|
| **主 Agent 调用方式** | 工具式调用（问完就回） | 分配任务，工人完成后汇报 |
| **子 Agent 返回后** | 直接返回结果给主 Agent | 回到主管，主管审查后再分配 |
| **适合** | 简单的问答转接 | 需要迭代审查的多步骤任务 |

#### 示例 1：无 LLM 版 — 规则驱动的软件开发团队

**流程图：**
```
START → 项目经理（分配任务）
            ├── researcher（调研）→ 回到项目经理
            ├── coder（编码）    → 回到项目经理
            └── reviewer（评审） → 回到项目经理 →（三次后完成）→ END
```

**关键代码片段：**
```python
def supervisor(state: SupervisorState):
    """项目经理：决定下一个任务分配给谁。"""
    # 按顺序分配：研究 → 编码 → 评审
    task_order = ["researcher", "coder", "reviewer"]
    next_worker = task_order[iteration % 3]

    return Command(goto=next_worker, update={
        "next": next_worker,
        "messages": [AIMessage(content=f"请 {next_worker} 处理: {task}")],
    })

def researcher(state: SupervisorState):
    """研究员：调研技术方案。完成后回到项目经理。"""
    product = f"调研报告：{task} 可采用 Python + LangGraph 实现..."
    return Command(goto="supervisor", update={
        "work_product": product,
        "all_products": state["all_products"] + [product],
        "iteration": state["iteration"] + 1,
        "next": "",  # 清空，让 supervisor 重新决定
    })

# 构建图：不需要条件边，Command(goto=...) 自己决定去哪
graph = (
    StateGraph(SupervisorState)
    .add_node("supervisor", supervisor)
    .add_node("researcher", researcher)
    .add_node("coder", coder)
    .add_node("reviewer", reviewer)
    .add_edge(START, "supervisor")
    .compile()
)
```

#### 示例 2：LLM 版 — 真正的 Supervisor

项目经理用 LLM 结构化输出决定分配给哪个工人，各工人用 LLM 执行专业工作。

**测试用例：**
| 项目需求 | 执行步骤 | 最终产出 |
|---------|---------|---------|
| "搭建一个智能问答 Agent" | 研究 → 编码 → 评审 | 3 步完成，各阶段产出汇总 |
| "实现文档检索功能" | 研究 → 编码 → 评审 | 3 步完成，各阶段产出汇总 |

**实际应用场景：**
- 💻 **软件开发团队**：需求 → 技术方案调研 → 编码 → 代码评审 → 交付
- 📰 **内容生产**：选题 → 资料收集 → 撰写 → 编辑审核 → 发布
- 🎨 **设计项目**：需求分析 → 概念设计 → 细化方案 → 评审验收
- 📊 **数据分析**：明确需求 → 数据收集 → 分析建模 → 报告审核 → 输出
- 📝 **学术论文**：选题调研 → 实验设计 → 撰写初稿 → 同行评审 → 定稿
- 🏗️ **工程项目**：需求分析 → 方案设计 → 实施开发 → 质量验收 → 交付

---

## 三种模式选择指南

### 何时用哪种？

| 判断条件 | 推荐模式 |
|---------|---------|
| 需要一个统一入口，子任务独立且简单返回 | **Agent-as-Tool** |
| 主 Agent 需要汇总所有子任务结果 | **Agent-as-Tool** |
| 每个环节独立处理，无需回传给前一个 | **Handoff** |
| 处理链是线性的（A → B → C → 结束） | **Handoff** |
| 需要统一审查和质量把控 | **Supervisor** |
| 需要迭代改进（不满意就重做） | **Supervisor** |
| 工人不能自行结束，必须由主管决定 | **Supervisor** |

### 模式组合

实际项目中可以组合使用：

```
用户请求
  ↓
[Agent-as-Tool] 总台 Agent 判断需求类型
  ↓
[Handoff] 转给对应业务线，开始接力处理
  ↓
[Supervisor] 业务线主管审查各阶段产出
  ↓
返回最终结果
```

---

## 在 LangGraph 学习路径中的位置

```
01_introduction          → LangGraph 是什么
02_state_and_branching   → 状态管理 + 条件分支
03_agent_loop            → ReAct Agent 循环（核心）
04_workflows             → 工作流模式（链/路由/并行/改进）
05_practical             → 综合实战（智能问答 Agent）
06_multi_agent           → 多智能体（← 你在这里）
```

**三种模式回顾：**
- **Agent-as-Tool** — 主 Agent 掌控全程，子 Agent 当工具调用
- **Handoff** — 控制权接力传递，交出就不管了
- **Supervisor** — 主管分配+审查，工人完成后回到主管

---

## 运行方式

```bash
# 确保 Ollama 正在运行
ollama serve

# 下载模型（首次运行）
ollama pull qwen3.5:2b

# 运行各个示例（示例 1 均无需 LLM，可立即运行）
python agent_as_tool.py    # Agent-as-Tool 模式
python handoffs.py         # Handoff 模式
python supervisor.py       # Supervisor 模式
```

建议学习顺序：先理解无 LLM 示例（理解控制流），再看 LLM 版本（理解智能决策）。

> 🎉 运行完成后，LangGraph 课程全部模块学习完毕！
