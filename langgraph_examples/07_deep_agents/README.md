# 07_deep_agents — DeepAgents 开箱即用的 Agent 框架

## 概述

本目录讲解 LangChain 的 **DeepAgents** 框架——在 LangGraph 之上封装的"开箱即用"Agent 解决方案。

### DeepAgents 是什么？

前面所学的 `create_react_agent()` 是"一行创建标准 Agent"，而 **`create_deep_agent()`** 是"一行创建**完整** Agent"——它自动内置文件系统、任务规划、子代理、Skills 等能力。

### 与已有模块的关系

```
手动 StateGraph (03)    →  最底层，完全控制
      ↑
create_react_agent() (03) → 封装标准 ReAct 循环
      ↑
create_deep_agent()      → 再封装：内置文件系统/子代理/Skills/上下文工程
```

| 维度 | `create_react_agent()` | `create_deep_agent()` |
|------|------------------------|----------------------|
| **代码量** | 1 行 | 1 行 |
| **内置工具** | 无（只支持用户传入的 tools） | 文件系统 / 任务规划 / 子代理 |
| **文件操作** | 需手动定义 | 内置（ls / read / write / edit / glob / grep） |
| **子代理** | 不支持 | 内置 Subagent 委派机制 |
| **Skills** | 不支持 | 内置 SKILL.md 插件系统 |
| **适合** | 简单工具调用 | 生产级多步骤任务 |

---

## 文件说明

### 1. `what_is_deepagent.py` — 纯概念认知

**学什么：**
- DeepAgents 的定位和六大核心内置能力
- 与手动 StateGraph / `create_react_agent()` 的层级关系
- 何时用 DeepAgents，何时用其它方式

**六大内置能力：**

| # | 能力 | 说明 |
|---|------|------|
| 1 | 文件系统工具 | ls / read_file / write_file / edit_file / glob / grep |
| 2 | 任务规划 | write_todos — Agent 自动拆解复杂任务为 TODO 列表 |
| 3 | 子代理 | Subagent — 委派专业任务给独立 Agent |
| 4 | Skills | SKILL.md 文件 — 可复用的行为模板 |
| 5 | 上下文工程 | 长期会话的上下文管理机制 |
| 6 | 模型灵活 | 支持任何 LangChain 模型 |

本文件为**纯概念**，无需模型或 API，直接运行即可理解。

---

### 2. `quickstart.py` — 快速入门

**学什么：**
- 用 `create_deep_agent()` 一行创建 Agent
- 添加自定义工具
- 配置 `FilesystemBackend` 实现安全的文件读写

#### 示例 1：最简 DeepAgent

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model=model,           # 任意 LangChain 模型
    tools=[my_tool],       # 用户自定义工具
    instructions="你是一个智能助手。",
)
result = agent.invoke({"messages": [HumanMessage(content="...")]})
```

**重点：** 对比 `create_react_agent()`——DeepAgent 自动内置了文件系统、任务规划等额外能力。

#### 示例 2：文件操作能力

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

backend = FilesystemBackend(root_dir="./workspace")  # 安全沙箱
agent = create_deep_agent(
    model=model,
    backend=backend,
    instructions="你是一个写作助手，可以读写文件。",
)
# Agent 可以自主使用 ls / read_file / write_file / edit_file / glob / grep
```

**测试场景：**
| 用户请求 | Agent 行为 | 预期结果 |
|---------|-----------|---------|
| "创建 hello.txt 并写入问候语" | write_file → 创建文件 | 文件中包含问候语 |
| "查看当前目录有哪些文件" | ls → 列出目录 | 返回目录内容 |

**实际应用场景：**
- 📝 **写作助手**：用户口述大纲 → Agent 创建草稿 → 反复编辑 → 保存终稿
- 📊 **数据分析**：用户上传 CSV → Agent 读取分析 → 写入报告文件
- 🔧 **代码生成**：描述需求 → Agent 生成代码文件 → 自动测试 → 修复 bug
- 📋 **文档处理**：批量处理文件 → 提取关键信息 → 汇总输出
- 🏢 **企业助手**：查询内部文档 → 生成报告 → 保存到指定目录

---

### 3. `subagents_and_skills.py` — 子代理 + Skills

**学什么：**
- 用 `Subagent` 定义独立的专业子代理
- 理解"上下文隔离"的委派机制
- 用 `SKILL.md` 文件定义可复用技能

#### 示例 1：子代理基础

```python
from deepagents import create_deep_agent, Subagent

# 定义专业子代理
data_analyst = Subagent(
    name="data_analyst",
    description="分析数据并生成统计报告",
    model=model,
    system_prompt="你是一个数据分析师...",
)

# 主 Agent 注册子代理
agent = create_deep_agent(
    model=model,
    subagents=[data_analyst],
    instructions="遇到数据分析任务请委派给数据分析师。",
)
```

**关键特征：**
1. **上下文隔离**：子代理有独立的对话上下文，不影响主 Agent
2. **独立模型**：子代理可用不同模型（如主 Agent 用轻量模型，子代理用强模型）
3. **自动委派**：主 Agent 根据任务描述自动判断是否委派

#### 示例 2：Skills 加载

创建 `SKILL.md` 文件：
```markdown
---
name: 代码审查
description: 审查代码并指出潜在问题
triggers: 当用户请求审查代码时
---

# 代码审查技能
你是一个资深的代码审查员...
```

加载方式：
```python
agent = create_deep_agent(
    model=model,
    skills=["./skills/"],  # Agent 自动扫描 SKILL.md 文件
    backend=FilesystemBackend(root_dir="./workspace"),
)
```

**Skills vs 子代理的区别：**
| | Skills | 子代理 |
|---|--------|--------|
| **本质** | 行为模板（Markdown 文件） | 独立 Agent |
| **上下文** | 在主 Agent 上下文中执行 | 独立上下文 |
| **适合** | 简单规则/格式化的行为 | 需要独立推理的复杂任务 |
| **文件化** | 是（SKILL.md） | 否（Python 代码定义） |

**实际应用场景：**
- 🏢 **企业知识库**：每个部门一个 Skill（HR 政策 / IT 指南 / 财务流程），Agent 自动匹配
- 📚 **教育辅导**：每个学科一个 Subagent（数学/英语/物理），独立上下文不互相污染
- 🔧 **开发团队**：代码审查 Skill + 测试 Subagent + 部署 Subagent，各司其职
- 📰 **内容生产**：选题 Skill → 写作 → 编辑 Skill → 发布，流水线协作
- 🌐 **多语言翻译**：每个语言对一个 Subagent，翻译质量独立把控

---

## 安装与运行

```bash
# 1. 安装 DeepAgents（额外依赖）
pip install deepagents

# 2. 确保 Ollama 正在运行
ollama serve

# 3. 下载模型（首次运行）
ollama pull qwen3.5:2b

# 4. 运行各个示例
python what_is_deepagent.py        # 纯概念，无需模型
python quickstart.py               # 最简入门 + 文件操作
python subagents_and_skills.py     # 子代理 + Skills
```

建议学习顺序：先理解概念（`what_is_deepagent.py`），再体验最简流程（`quickstart.py`），最后学习进阶用法（`subagents_and_skills.py`）。

---

## 在 LangGraph 学习路径中的位置

```
01_introduction          → LangGraph 是什么
02_state_and_branching   → 状态管理 + 条件分支
03_agent_loop            → ReAct Agent 循环（核心）
04_workflows             → 工作流模式
05_practical             → 综合实战
06_multi_agent           → 多智能体协作模式
07_deep_agents           → DeepAgents 开箱即用框架（← 你在这里）
```

**回顾：**
- 03 学会了手动构建 Agent（StateGraph）和一行封装（`create_react_agent()`）
- 07 在此基础上学习**生产级**的 Agent 框架——内置文件系统、子代理、Skills 等全套能力

---

> 🎉 学完本模块，你将能用 DeepAgents 快速搭建生产级的智能 Agent！
