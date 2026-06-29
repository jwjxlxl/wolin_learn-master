# Agent 开发范式教学模块

## 学习路线

```
01_reasoning              02_action                    03_collaboration
     │                         │                             │
     ▼                         ▼                             ▼
  推理范式  ──────────→   行动范式  ─────────────→  协作范式
 CoT / Self-Ask / ToT     ReAct / Plan-Execute        Reflexion / Role-playing
  (纯 Prompt)            (Prompt + LangGraph)          (多轮对话循环)
  ★★☆                    ★★★                         ★★★
```

## 模块概览

| # | 模块 | 内容 | 包含范式 | 难度 | API Key |
|---|------|------|---------|:---:|:---:|
| 01 | `01_reasoning/` | 推理范式 | CoT, Self-Ask, ToT | ★★☆ | ❌ Ollama |
| 02 | `02_action/` | 行动范式 | ReAct, Plan-and-Execute | ★★★ | ❌ Ollama |
| 03 | `03_collaboration/` | 协作范式 | Reflexion, Role-playing | ★★★ | ❌ Ollama |

所有范式默认使用 Ollama 本地模型 `qwen3.5:2b`，无需 API Key 即可运行。
可选配置 `ALIYUN_API_KEY` 使用云端 Qwen 模型（效果更佳）。

## 文件清单

```
agent/
├── README.md                            ← 本文件
├── .env.example                         ← 环境变量模板
├── convert_py_to_ipynb.py              ← .py → .ipynb 转换脚本
├── __init__.py
├── helpers.py                           ← 共享辅助函数
│
├── 01_reasoning/                        ← 推理范式（纯 Prompt 方式）
│   ├── __init__.py
│   ├── chain_of_thought.py + .ipynb     CoT：直接回答 vs 一步步思考对比
│   ├── self_ask.py + .ipynb             Self-Ask：追问式拆问题
│   └── tree_of_thoughts.py + .ipynb     ToT：多分支生成 + 投票选优
│
├── 02_action/                           ← 行动范式（Prompt + LangGraph）
│   ├── __init__.py
│   ├── react.py + .ipynb                ReAct：手动循环 vs 框架自动循环对比
│   └── plan_and_execute.py + .ipynb     P&E：先计划再执行
│
├── 03_collaboration/                    ← 协作范式（多轮对话循环）
│   ├── __init__.py
│   ├── reflexion.py + .ipynb            Reflexion：生成 → 评估 → 反思 → 重试
│   └── role_playing.py + .ipynb         Role-playing：双角色辩论 + 三角色头脑风暴
│
└── tests/
    ├── __init__.py
    └── test_helpers.py                  工具函数单元测试
```

## 范式速查表

| 范式 | 一句话 | 核心机制 | 适用场景 |
|------|--------|---------|---------|
| **CoT** | 一步步写过程 | 添加"请一步步思考"到 prompt | 逻辑推理、数学计算 |
| **Self-Ask** | 拆成小问题 | 生成 Follow-up → 搜索 → 组合 | 事实链路长的问题 |
| **ToT** | 树状多分支 | 生成多个思路 → 评估 → 选最优 | 复杂规划、创意生成 |
| **ReAct** | 既思考也动手 | Thought → Action → Observation 循环 | 需与外部交互的问题 |
| **Plan-Execute** | 先计划再执行 | 生成计划列表 → 逐步执行 → 汇总 | 多步骤长时间任务 |
| **Reflexion** | 自我反思迭代 | 评估 → 反思"哪里错了" → 带着反思重试 | 代码生成、流程执行 |
| **Role-playing** | 多人协作分工 | 不同角色交替发言，达成共识 | 复杂系统开发、头脑风暴 |

## 运行前准备

1. **安装依赖**（与项目一致，无需额外包）：
   ```bash
   pip install langchain-core langchain-ollama langgraph pydantic python-dotenv
   ```

2. **启动 Ollama 并下载模型**：
   ```bash
   ollama serve
   ollama pull qwen3.5:2b
   ```

3. **（可选）配置云端模型**：
   ```bash
   cp .env.example .env
   # 编辑 .env，填入 ALIYUN_API_KEY
   ```

## 学习建议

| 如果你是... | 建议路线 |
|------------|---------|
| **初学者** | 先学 01_reasoning（纯 Prompt，理解范式本质），再学 02_action（ReAct 是最常用范式） |
| **已有 Agent 经验** | 直接跳到 02_action 对比 Prompt 版 vs LangGraph 版，然后 03_collaboration 学多智能体 |
| **只学一个范式** | 学 ReAct（02_action/react.py），它是最核心、最通用的 Agent 范式 |

## 常见问题

**Q: 这些范式和 LangChain/LangGraph 有什么关系？**
A: 这些范式是**认知模式**（LLM 应该如何思考和工作），LangChain/LangGraph 是**实现工具**。一些范式（如 ReAct）已被 LangChain、LangGraph 等框架内置为基础能力，但理解范式本身能帮你更好地使用框架。

**Q: 可以混搭使用吗？**
A: 完全可以。例如：Plan-and-Execute 的"计划阶段"用 ToT 生成多个方案，"执行阶段"用 ReAct 调用工具。范式之间不是互斥的。

**Q: 为什么要手动实现 Prompt 版 ReAct？框架不是已经内置了吗？**
A: 手动实现帮助你理解框架底层做了什么。当你遇到 Agent 循环不终止、工具调用失败等问题时，理解底层原理比会调用 API 更重要。
