# LangGraph 教学模块

## 学习路线

```
01_introduction         02_state_and_branching     03_agent_loop          04_workflows            05_practical            06_multi_agent          07_deep_agents
    │                        │                         │                      │                      │                      │                      │
    ▼                        ▼                         ▼                      ▼                      ▼                      ▼                      ▼
 概念认知  ──────────→  状态+条件分支  ──────────→  ReAct Agent  ──────────→  工作流模式  ──────────→  综合实战  ──────────→  多智能体模式  ──────→  DeepAgents 框架
 (无需API)              (无需API)                 ★核心技能              (需要LLM)              (需要LLM)              (需要LLM)              (需要LLM+deepagents)
                                                    ├─ agent_react         ├─ prompt_chain          ├─ search_qa_agent       ├─ agent_as_tool
                                                    │                      ├─ routing                                        ├─ handoffs
                                                    │                      ├─ parallelization                                └─ supervisor
                                                    │                      └─ evaluator_optimizer
```

## 模块概览

| # | 模块 | 内容 | 教学方式 | API Key |
|---|------|------|----------|:---:|
| 01 | `01_introduction/` | LangGraph 是什么 + 最简图 | 概念讲解 + 纯代码演示 | ❌ |
| 02 | `02_state_and_branching/` | 状态管理 + 条件分支 | 关键词匹配路由演示 | ❌ |
| 03 | `03_agent_loop/` | **ReAct Agent 循环** | 手动构建，理解底层原理 | ✅ |
| 04 | `04_workflows/` | 四大工作流模式全覆盖 | 提示链 / LLM 路由 / 并行化 / 评估器-优化器 | ✅ |
| 05 | `05_practical/` | 智能问答 Agent 综合实战 | 知识搜索 + 计算 + 日期 | ✅ |
| 06 | `06_multi_agent/` | **三大多智能体模式** | Agent-as-Tool / Handoff / Supervisor | ✅ |
| 07 | `07_deep_agents/` | **DeepAgents 开箱即用框架** | 一行创建 + 文件系统 / 子代理 / Skills | ✅ |

## 文件清单

```
langgraph_examples/
├── README.md                           ← 本文件
├── .env.example                        ← 环境变量模板（ALIYUN_API_KEY）
├── __init__.py
├── practice_exam.md                    ← 练习题
├── images/                             ← 各教学文件运行时生成的可视化 PNG
│
├── 01_introduction/                    ← LangGraph 是什么 + 最简图
│   ├── what_is_langgraph.py            ★ 概念认知（不依赖任何框架）
│   └── simple_graph.py                 两个节点顺序执行
│
├── 02_state_and_branching/             ← 状态管理 + 条件分支
│   ├── conditional_branch.py           关键词路由 + 三种回复分支
│   └── reflect_demo.py                 反射机制演示
│
├── 03_agent_loop/                      ← ★ ReAct Agent 循环（LangGraph 核心）
│   └── agent_react.py                  数学工具 + 多工具组合 + create_react_agent 对比
│
├── 04_workflows/                       ← 工作流模式（四大模式全覆盖）
│   ├── prompt_chain.py                 提示链（生成→评估→改进→润色）
│   ├── routing.py                      LLM 智能路由（三路分支）
│   ├── parallelization.py              并行化（Send API 扇出-扇入）
│   └── evaluator_optimizer.py          评估器-优化器（循环改进）
│
├── 05_practical/                       ← 综合实战
│   └── search_qa_agent.py              知识库 + 计算 + 日期 Agent
│
├── 06_multi_agent/                     ← 多智能体模式（三大架构）
│   ├── agent_as_tool.py                Agent-as-Tool（主Agent调用子Agent为工具）
│   ├── handoffs.py                     Handoff 模式（控制权接力传递）
│   └── supervisor.py                   Supervisor 模式（主管分配+审查）
│
├── 07_deep_agents/                     ← DeepAgents 开箱即用框架
│   ├── __init__.py
│   ├── what_is_deepagent.py            ★ 概念认知（DeepAgents 定位 + 六大能力）
│   ├── quickstart.py                   最简入门 + 文件系统操作
│   ├── subagents_and_skills.py         子代理委派 + Skills 加载
│   └── README.md                       教学大纲
│
├── tests/                              ← 测试（pytest，17 个）
│   ├── __init__.py
│   ├── test_graph_helpers.py           单元测试（tool_node + router）
│   └── test_graphs.py                  图结构集成测试
│
└── utils/                              ← 共享工具模块
    ├── __init__.py
    └── graph_helpers.py                create_tool_node / create_router / build_react_agent / save_graph_png
```

## 运行前准备

1. **安装依赖**：
   ```bash
   pip install langgraph langchain-core langchain-openai pydantic
   # 模块 06（多智能体）用原生 langgraph 实现，无需额外包
   # 模块 07（DeepAgents）需要额外安装：
   pip install deepagents
   # 所有模式均用原生 langgraph 实现，教底层原理而非黑盒
   ```

2. **配置阿里云 API Key**（需要 LLM 的模块使用云端 Qwen 模型）：
   ```bash
   # 复制 .env.example 为 .env，填写 ALIYUN_API_KEY
   # 未配置时示例会打印提示并跳过；纯逻辑示例（01/02）无需 Key
   ```

4. **运行测试**：
   ```bash
   pip install pytest
   pytest langgraph_examples/tests/ -v
   ```

## 关键规则

- **默认云端 Qwen 模型**：`qwen-plus`，需配置 `ALIYUN_API_KEY`；未配置时示例自动跳过，纯逻辑示例（01/02）无需 Key
- **中文母语学生，Windows 为主**：注释、docstring、打印输出全部中文；变量/函数/类名英文 snake_case
- **代码即文档**：详细 docstring + 行内注释，每个函数独立可运行
- **生活化比喻**：StateGraph = "地铁图"、Node = "站点"、Edge = "轨道"、Tool = "员工"、LLM = "老板"
- **渐进式设计**：01（概念+最简图）→ 02（条件分支）→ 03（Agent 循环）→ 04（工作流）→ 05（综合实战）
- **共享工具**：`utils/graph_helpers.py` 提取了通用模式，教学文件不重复造轮子
