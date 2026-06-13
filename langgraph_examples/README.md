# LangGraph 教学模块

## 学习路线

```
01_introduction         02_state_and_branching     03_agent_loop          04_workflows            05_practical
    │                        │                         │                      │                      │
    ▼                        ▼                         ▼                      ▼                      ▼
 概念认知  ──────────→  状态+条件分支  ──────────→  ReAct Agent  ──────────→  工作流模式  ──────────→  综合实战
 (无需API)              (无需API)                 ★核心技能              (需要LLM)              (需要LLM)
                                                    ├─ agent_react         ├─ prompt_chain
                                                    ├─ create_agent_demo   ├─ routing
                                                    │                      ├─ parallelization
                                                    │                      └─ evaluator_optimizer
```

## 模块概览

| # | 模块 | 内容 | 教学方式 | API Key |
|---|------|------|----------|:---:|
| 01 | `01_introduction/` | LangGraph 是什么 + 最简图 | 概念讲解 + 纯代码演示 | ❌ |
| 02 | `02_state_and_branching/` | 状态管理 + 条件分支 | 关键词匹配路由演示 | ❌ |
| 03 | `03_agent_loop/` | **ReAct Agent 循环 + 构建方式对比** | Agent 手动构建 vs create_react_agent() | ❌ Ollama |
| 04 | `04_workflows/` | 四大工作流模式全覆盖 | 提示链 / LLM 路由 / 并行化 / 评估器-优化器 | ❌ Ollama |
| 05 | `05_practical/` | 智能问答 Agent 综合实战 | 知识搜索 + 计算 + 日期 | ❌ Ollama |

## 文件清单

```
langgraph_examples/
├── README.md                           ← 本文件
├── .env.example                        ← 环境变量模板
├── convert_py_to_ipynb.py             ← .py → .ipynb 转换脚本
├── __init__.py
├── images/
│   └── graph.png
│
├── 01_introduction/                    ← LangGraph 是什么 + 最简图
│   ├── what_is_langgraph.py            ★ 概念认知（不依赖任何框架）
│   └── simple_graph.py + .ipynb        两个节点顺序执行
│
├── 02_state_and_branching/             ← 状态管理 + 条件分支
│   └── conditional_branch.py + .ipynb  关键词路由 + 三种回复分支
│
├── 03_agent_loop/                      ← ★ ReAct Agent 循环（LangGraph 核心）
│   ├── agent_react.py + .ipynb         数学工具 + 多工具组合
│   └── create_agent_demo.py + .ipynb   create_react_agent() vs 手动构建对比
│
├── 04_workflows/                       ← 工作流模式（四大模式全覆盖）
│   ├── prompt_chain.py + .ipynb        提示链（生成→评估→改进→润色）
│   ├── routing.py + .ipynb             LLM 智能路由（三路分支）
│   ├── parallelization.py + .ipynb     并行化（Send API 扇出-扇入）
│   └── evaluator_optimizer.py + .ipynb 评估器-优化器（循环改进）
│
├── 05_practical/                       ← 综合实战
│   └── search_qa_agent.py + .ipynb     知识库 + 计算 + 日期 Agent
│
├── tests/                              ← 测试
│   ├── __init__.py
│   ├── test_graph_helpers.py           11 个单元测试（tool_node + router）
│   └── test_graphs.py                  图结构集成测试
│
└── utils/                              ← 共享工具模块
    ├── __init__.py
    └── graph_helpers.py                get_model / build_react_agent / create_tool_node / ...
```

## 运行前准备

1. **安装依赖**：
   ```bash
   pip install langgraph langchain-core langchain-ollama
   # 部分模块需要：
   pip install langchain-openai pydantic
   ```

2. **安装 Ollama 并下载模型**（所有需要的模块默认使用 Ollama）：
   ```bash
   # 安装 Ollama：https://ollama.ai
   ollama pull qwen3.5:2b
   ```

3. **（可选）使用云端 API**：
   ```bash
   # 复制 .env.example 为 .env，填写阿里云 API Key
   # 然后将代码中的 get_model() 改为 get_model(use_cloud=True)
   ```

4. **运行测试**：
   ```bash
   pip install pytest
   pytest langgraph_examples/tests/ -v
   ```

## 关键规则

- **默认 Ollama 本地模型**：`qwen3.5:2b`，与 langchain_examples 保持一致，学生无需 API Key 即可运行
- **中文母语学生，Windows 为主**：注释、docstring、打印输出全部中文；变量/函数/类名英文 snake_case
- **代码即文档**：详细 docstring + 行内注释，每个函数独立可运行
- **生活化比喻**：StateGraph = "地铁图"、Node = "站点"、Edge = "轨道"、Tool = "员工"、LLM = "老板"
- **渐进式设计**：01（概念+最简图）→ 02（条件分支）→ 03（Agent 循环）→ 04（工作流）→ 05（综合实战）
- **共享工具**：`utils/graph_helpers.py` 提取了通用模式，教学文件不重复造轮子
- **双格式交付**：每个教学 .py 都有对应的 .ipynb（通过 `convert_py_to_ipynb.py` 生成）
