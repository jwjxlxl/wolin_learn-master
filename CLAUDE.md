# CLAUDE.md — 沃林 AI 课程项目

## 快速定位

| 我想... | 去这里 |
|---------|--------|
| 修改 LangChain 教学内容 | `langchain_examples/` → 改 `.py` → 运行 `convert_py_to_ipynb.py` |
| 修改 LangGraph/Agent 教学内容 | `langgraph_examples/` → 改 `.py` → 运行 `convert_py_to_ipynb.py` |
| 修改 RAG/Milvus 教学内容 | `rag_examples/`（基础）→ `rag_demo/`（实战） |
| 添加新的教学模块 | 遵循对应子目录的格式规范（见下方"关键规则"） |
| 获取通用模型/工具 | `utils/model_utils.py`（`get_model` / `get_qwen_client`） |
| 理解项目约定 | 阅读本文档"关键规则"和"常见反模式" |
| 了解之前改了什么 | 跳到末尾"改造历史" |

## 关键规则

### 全局约束（所有子项目必须遵守）

- **环境变量优先**：API Key、Milvus URI、数据库名一律用 `os.getenv()` 读取，提供合理默认值。**绝不硬编码** IP 地址、密码、API Key。
- **向量维度 = 1024**：与 `text-embedding-v4` 一致。改维度前审计全项目所有 `1024`、`DEFAULT_DIMENSION`、`DEFAULT_EMBEDDING_DIMENSION` 引用。**1024 维向量插入非 1024 维 Collection 会直接失败**。
- **中文母语学生，Windows 为主**：注释、docstring、打印输出全部中文；变量/函数/类名英文 snake_case。
- **双格式 (.py + .ipynb)**：`.py` 是源文件（规范格式），`.ipynb` 通过转换脚本生成。**修改代码改 .py，不要直接改 .ipynb JSON**。

### langchain_examples/ 专属（LangChain 教程，9 模块）

| # | 模块 | 内容 | 默认模型 |
|---|------|------|----------|
| 01 | `introduction` | 概念 + 第一个程序 | Ollama `qwen3.5:2b` |
| 02 | `llm_call` | LLM / Chat / 流式 | Ollama `qwen3.5:2b` |
| 03 | `prompt` | PromptTemplate / Few-Shot | Ollama `qwen3.5:2b` |
| 04 | `output_parser` | Str/Json/Pydantic Parser | Ollama `qwen3.5:2b` |
| 05 | `memory` | 对话记忆 | Ollama `qwen3.5:2b` |
| 06 | `chains` | LCEL / 顺序链 / 路由链 | Ollama `qwen3.5:2b` |
| 07 | `retrieval` | 文档加载 / 向量存储 / RAG | Ollama `qwen3.5:2b` |
| 08 | `project` | 文档问答 / 研究助手 | Ollama `qwen3.5:2b` |
| 09 | `agent` | Agent / Tool / Memory / Middleware / HITL | Ollama + 云端 Qwen |

- **默认使用 Ollama 本地模型 `qwen3.5:2b`**，确保学生无需 API Key 即可运行。
- 云端 API 示例（`first_chain.cloud_api_call()`、`human_in_the_loop.py`）用 `utils/model_utils.get_qwen_client()` 统一获取客户端。
- **`human_in_the_loop.py` 不转 .ipynb**（交互式 `input()` 不适合 notebook）。
- **`utils/` 和 `学习路线.py` 不转 .ipynb**（工具/元信息文件）。
- 每个教学文件 2-3 个示例，用 `print(f"\n-- 示例 N: 标题")` 分隔。

### langgraph_examples/ 专属（LangGraph Agent 教程，5 模块）

| # | 模块 | 内容 | 默认模型 |
|---|------|------|----------|
| 01 | `01_introduction/` | 概念 + 最简图 | 无需模型 |
| 02 | `02_state_and_branching/` | 状态管理 + 条件分支 | 无需模型 |
| 03 | `03_agent_loop/` | ReAct Agent 循环 + create_agent 对比 | Ollama `qwen3.5:2b` |
| 04 | `04_workflows/` | 提示链 / 路由 / 并行化 / 评估器-优化器 | Ollama `qwen3.5:2b` |
| 05 | `05_practical/` | 智能问答 Agent 综合实战 | Ollama `qwen3.5:2b` |

- **默认使用 Ollama 本地模型**：通过 `utils/model_utils.get_model()` 获取（默认 `qwen3.5:2b`）。
- 云端 API 切换：`get_model(use_cloud=True)` → 自动使用 `get_qwen_client()`。
- `.ipynb` 转换脚本：`convert_py_to_ipynb.py`（9 个 notebook）。
- 测试：`tests/`，pytest 框架（11 个测试）。
- 共享工具：`utils/graph_helpers.py`（`create_tool_node` / `create_router` / `build_react_agent`）。
- **`what_is_langgraph.py` 不转 .ipynb**（纯概念文件，类似 `what_is_langchain.py`）。
- 每个教学文件含 2 个示例（1 个 LLM + 1 个纯逻辑），确保无 API Key 也能跑通至少一个。

### rag_examples/ 专属（RAG + Milvus 主课程）

| # | 模块 | 内容 | 默认模型 |
|---|------|------|----------|
| 00 | `00_setup/` | 环境搭建指南 | 无需模型 |
| 01 | `01_milvus_basics/` | Milvus 连接 / Collection / 插入 / 索引 | 模拟向量 |
| 02 | `02_document_chunking/` | 固定切片 / 滑动窗口 / AI 切片 / 概要 / 对比 | 可选 Qwen |
| 03 | `03_retrieval_methods/` | 标量查询 / 向量 / 关键字 / 混合 / Rerank | 模拟向量 |
| 04 | `04_rag_api/` | RAG 检索 API + 问答 API 封装 | Qwen |
| 05 | `05_rag_pipeline/` | RAG 最小版 / 分步详解 / 完整流程 | Qwen |
| 06 | `06_rag_advanced/` | 混合检索进阶 / 双集合设计 / Mock→Real | Qwen |
| 07 | `06_rag_evaluation/` | RAGAS 四指标评估 | 无需模型 |
| — | `embedding_examples/` | Embedding 基础 / 阿里云 / 本地 / 对比 | 阿里云 API |

- 配置入口：`milvus_config.py`（`DEFAULT_DIMENSION = 1024`、`MILVUS_URI`、`MILVUS_DB_NAME`）
- 导入方式：**统一使用** `from rag_examples.milvus_config import MILVUS_URI, DEFAULT_DIMENSION`，**禁止** `sys.path.insert()` hack 导入包内模块
- `.ipynb` 转换脚本：`convert_py_to_ipynb.py`（8 个模块 27 个 notebook）
- 语法验证：`run_tests.py`（同时支持 .py 和 .ipynb，含降级逻辑）
- 共享工具：`utils/helpers.py`（`ensure_env_loaded` / `get_api_key` / `safe_milvus_operation` / `format_score` / `truncate_text`）
- 新增模块后需同步更新 `run_tests.py` 的 `modules_*` 列表和 `convert_py_to_ipynb.py` 的 `files_to_convert` 列表
- `.py` 是源文件（规范格式），`.ipynb` 通过转换脚本生成。**所有模块必须有对应的 .py 文件**
- 每个教学文件用 `print(f"\n-- 示例 N: 标题")` 分隔

### rag_demo/ 专属（综合实战项目）

- 配置入口：`config.py`（共享 Milvus 客户端、集合名常量）
- Embedding：**导入** `util/embedding.py`，不要在各模块重复定义 `generate_embedding()`
- 测试：`tests/`，pytest 框架（40 个测试）

## 常见反模式（绝对不要做）

| # | ❌ 错误做法 | ✅ 正确做法 | 原因 |
|---|-----------|-----------|------|
| 1 | 在子模块文件中用 `sys.path.insert()` hack 导入其他子模块的函数 | 共享代码放项目根 `utils/`；教学文件统一 `sys.path.insert(0, os.path.join(...))` 仅用于导入 `utils/` | 跨子模块 hack 已全清理；导入 `utils/` 的路径设置是必要且统一的 |
| 2 | `print(f"\n{'─'*50}")\nprint("标题")\nprint(f"{'─'*50}")` | `print(f"\n-- 标题")` | 3 行冗余，刚全局清洗过 100+ 处 |
| 3 | 用 Read/Write 直接编辑 .ipynb JSON | NotebookEdit 工具 / 修改 .py 后重新转换 | 直接编辑易破坏 JSON 结构 |
| 4 | 在多个文件重复定义相同函数 | 提取到 `util/` 共享模块 | 已去重 `generate_embedding()`（原 3 处重复） |
| 5 | 在入门文件（`01_introduction/`）加入 Agent/HITL/Memory 代码 | 入门只讲基础调用，高级概念留在对应模块 | 已从 `first_chain.py` 剥离 Agent 代码 |
| 6 | 硬编码 `47.115.57.130` 等 IP / 数据库名 | `os.getenv("MILVUS_URI")` | 安全风险 + 多学生冲突，已全部替换 |
| 7 | 用玩笑/人身数据做测试用例 | 用与课程主题相关的专业数据 | `qa_paris_additional.json` 已替换为三国 QA |

## 技术栈与约束

| 组件 | 技术 | 约束 |
|------|------|------|
| 本地 LLM | Ollama `qwen3.5:2b` | langchain_examples + langgraph_examples 默认模型，需本地运行 `ollama serve` |
| 云端 LLM | 阿里云 Qwen `qwen-plus` | rag_examples AI 切片/摘要 + langgraph 可选云端模式；需 `ALIYUN_API_KEY` |
| 云端 LLM | DeepSeek `deepseek-chat` | rag_demo QA 生成；需 `DEEPSEEK_API_KEY` |
| LLM 框架 | LangChain 1.0+ (`create_agent`, LCEL) | langchain_examples 使用；**已弃用** `ConversationBufferMemory`、`SequentialChain` |
| Agent 框架 | LangGraph (`StateGraph`, `Send`, `create_react_agent`) | langgraph_examples 核心 + langchain_examples 09_agent 使用 |
| 共享工具 | `utils/model_utils.py` | `get_model()`（Ollama 默认）和 `get_qwen_client()`（云端 Qwen）的统一入口 |
| 向量数据库 | Milvus 2.4+ (pymilvus) | `MILVUS_URI` 配置；Windows 不支持 Lite 版，需 Docker |
| Embedding | 阿里云 `text-embedding-v4` | **固定 1024 维**，通过 OpenAI 兼容接口调用 |
| 文本切片 | LangChain `RecursiveCharacterTextSplitter` | rag_demo 使用 |
| 分词 | jieba | BM25 中文分词 |
| 本地 Embedding | sentence-transformers, FlagEmbedding | embedding_examples 对比实验用 |

## 环境变量

```
# langchain_examples（云端 API 示例用）
ALIYUN_API_KEY=your_dashscope_api_key

# rag_demo（问答生成用）
DEEPSEEK_API_KEY=your_deepseek_api_key

# rag_examples + rag_demo 共享（Milvus 连接）
MILVUS_URI=http://localhost:19530
MILVUS_DB_NAME=default
```

各子项目有独立的 `.env.example` 模板文件。

## 改造历史

### 2026-06-13 — langgraph_examples 三阶段重构

- **Phase 1 — 工程基础**：修复 `model_untils` 拼写错误，消除 `sys.path` hack，创建 5 子模块结构（`01_introduction` ~ `05_practical`），提取共享函数到 `utils/graph_helpers.py`
- **Phase 2 — 教学体验**：全部文件默认 Ollama 本地模型（`get_model()`），统一分隔符格式，`.py→.ipynb` 批量转换（9 个 notebook），README 教学大纲
- **Phase 3 — 内容充实**：新增并行化（Send API）、评估器-优化器、`create_react_agent()` 对比示例，11 个 pytest 测试
- **全局影响**：`get_model()` 提升至 `utils/model_utils.py`（跨子项目共享），`utils/model_untils.py` 更名为 `utils/model_utils.py`，mcp_examples 四个文件随同修正导入
- 详情见 `langgraph_examples/README.md`

### 2026-06-13 — langchain_examples 全面优化

- **代码精简**：27 个 .py 文件 ~7500→~4200 行（-44%），删除 `test.py`、`pipeline_prompt.py`
- **架构改进**：创建 `utils/model_utils.py` 共享模块，消除所有 `sys.path` hack
- **导入路径修复**：09_agent 全部文件改为标准包导入
- **入门文件去杂**：`first_chain.py` 移除 Agent/HITL 代码，回归基础
- **分隔符统一**：100+ 处 3 行 `print` 块 → 1 行格式
- **.ipynb 批量生成**：编写 `convert_py_to_ipynb.py`（状态机架构），24/24 转换成功
- **自动剥离**：Windows 编码样板（`sys.stdout = io.TextIOWrapper(...)`）、空单元检测
- **验证**：215 个单元，JSON 有效、函数完整、中文无乱码、无空单元

### 2026-06-13 — rag_examples 全面规范化修复

- **P0 — 阻塞性修复**：`rag_step_by_step.py` 768→1024 维统一；`02_create_collection.ipynb` 清除硬编码 IP/db_name/768（6 个 cell）；为 02/03/04 模块补建 12 个 .py 源文件
- **P1 — 工程规范**：`__init__.py` 移除 sys.path hack 改用相对导入；`rag_full_pipeline.py` 删除重复 sys.path；05_rag_pipeline 统一导入方式（3 文件）；`rag_step_by_step.py` 7 处分隔符统一；`utils/__init__.py` 补全导出；`convert_py_to_ipynb.py` 补全 06 系列模块 + 修复硬编码路径
- **P2 — 教学润色**：`01_connect_milvus.py` 默认密码安全注释；`01_embedding_basics.py` 维度标准标注

### 2026-06-11 — rag_examples + rag_demo 修复与增强

- **维度统一**：全部改为 1024 维，修正 13+ 处不一致
- **环境变量化**：移除所有硬编码 IP 和数据库名
- **代码去重**：`generate_embedding()` 提取到共享模块，消除 3 处重复
- **新增模块**：`00_setup`（环境搭建）、`06_rag_advanced`（BM25/双集合/Mock→Real）、`06_rag_evaluation`（RAGAS 四指标）、共享 `utils/`
- **测试**：rag_demo 40 个 pytest 测试
- **文档**：CHEATSHEET、MILVUS_CONFIG（Windows 警告 + 实例隔离）、.env.example
