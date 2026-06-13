# CLAUDE.md — 沃林 AI 课程项目

## 快速定位

| 我想... | 去这里 |
|---------|--------|
| 修改 LangChain 教学内容 | `langchain_examples/` → 改 `.py` → 运行 `convert_py_to_ipynb.py` |
| 修改 RAG/Milvus 教学内容 | `rag_examples/`（基础）→ `rag_demo/`（实战） |
| 添加新的教学模块 | 遵循对应子目录的格式规范（见下方"关键规则"） |
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

### rag_examples/ 专属（RAG + Milvus 主课程）

- 配置入口：`milvus_config.py`（`DEFAULT_DIMENSION = 1024`、`MILVUS_URI`、`MILVUS_DB_NAME`）
- `.ipynb` 转换脚本：`convert_py_to_ipynb.py`
- 语法验证：`run_tests.py`（同时支持 .py 和 .ipynb）
- 新增模块后需同步更新 `run_tests.py` 的 `modules_*` 列表

### rag_demo/ 专属（综合实战项目）

- 配置入口：`config.py`（共享 Milvus 客户端、集合名常量）
- Embedding：**导入** `util/embedding.py`，不要在各模块重复定义 `generate_embedding()`
- 测试：`tests/`，pytest 框架（40 个测试）

## 常见反模式（绝对不要做）

| # | ❌ 错误做法 | ✅ 正确做法 | 原因 |
|---|-----------|-----------|------|
| 1 | 用 `sys.path.insert()` hack 跨模块导入 | 用 `utils/` 共享模块 + 标准包导入 | hack 脆弱、IDE 不识别，已全部清理 |
| 2 | `print(f"\n{'─'*50}")\nprint("标题")\nprint(f"{'─'*50}")` | `print(f"\n-- 标题")` | 3 行冗余，刚全局清洗过 100+ 处 |
| 3 | 用 Read/Write 直接编辑 .ipynb JSON | NotebookEdit 工具 / 修改 .py 后重新转换 | 直接编辑易破坏 JSON 结构 |
| 4 | 在多个文件重复定义相同函数 | 提取到 `util/` 共享模块 | 已去重 `generate_embedding()`（原 3 处重复） |
| 5 | 在入门文件（`01_introduction/`）加入 Agent/HITL/Memory 代码 | 入门只讲基础调用，高级概念留在对应模块 | 已从 `first_chain.py` 剥离 Agent 代码 |
| 6 | 硬编码 `47.115.57.130` 等 IP / 数据库名 | `os.getenv("MILVUS_URI")` | 安全风险 + 多学生冲突，已全部替换 |
| 7 | 用玩笑/人身数据做测试用例 | 用与课程主题相关的专业数据 | `qa_paris_additional.json` 已替换为三国 QA |

## 技术栈与约束

| 组件 | 技术 | 约束 |
|------|------|------|
| 本地 LLM | Ollama `qwen3.5:2b` | langchain_examples 默认模型，需本地运行 `ollama serve` |
| 云端 LLM | 阿里云 Qwen `qwen-plus` | rag_examples AI 切片/摘要；需 `ALIYUN_API_KEY` |
| 云端 LLM | DeepSeek `deepseek-chat` | rag_demo QA 生成；需 `DEEPSEEK_API_KEY` |
| LLM 框架 | LangChain 1.0+ (`create_agent`, LCEL) | langchain_examples 使用；**已弃用** `ConversationBufferMemory`、`SequentialChain` |
| Agent 状态 | LangGraph (`InMemorySaver`, `ToolRuntime`) | langchain_examples 09_agent 使用 |
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

### 2026-06-13 — langchain_examples 全面优化

- **代码精简**：27 个 .py 文件 ~7500→~4200 行（-44%），删除 `test.py`、`pipeline_prompt.py`
- **架构改进**：创建 `utils/model_utils.py` 共享模块，消除所有 `sys.path` hack
- **导入路径修复**：09_agent 全部文件改为标准包导入
- **入门文件去杂**：`first_chain.py` 移除 Agent/HITL 代码，回归基础
- **分隔符统一**：100+ 处 3 行 `print` 块 → 1 行格式
- **.ipynb 批量生成**：编写 `convert_py_to_ipynb.py`（状态机架构），24/24 转换成功
- **自动剥离**：Windows 编码样板（`sys.stdout = io.TextIOWrapper(...)`）、空单元检测
- **验证**：215 个单元，JSON 有效、函数完整、中文无乱码、无空单元

### 2026-06-11 — rag_examples + rag_demo 修复与增强

- **维度统一**：全部改为 1024 维，修正 13+ 处不一致
- **环境变量化**：移除所有硬编码 IP 和数据库名
- **代码去重**：`generate_embedding()` 提取到共享模块，消除 3 处重复
- **新增模块**：`00_setup`（环境搭建）、`06_rag_advanced`（BM25/双集合/Mock→Real）、`06_rag_evaluation`（RAGAS 四指标）、共享 `utils/`
- **测试**：rag_demo 40 个 pytest 测试
- **文档**：CHEATSHEET、MILVUS_CONFIG（Windows 警告 + 实例隔离）、.env.example
