# CLAUDE.md - 沃林 AI 课程项目

## 项目概述

这是一个 AI 技术教学项目，面向已掌握 Python 基础、大模型基础、调用大模型 API、通过 Dify 构建 Agent 和知识库的学生。项目涵盖多个 AI 技术模块，包含 **LangChain 基础教程**、**Milvus 向量数据库** 和 **RAG（检索增强生成）** 三大教学主线。

## 目录结构

```
wolin_learn-master/
├── langchain_examples/    # ★ LangChain 渐进式教程（9 个模块，28 个教学文件）
├── rag_examples/          # ★ RAG + Milvus 渐进式教程（9 个模块，~30+ 教学单元）
├── rag_demo/              # ★ 综合实战项目：双集合混合检索 RAG 系统
├── demo/                  # 早期简化版 demo（类似 rag_demo 的简化版）
├── langgraph_examples/    # LangGraph 工作流示例
├── cloud_api_examples/    # 云端 API 调用示例
├── ollama_examples/       # Ollama 本地大模型调用
├── neo4j_examples/        # Neo4j 图数据库示例
├── mcp_examples/          # MCP (Model Context Protocol) 示例
├── vllm_examples/         # vLLM 推理引擎示例
├── utils/                 # 共享工具模块
└── CLAUDE.md              # 本文件
```

## 当前工作重点

### langchain_examples/ — LangChain 基础教程（2026-06-13 完成改造）

9 个模块按渐进式难度排列，同时提供 `.py` 和 `.ipynb` 两种格式：

| 模块 | 内容 | 文件 |
|------|------|------|
| `01_introduction/` | 什么是 LangChain + 第一个程序 | `what_is_langchain`, `first_chain` |
| `02_llm_call/` | LLM 基础调用 / Chat Model / 流式输出 | `llm_basic`, `chat_model`, `streaming_output` |
| `03_prompt/` | PromptTemplate / Few-Shot 提示词 | `prompt_template`, `few_shot_prompt` |
| `04_output_parser/` | StrOutputParser / JsonOutputParser / PydanticOutputParser | `string_parser`, `json_parser`, `pydantic_parser` |
| `05_memory/` | 对话记忆 / Buffer Memory | `buffer_memory`, `conversation_memory` |
| `06_chains/` | LCEL Pipeline / 顺序链 / 路由链 | `simple_chain`, `sequential_chain`, `router_chain` |
| `07_retrieval/` | 文档加载 / 向量存储 / RAG 基础 | `document_loader`, `vector_store`, `rag_basic` |
| `08_project/` | 综合实战：文档问答 + 研究助手 | `qna_bot`, `research_assistant` |
| `09_agent/` | Agent / Tool / Memory / Middleware / HITL | `agent`, `tools`, `agent_memory`, `middleware`, `human_in_the_loop` |

关键文件：
- `convert_py_to_ipynb.py` — `.py` → `.ipynb` 批量转换脚本（状态机架构）
- `utils/model_utils.py` — 共享模型客户端（`get_qwen_client`）
- `utils/__init__.py` — 包导出
- `学习路线.py` — 学习路径规划
- `install_deps.bat` — Windows 依赖安装脚本
- `requirements.txt` — Python 依赖清单

### rag_examples/ — RAG + Milvus 主课程

课程模块按渐进式难度排列：

| 模块 | 内容 | 格式 |
|------|------|------|
| `00_setup/` | 环境搭建（Docker + Python + API Key） | `.md` |
| `embedding_examples/` | Embedding 基础/阿里云 API/本地模型/对比 | `.py` + `.ipynb` |
| `01_milvus_basics/` | Milvus 连接/集合/插入/索引 | `.py` + `.ipynb` + `.md` |
| `02_document_chunking/` | 固定/滑动窗口/AI/摘要切片 + 对比 | `.py` + `.ipynb` |
| `03_retrieval_methods/` | 标量/向量/关键词/混合检索 + Rerank | 仅有 `.ipynb` |
| `04_rag_api/` | RAG 检索器 API 封装 + Q&A API 封装 | 仅有 `.ipynb` |
| `05_rag_pipeline/` | 最小 RAG / 分步讲解 / 完整管道 | `.py` + `.ipynb` |
| `06_rag_advanced/` | BM25 Function / 双集合设计 / Mock→Real 过渡 | `.py` + `.md` |
| `06_rag_evaluation/` | RAGAS 四指标评估 + RAGEvaluator 类 | `.py` + `.md` |

关键文件：
- `milvus_config.py` — 统一配置中心（从环境变量读取 Milvus URI、维度=1024、度量类型）
- `run_tests.py` — 语法和函数签名验证（支持 .py 和 .ipynb，23/23 语法通过）
- `convert_py_to_ipynb.py` — `.py` → `.ipynb` 格式转换工具
- `MILVUS_CONFIG.md` — Milvus 部署方式说明（含 Windows 警告、实例隔离建议）
- `CHEATSHEET.md` — Embedding 模型/索引/切片策略速查表
- `.env.example` — 环境变量模板
- `utils/` — 共享工具（`ensure_env_loaded`、`get_api_key`）
- `data/` — 全文数据（txt/json/structured）

### rag_demo/ — 综合实战项目

```
rag_demo/
├── config.py                       # 共享配置（Milvus 客户端、集合名常量）
├── README.md                       # 项目说明
├── core/
│   ├── rag_query.py               # 完整 RAG QA（双集合混合检索 + DeepSeek LLM）
│   └── file_chunk_retrieval.py    # 文档块混合检索（稠密 + BM25 + RRF）
├── db/
│   └── vdb_init_milvus.py         # Milvus 初始化（schema 设计 + 数据摄入）
├── util/
│   ├── __init__.py                # 导出所有工具函数
│   ├── embedding.py               # ★ 共享 Embedding 模块（text-embedding-v4, 1024 维）
│   ├── text_parser.py             # TXT/PDF/DOCX 解析
│   └── text_splitter.py           # 递归文本切片（基于 LangChain）
├── tests/
│   ├── conftest.py                # pytest 配置
│   ├── test_embedding.py          # Embedding 测试（mock + 真实 API）
│   ├── test_text_parser.py        # 文本解析测试
│   └── test_text_splitter.py      # 文本切片测试
├── .env.example                   # 环境变量模板
└── datas/
    ├── 三国演义.txt                # 原始语料（1.75 MB）
    ├── qa_paris.json              # 50 条三国 QA 对
    └── qa_paris_additional.json   # 增量测试数据（3 条三国 QA）
```

## 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| LLM 调用 | Ollama (`qwen3.5:2b`) | langchain_examples 本地教学默认模型 |
| LLM SDK | LangChain 1.0+ (`create_agent`) | langchain_examples 框架 |
| Agent 工具 | LangGraph (`InMemorySaver`, `ToolRuntime`) | langchain_examples Agent 模块 |
| 向量数据库 | Milvus 2.4+ (pymilvus) | 通过环境变量 `MILVUS_URI` 配置（默认 localhost:19530） |
| Embedding | 阿里云 DashScope `text-embedding-v4` | 1024 维，通过 OpenAI 兼容接口调用 |
| LLM | DeepSeek (`deepseek-chat`) | rag_demo 中用于生成答案 |
| LLM | 阿里云百炼 Qwen (`qwen-plus`) | rag_examples 中用于 AI 切片/摘要 |
| 文本切片 | LangChain `RecursiveCharacterTextSplitter` | rag_demo 中使用 |
| 关键词检索 | BM25 | 自实现 + Milvus 原生 Function |
| 本地 Embedding | sentence-transformers, FlagEmbedding | embedding_examples 和 rerank 中使用 |
| 分词 | jieba | BM25 中文分词 |

## 编码规范

### 语言
- **注释、docstring、打印输出、README**：全部使用中文
- **函数名、变量名、类名**：英文，遵循 Python 惯例（snake_case / PascalCase）
- **学生群体**：中文母语，Windows 为主

### 教学代码风格
- 每个 `.py` 文件应尽量**自包含**，但共享逻辑（embedding、config）需提取到共享模块
- 每个文件末尾应有 `if __name__ == "__main__":` 代码块，提供可运行的演示
- 关键配置（API 密钥、服务器地址）通过 `os.getenv()` 从环境变量读取，提供合理默认值
- 复杂概念使用 **生活化中文比喻** 解释（如"Excel表格""切蛋糕""面试筛选""乐高积木""餐厅服务员"）
- 函数应有中文 docstring 说明参数和返回值
- 每个教学文件 2-3 个精选示例，每个示例配一个生活化比喻
- 示例间用 `print(f"\n-- 示例 N: 标题")` 分隔，保持简洁

### 环境变量约定
```
# langchain_examples
ALIYUN_API_KEY    # 阿里云百炼/ DashScope API Key（云端 API 示例用）

# rag_demo
DEEPSEEK_API_KEY  # DeepSeek API Key

# rag_examples + rag_demo 共享
MILVUS_URI        # Milvus 连接地址（默认 http://localhost:19530）
MILVUS_DB_NAME    # Milvus 数据库名（默认 default）
```

### 向量维度约定
- **统一使用 1024 维**（`text-embedding-v4` 默认输出维度）
- 配置位置：`rag_examples/milvus_config.py` 的 `DEFAULT_DIMENSION` 和 `rag_demo/util/embedding.py` 的 `DEFAULT_EMBEDDING_DIMENSION`

## 修改代码时的注意事项

1. **修改维度前先审计**：搜索 `1024`、`DEFAULT_DIMENSION`、`DEFAULT_EMBEDDING_DIMENSION` 确保全部一致。
2. **修改 Milvus URI 前先理解影响范围**：`milvus_config.py` 和 `rag_demo/config.py` 的 `MILVUS_URI` 被所有文件引用。
3. **编辑 .ipynb 文件使用 NotebookEdit 工具**：不要用 Read/Write 直接编辑 JSON。
4. **rag_examples 和 rag_demo 的 Embedding 维度必须一致**：两者都使用 1024 维。
5. **新增模块需同步更新**：`run_tests.py` 的 `modules_*` 列表、`convert_py_to_ipynb.py` 的 `FILES_TO_CONVERT` 列表。
6. **rag_demo 依赖 rag_examples 的基础知识**：修改 rag_demo 的高级特性时，确保 rag_examples 有对应的基础内容做铺垫。
7. **项目使用 `.venv` 虚拟环境**，`rag_examples/requirements.txt` 列出依赖。
8. **langchain_examples 的 .py 修改后需重新生成 .ipynb**：运行 `python langchain_examples/convert_py_to_ipynb.py`。
9. **Windows 编码样板代码**：.py 文件中的 `sys.stdout = io.TextIOWrapper(...)` 仅用于终端，转换为 .ipynb 时会自动剥离。
10. **langchain_examples 使用 Ollama 本地模型**（`qwen3.5:2b`）作为默认，不想依赖本地模型的示例用 `get_qwen_client()` 调用云端 API。

## 改造历史

### langchain_examples 改造（2026-06-13）
- **Phase 1: 代码优化** — 27 个 .py 文件精简（~7500→~4200 行），删除 test.py 和 pipeline_prompt.py
- 创建 `utils/model_utils.py` 共享模块，消除 `sys.path` hack
- 全局统一分隔符格式：100+ 处 3 行 `print` 块 → 1 行格式
- `first_chain.py` 移除 Agent/HITL 高级代码，回归入门定位
- **Phase 2: .ipynb 转换** — 编写 `convert_py_to_ipynb.py`，24 个 .py → .ipynb 批量转换
- 自动剥离 Windows 编码样板、docstring 标题提取、裸文档字符串检测
- `human_in_the_loop.py` 保留 .py（交互式 `input()` 不适合 notebook）
- 全部 215 个单元验证通过（JSON 有效/函数完整/中文无乱码/无空单元）

### rag_examples + rag_demo 修复（2026-06-11）
- **Phase 1: 必须修复** — 维度统一（1024）、环境变量、代码去重、.env.example
- **Phase 2: 改进** — mock→真实衔接、格式工作流、Windows 警告、死代码清理
- **Phase 3: 新增** — CHEATSHEET、rag_demo 测试（40 个 pytest）、RAG 评估模块、过渡模块、00_setup、共享 utils

## 审查计划文件

所有待修复问题和详细建议在：
`.claude/plans/rag-demo-rag-examples-milvus-rag-python-sharded-hippo.md`
