# CLAUDE.md — 沃林 AI 课程项目

## 快速定位

| 我想... | 去这里 |
|---------|--------|
| 修改 LangChain 教学内容 | `langchain_examples/` |
| 修改 LangGraph/Agent 教学内容 | `langgraph_examples/` |
| 修改 RAG/Milvus 教学内容 | `rag_examples/`（基础）→ `rag_demo/`（实战） |
| 学习 DeepEval 评估思想 | `deepeval_examples/`（回答评分 → RAG 诊断 → 安全门禁，3 个案例） |
| 学习 Langfuse 可观测性 | `langfuse_demo/`（基础追踪 → RAG 分层追踪 → 提示词与实验，3 课 + Agent 进阶选学） |
| 分词 / 对象存储 / 缓存小示例 | `jieba_example/` · `minio_example/` · `redis_example/` |
| 添加新的教学模块 | 遵循对应子目录的格式规范（见下方"关键规则"） |
| 获取通用模型/工具 | `utils/model_utils.py`（`get_model` / `get_qwen_client`） |
| 理解项目约定 | 阅读本文档"关键规则"和"常见反模式" |
| 了解之前改了什么 | [CHANGELOG.md](CHANGELOG.md) |

> `hand_code/` 为临时实验代码（.ipynb），不作为教学交付物，不适用本文档约定。

## 关键规则

### 全局约束（所有子项目必须遵守）

- **环境变量优先**：API Key、Milvus URI、数据库名一律用 `os.getenv()` 读取，提供合理默认值。**绝不硬编码** IP 地址、密码、API Key。
- **向量维度 = 1024**：与 `text-embedding-v4` 一致。改维度前审计全项目所有 `1024`、`DEFAULT_DIMENSION`、`DEFAULT_EMBEDDING_DIMENSION` 引用。**1024 维向量插入非 1024 维 Collection 会直接失败**。
- **中文母语学生，Windows 为主**：注释、docstring、打印输出全部中文；变量/函数/类名英文 snake_case。
- **教学代码只维护 `.py`**：不再新增、生成或同步 `.ipynb`，不运行 `.py → .ipynb` 转换脚本；历史 Notebook 不作为交付要求。
- **教学体量约束**：单模块 ≤ 7 个文件，单教学文件 2-4 个示例。新内容先问"能否并入现有示例"，超限时拆分主题或标注"进阶选学"。
- **示例分隔**：每个教学文件用 `print(f"\n-- 示例 N: 标题")` 分隔各示例。
- **无 Key 优雅降级**：依赖云端 API 的示例，未配置对应 Key 时打印获取指引并跳过 LLM 调用，不抛异常。

### langchain_examples/ 专属（LangChain 教程，9 模块）

| # | 模块 | 内容 |
|---|------|------|
| 01 | `introduction` | 概念 + 第一个程序 |
| 02 | `llm_call` | LLM / Chat / 流式 |
| 03 | `prompt` | PromptTemplate / Few-Shot |
| 04 | `output_parser` | Str/Json/Pydantic Parser |
| 05 | `memory` | 对话记忆 |
| 06 | `chains` | LCEL / 顺序链 / 路由链 |
| 07 | `retrieval` | 文档加载 / 向量存储 / RAG |
| 08 | `project` | 文档问答 / 研究助手 |
| 09 | `agent` | Agent / Tool / Memory / Middleware / MCP / HITL |

- **默认 Ollama 本地模型 `qwen3.5:2b`**，学生无需 API Key 即可运行。
- 云端 API 示例（`first_chain.cloud_api_call()`、`human_in_the_loop.py`、`agent_mcp.py`）统一用 `utils/model_utils.get_qwen_client()`。
- MCP 接入见 `agent_mcp.py` + `mcp_config.json`（配置约定见"经验教训"）。

### langgraph_examples/ 专属（LangGraph Agent 教程，10 模块）

| # | 模块 | 内容 | 模型 |
|---|------|------|------|
| 01 | `01_introduction/` | 概念 + 最简图 | 无需 |
| 02 | `02_state_and_branching/` | 状态管理 + 条件分支 | 无需 |
| 03 | `03_agent_loop/` | ReAct Agent 循环 | qwen-plus |
| 04 | `04_workflows/` | 提示链 / 路由 / 并行化 / 评估器-优化器 | qwen-plus |
| 05 | `05_practical/` | 智能问答 Agent 综合实战 | qwen-plus |
| 06 | `06_multi_agent/` | Agent-as-Tool / Handoff / Supervisor | qwen-plus |
| 07 | `07_deep_agents/` | DeepAgents（子代理 / Skills / 文件系统） | qwen-plus + `deepagents` |
| 08 | `08_student_helper/` | 学生助手 HITL（interrupt + Checkpointer） | 无需 |
| 09 | `09_checkpointing/` | Checkpoint / 断点恢复 / 持久化 / 时间旅行 | 无需 |
| 10 | `10_memory/` | 三层记忆 + 置信度机制 + Milvus 进阶 | qwen-plus（各文件示例 1 无需） |

- **需模型的模块默认云端 `qwen-plus`**：`utils/model_utils.get_qwen_client()`，需 `ALIYUN_API_KEY`。
- 测试：`tests/`（pytest），改完跑通全部测试再交付。
- 共享工具：`utils/graph_helpers.py`（`create_tool_node` / `create_router` / `build_react_agent` / `save_graph_png`）。

### rag_examples/ 专属（RAG + Milvus 主课程）

| # | 模块 | 内容 | 默认模型 |
|---|------|------|----------|
| 00 | `00_setup/` | 环境搭建指南 | 无需模型 |
| 01 | `01_milvus_basics/` | Milvus 连接 / Collection / 插入 / 索引 | 模拟向量 |
| 02 | `02_document_chunking/` | 固定切片 / 滑动窗口 / AI 切片 / 概要 / 对比 | 可选 Qwen |
| 03 | `03_retrieval_methods/` | 标量查询 / 向量 / 关键字 / 混合 / Rerank | 模拟向量 |
| 04 | `04_rag_api/` | RAG 检索 API + 问答 API 封装 | Qwen |
| 05 | `05_rag_pipeline/` | RAG 最小版 / 分步详解 / 完整流程 | Qwen |
| 06 | `06_rag_advanced/` | 双集合设计 / Mock→Real | Qwen |
| 07 | `07_rag_evaluation/` | RAGAS 四指标评估 | 无需模型（LLM 判定需 Qwen） |
| — | `embedding_examples/` | Embedding 基础 / 阿里云 / 本地 / 对比 | 阿里云 API |

- 配置入口：`milvus_config.py`（`DEFAULT_DIMENSION = 1024`、`MILVUS_URI`、`MILVUS_DB_NAME`）
- 导入方式：**统一使用** `from rag_examples.milvus_config import MILVUS_URI, DEFAULT_DIMENSION`，**禁止** `sys.path.insert()` hack 导入包内模块
- 语法验证：`run_tests.py`（支持 .py 语法检查）
- 共享工具：`utils/helpers.py`（`ensure_env_loaded` / `get_api_key` / `safe_milvus_operation` / `format_score` / `truncate_text`）
- 新增模块后需同步更新 `run_tests.py` 的 `modules_*` 列表

### rag_demo/ 专属（综合实战项目）

- 配置入口：`config.py`（共享 Milvus 客户端、集合名常量）
- Embedding：**导入** `util/embedding.py`，不要在各模块重复定义 `generate_embedding()`
- 测试：`tests/`（pytest）

### deepeval_examples/ 专属（3 个核心案例）

- 只保留回答评分、RAG 错误定位、安全发布门禁三个代表性案例，不按参考文档的全部章节铺开。
- 用例、指标、阈值及核心调用直接写在案例中：01/02 使用 `measure()`，03 使用 `assert_test()`；不再封装通用评估执行器。
- 01 只讲评分，02 只讲忠实度与相关性，03 只讲隐私断言和评估日志；避免重复业务评分。日志编号使用 `evaluation_id`，不冒充应用链路追踪。
- 运行代码只保留 4 个 `.py`：三个可直接运行的案例和 `judge.py` 模型配置。在线客户端统一通过 `utils.model_utils.get_model("qwen")` 获取，`judge.py` 只保留必要的 DeepEval 接口适配；不增加启动器、公共日志模块或多服务商切换逻辑。
- 案例直接调用真实裁判；用例写在本文件中供阅读，不再维护统一 `--preview` 入口。邮箱脱敏及 JSON 日志留在第三课内。
- 依赖统一写入根目录 `requirements.txt`，配置读取根目录 `.env`、模板维护在根目录 `.env.example`，复用根目录 Python 环境。回归测试在 `tests/test_deepeval_examples.py`，生成日志在 `reports/deepeval/`；教学目录只保留四个 `.py` 和 README。

### langfuse_demo/ 专属（Langfuse 可观测性，3 课 + Agent 进阶选学）

- 主线顺序：`first_trace.py`（自动 Trace + 简短的函数埋点）→ `rag_with_tracing.py`（RAG 检索/生成分层）→ `prompt_experiment.py`（平台 Prompt、引用规则分、Dataset 实验）。
- `qa_agent_with_tracing.py` 为进阶选学，演示 LangGraph CallbackHandler、多轮会话和工具调用追踪。
- 共用连接与环境检查在 `common.py`，各文件不重复定义。
- 需 `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL`；模型调用另需 `ALIYUN_API_KEY`（默认 `qwen-plus`）。依赖统一维护在根目录 `requirements.txt`。

### 独立小示例目录

- `jieba_example/`：三种分词模式 + 自定义词典（`my_words_dict.txt`）+ 词云（`words_demo.py`）
- `minio_example/`：MinIO 对象存储（连接 / 桶 / 上传下载）
- `redis_example/`：Redis 基础（连接 / 序列化 / 缓存），共用客户端见 `redis_client.py`

## 常见反模式（绝对不要做）

| # | ❌ 错误做法 | ✅ 正确做法 | 原因 |
|---|-----------|-----------|------|
| 1 | 在子模块文件中用 `sys.path.insert()` hack 导入其他子模块的函数 | 共享代码放项目根 `utils/`；教学文件统一 `sys.path.insert(0, os.path.join(...))` 仅用于导入 `utils/` | 跨子模块 hack 已全清理；导入 `utils/` 的路径设置是必要且统一的 |
| 2 | `print(f"\n{'─'*50}")\nprint("标题")\nprint(f"{'─'*50}")` | `print(f"\n-- 标题")` | 3 行冗余，刚全局清洗过 100+ 处 |
| 3 | 在多个文件重复定义相同函数 | 提取到 `util/` 共享模块 | 已去重 `generate_embedding()`（原 3 处重复） |
| 4 | 在入门文件（`01_introduction/`）加入 Agent/HITL/Memory 代码 | 入门只讲基础调用，高级概念留在对应模块 | 已从 `first_chain.py` 剥离 Agent 代码 |
| 5 | 硬编码 `47.115.57.130` 等 IP / 数据库名 | `os.getenv("MILVUS_URI")` | 安全风险 + 多学生冲突，已全部替换 |
| 6 | 用玩笑/人身数据做测试用例 | 用与课程主题相关的专业数据 | `qa_paris_additional.json` 已替换为三国 QA |

## 技术栈与约束

| 组件 | 技术 | 约束 |
|------|------|------|
| 本地 LLM | Ollama `qwen3.5:2b` | langchain_examples 默认模型，需本地运行 `ollama serve` |
| 云端 LLM | 阿里云 Qwen `qwen-plus` | rag_examples AI 切片/摘要 + langgraph/langfuse_demo 默认模型；需 `ALIYUN_API_KEY` |
| 云端 LLM | DeepSeek `deepseek-chat` | rag_demo QA 生成；需 `DEEPSEEK_API_KEY` |
| LLM 框架 | LangChain 1.0+ (`create_agent`, LCEL) | langchain_examples 使用；**已弃用** `ConversationBufferMemory`、`SequentialChain` |
| Agent 框架 | LangGraph (`StateGraph`, `Send`, `create_react_agent`) | langgraph_examples 核心 + langchain_examples 09_agent 使用 |
| 可观测性 | Langfuse | langfuse_demo 使用；OpenAI 兼容埋点 + CallbackHandler 追踪 |
| 共享工具 | `utils/model_utils.py` | `get_model()`（Ollama 默认）和 `get_qwen_client()`（云端 Qwen）的统一入口 |
| 向量数据库 | Milvus 2.4+ (pymilvus) | `MILVUS_URI` 配置；Windows 不支持 Lite 版，需 Docker |
| Embedding | 阿里云 `text-embedding-v4` | **固定 1024 维**，通过 OpenAI 兼容接口调用 |
| 文本切片 | LangChain `RecursiveCharacterTextSplitter` | rag_demo 使用 |
| 分词 | jieba | BM25 中文分词 + `jieba_example/` 演示 |
| 对象存储 / 缓存 | MinIO / Redis | `minio_example/`、`redis_example/` 演示用 |
| 本地 Embedding | sentence-transformers, FlagEmbedding | embedding_examples 对比实验用 |

## 环境变量

```
# langchain_examples（云端 API 示例 + 高德 MCP）
ALIYUN_API_KEY=your_dashscope_api_key
AMAP_KEY=your_amap_key

# langfuse_demo
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com

# rag_demo（问答生成用）
DEEPSEEK_API_KEY=your_deepseek_api_key

# rag_examples + rag_demo 共享（Milvus 连接）
MILVUS_URI=http://localhost:19530
MILVUS_DB_NAME=default
```

DeepEval 使用根目录 `.env.example`；其他子项目按各自 README 使用相应模板。

## 经验教训（长期有效的结论）

- **MCP 配置**（`mcp_config.json`）：字段名是 `transport` 不是 `type`；Key 写 `${VAR}` 占位符，代码用 `os.path.expandvars()` 展开。
- **deepagents Skills**：`SKILL.md` 必须放在技能目录的**子目录**里；frontmatter 的 `name` 用小写连字符且与目录同名，触发条件并入 `description`；Windows 下用 `FilesystemBackend(virtual_mode=True)`。
- **演示类示例必须确定性触发**：用预构造历史、`run_limit=1`、`can_jump_to` 等手段，不要赌小模型的指令遵循率。
- **Agent 示例结尾打印工具调用轨迹**，让学员看到内部行为。
- **共享库确定性可测**：时间函数接受 `now=None`，LLM 函数接受 `llm=None` 且带规则降级。
- **Milvus / Store 踩坑 8 条**（`$in` 报错、`$and/$or` 静默返回空、无索引 `score=None`、TTL 不支持等）见 `langgraph_examples/10_memory/README.md`。

完整改造记录见 [CHANGELOG.md](CHANGELOG.md)。
