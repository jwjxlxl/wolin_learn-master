# RAG 教学示例 — 从零到一的渐进式学习路线

> 为零基础 Python 初学者设计的 RAG（检索增强生成）课程
>
> **学习理念**：每个模块只引入一个新概念，建立在上一模块的基础上，像搭积木一样构建完整的 RAG 知识体系

---

## 🗺️ 学习路线图

```
                    RAG 完整知识体系
    ┌─────────────────────────────────────────────┐
    │                                             │
    │  00_setup                                   │
    │  ↓ 搭建环境，安装 Docker + Python 依赖       │
    │                                             │
    │  embedding_examples                         │
    │  ↓ 理解"文字 → 数字指纹"的概念               │
    │                                             │
    │  01_milvus_basics                           │
    │  ↓ 学会与向量数据库对话（连接→建表→插入→索引）│
    │                                             │
    │  02_document_chunking                       │
    │  ↓ 学会把长文档切成合适的小块                 │
    │                                             │
    │  03_retrieval_methods                       │
    │  ↓ 学会多种检索方式（标量→向量→关键字→混合→重排）│
    │                                             │
    │  04_rag_api                                 │
    │  ↓ 把检索和生成封装成可调用的 API             │
    │                                             │
    │  05_rag_pipeline                            │
    │  ↓ 将所有步骤串成完整的 RAG 流水线            │
    │                                             │
    │  06_rag_advanced                            │
    │  ↓ 学习生产级高级架构（BM25/双集合/Mock→Real） │
    │                                             │
    │  06_rag_evaluation                          │
    │  ↓ 用 RAGAS 四大指标科学评估系统质量          │
    │                                             │
    └─────────────────────────────────────────────┘
```

---

## 📖 课程说明

本教程专为 **Python 基础薄弱的 AI 初学者** 设计，用通俗易懂的语言和生活化比喻，带你从零开始理解并构建 RAG（检索增强生成）系统。

### 学完本课程后，你将能够：

- ✅ 理解 RAG 的核心概念和完整工作流
- ✅ 理解 Embedding 的原理并选择合适的模型
- ✅ 掌握 Milvus 向量数据库的全部基本操作
- ✅ 理解不同文档切片方法的差异和适用场景
- ✅ 掌握多种检索方式并能组合使用
- ✅ 将 RAG 流程封装为可复用的 API
- ✅ 构建完整的 RAG 问答系统
- ✅ 设计生产级高级检索架构
- ✅ 用科学指标评估和优化 RAG 系统

### 前置要求

- 会基础的 Python 语法（变量、函数、循环、字典）
- 对 AI 和大模型有基本了解
- 有好奇心和学习热情

---

## 🚀 快速开始

### 1. 搭建环境

```bash
# 详细步骤见 00_setup/README.md
# 核心三件事：装 Docker → 启 Milvus → 配 .env
```

### 2. 从第一个模块开始

```bash
# 理解 Embedding 概念（纯理论，无需 API Key）
python embedding_examples/01_embedding_basics.py

# 连接 Milvus 数据库
python 01_milvus_basics/01_connect_milvus.py
```

### 3. 按顺序学习

每个模块的 `.py` 文件都是源文件，可以直接运行。也可以在 Jupyter Notebook 中打开对应的 `.ipynb` 文件逐单元格执行。

---

## 📚 渐进式课程目录

### 第 0 阶段：环境搭建（⏱️ 30 分钟）

> **目标**：让所有工具跑起来，不写一行 RAG 代码

| 文件 | 内容 | 难度 | 核心收获 |
|------|------|------|---------|
| [00_setup/README.md](00_setup/README.md) | 环境搭建指南 | ⭐ | Docker + Python + API Key 全套配置 |

**为什么先学这个？**
没有运行环境，后续所有代码都无法执行。本阶段解决"工具准备"问题。

**循序渐进提示**：
1. 先装 Docker（Milvus 的"房子"）
2. 再装 Python 依赖（课程的"工具箱"）
3. 最后配 API Key（调用 AI 模型的"钥匙"）

---

### 第 1 阶段：理解 Embedding（⏱️ 1 小时）

> **目标**：理解"文字如何变成计算机能理解的数字"

| 文件 | 内容 | 难度 | 核心收获 |
|------|------|------|---------|
| [01_embedding_basics.py](embedding_examples/01_embedding_basics.py) | Embedding 基础概念 | ⭐ | 什么是向量、余弦相似度、语义空间 |
| [02_aliyun_embedding.py](embedding_examples/02_aliyun_embedding.py) | 阿里云百炼 Embedding API | ⭐⭐ | 调用真实 API 生成 1024 维向量 |
| [03_local_embedding.py](embedding_examples/03_local_embedding.py) | 本地 Embedding 模型 | ⭐⭐ | 离线部署 BGE/M3E 模型 |
| [04_embedding_comparison.py](embedding_examples/04_embedding_comparison.py) | Embedding 模型对比 | ⭐⭐ | 不同模型的效果和速度差异 |

**学习路径**：
```
概念理解 → 云端 API 调用 → 本地模型部署 → 对比选
   ↑              ↑              ↑            ↑
 纯理论        需要API Key     需要GPU       综合判断
```

**与后续的关系**：Embedding 是 RAG 的基石。不理解 Embedding，就无法理解后面的向量检索。

---

### 第 2 阶段：Milvus 基础（⏱️ 2 小时）

> **目标**：学会与向量数据库"对话"

| 文件 | 内容 | 难度 | 核心收获 |
|------|------|------|---------|
| [01_connect_milvus.py](01_milvus_basics/01_connect_milvus.py) | 连接 Milvus | ⭐ | 多种连接方式、健康检查 |
| [02_create_collection.py](01_milvus_basics/02_create_collection.py) | 创建 Collection | ⭐⭐ | 简单/自定义/多向量/动态字段 |
| [03_insert_data.py](01_milvus_basics/03_insert_data.py) | 插入数据 | ⭐⭐ | 单条/批量/自定义字段/手动ID |
| [04_create_index.py](01_milvus_basics/04_create_index.py) | 创建索引 | ⭐⭐⭐ | FLAT/IVF_FLAT/HNSW 索引对比 |

**学习路径**：
```
连接数据库 → 创建"表" → 插入数据 → 建索引加速查询
    ↑            ↑            ↑            ↑
  第1步        第2步         第3步        第4步
```

**生活化比喻**：
- Collection = Excel 表格
- Field = 表格的列（标题、内容、分类...）
- Vector = 每篇文章的"数字指纹"
- Index = 图书索引，让查找从"逐页翻"变成"按目录找"

**关键概念**：
- `auto_id=True`：Milvus 自动分配 ID（推荐）
- `metric_type="COSINE"`：用余弦相似度衡量向量距离
- `dimension=1024`：与 `text-embedding-v4` 保持一致

---

### 第 3 阶段：文档切片（⏱️ 2 小时）

> **目标**：学会把长文档切成适合检索的小块

| 文件 | 内容 | 难度 | 核心收获 |
|------|------|------|---------|
| [01_fixed_chunking.py](02_document_chunking/01_fixed_chunking.py) | 固定规则切片 | ⭐⭐ | 按字符数/段落切分 |
| [02_sliding_window.py](02_document_chunking/02_sliding_window.py) | 滑动窗口切片 | ⭐⭐ | 重叠窗口保持上下文 |
| [03_ai_chunking.py](02_document_chunking/03_ai_chunking.py) | AI 辅助切片 | ⭐⭐⭐ | 让 LLM 按语义边界切分 |
| [04_summary_chunking.py](02_document_chunking/04_summary_chunking.py) | 概要生成切片 | ⭐⭐⭐ | 为每个 chunk 生成摘要 |
| [05_chunking_comparison.py](02_document_chunking/05_chunking_comparison.py) | 切片方法对比 | ⭐⭐ | 对比不同方法的效果 |

**学习路径**：
```
最简单（固定切） → 更智能（滑动窗） → 最精确（AI切） → 对比选择
    ↑                  ↑                 ↑               ↑
  速度快            保上下文          语义完整        科学决策
```

**为什么切片很重要？**
- 切太大 → 检索时混入太多无关信息，LLM 被干扰
- 切太小 → 语义不完整，检索不到相关内容
- 好的切片 = RAG 系统效果好坏的决定性因素之一

---

### 第 4 阶段：检索方法（⏱️ 3 小时）

> **目标**：掌握从数据库中找到"最相关"文档的多种方法

| 文件 | 内容 | 难度 | 核心收获 |
|------|------|------|---------|
| [01_scalar_query.py](03_retrieval_methods/01_scalar_query.py) | 标量查询 | ⭐⭐ | 条件筛选（按类别、日期过滤） |
| [02_vector_search.py](03_retrieval_methods/02_vector_search.py) | 向量检索 | ⭐⭐⭐ | 语义搜索（理解"意思"而非"字眼"） |
| [03_keyword_search.py](03_retrieval_methods/03_keyword_search.py) | 关键字检索 | ⭐⭐⭐ | BM25 精确匹配关键词 |
| [04_hybrid_search.py](03_retrieval_methods/04_hybrid_search.py) | 混合检索 | ⭐⭐⭐⭐ | 向量 + 关键字双管齐下 |
| [05_rerank.py](03_retrieval_methods/05_rerank.py) | Rerank 重排序 | ⭐⭐⭐⭐ | 对初筛结果二次排序 |

**学习路径**：
```
标量筛选 → 语义检索 → 关键字匹配 → 混合检索 → 重排序优化
   ↑           ↑            ↑            ↑            ↑
 最基础     RAG核心      传统搜索     生产方案     精度提升
```

**每种方法的适用场景**：
- **标量查询**：`category == "AI" AND views > 500`（有明确过滤条件）
- **向量检索**：用户搜"机器学习是什么"（理解语义，不需要精确匹配字眼）
- **关键字检索**：用户搜"Milvus 2.4 版本"（专有名词需要精确匹配）
- **混合检索**：生产环境首选（语义 + 关键字互补）
- **Rerank**：对精度要求极高的场景（额外计算成本换取更高准确率）

---

### 第 5 阶段：API 封装（⏱️ 1.5 小时）

> **目标**：把前面学的检索方法封装成可调用的 API

| 文件 | 内容 | 难度 | 核心收获 |
|------|------|------|---------|
| [rag_retrieval_api.py](04_rag_api/rag_retrieval_api.py) | 检索 API | ⭐⭐⭐⭐ | 封装检索逻辑为可复用接口 |
| [rag_qna_api.py](04_rag_api/rag_qna_api.py) | RAG 问答 API | ⭐⭐⭐⭐ | 在检索基础上增加 LLM 生成 |

**学习路径**：
```
检索 API（只返回相关文档） → 问答 API（返回 AI 生成的答案）
        ↑                              ↑
    更基础                        在前者基础上加 LLM
```

**为什么需要 API 封装？**
前面几个模块的代码是"教学脚本"——每步都打印详细过程方便学习。
API 封装让你把这些能力变成**可在其他项目中调用的模块**。

---

### 第 6 阶段：完整 RAG 流水线（⏱️ 2 小时）

> **目标**：把所有知识点串联成完整的 RAG 闭环

| 文件 | 内容 | 难度 | 核心收获 |
|------|------|------|---------|
| [rag_minimal.py](05_rag_pipeline/rag_minimal.py) | 最小可运行版 | ⭐⭐ | 50 行代码跑通全流程 |
| [rag_full_pipeline.py](05_rag_pipeline/rag_full_pipeline.py) | 完整流程封装 | ⭐⭐⭐ | 生产级 RAG 流水线 |

**完整流程**：
```
【知识库构建】                    【检索问答】

1. 加载文档 ──→ 2. 切片 ──→ 3. Embedding ──→ 4. 存入 Milvus
                                                    │
                                                    │
7. LLM 生成答案 ←── 6. Rerank ←── 5. 向量检索 ←── 用户提问
```

**学习建议**：
1. 先运行 `rag_minimal.py`——最小可运行版，快速建立信心
2. 再研究 `rag_full_pipeline.py`——学习完整的工程化封装
3. 修改参数（chunk_size、top_k 等），观察效果变化

---

### 第 7 阶段：高级技术（⏱️ 3 小时）

> **目标**：学习生产级 RAG 架构，为 rag_demo 实战项目做准备

| 文件 | 内容 | 难度 | 核心收获 |
|------|------|------|---------|
| [01_hybrid_search_advanced.py](06_rag_advanced/01_hybrid_search_advanced.py) | BM25 Function + 混合检索 API | ⭐⭐⭐⭐ | AnnSearchRequest、RRF 融合 |
| [02_dual_collection_design.py](06_rag_advanced/02_dual_collection_design.py) | 双 Collection 设计 | ⭐⭐⭐⭐ | 文档切片 + QA 对双路检索 |
| [03_from_mock_to_real.py](06_rag_advanced/03_from_mock_to_real.py) | 从 Mock 到真实 Embedding | ⭐⭐⭐ | 理解教学代码到生产代码的跨越 |

**学习路径**：
```
BM25 稀疏向量检索 → 双集合架构设计 → 真实向量替换 Mock
      ↑                    ↑                    ↑
  理解高级 API          理解架构设计          理解工程化
```

**与 rag_demo 的关系**：
`rag_demo/` 实战项目使用了本节的全部技术。学完本节后，你应该能完全理解 `rag_demo` 的每一行代码。

---

### 第 8 阶段：系统评估（⏱️ 1.5 小时）

> **目标**：用科学指标评估 RAG 系统质量，告别"凭感觉"

| 文件 | 内容 | 难度 | 核心收获 |
|------|------|------|---------|
| [01_rag_evaluation.py](06_rag_evaluation/01_rag_evaluation.py) | RAGAS 评估框架 | ⭐⭐⭐ | 四大指标自动化评估 |

**为什么需要评估？**
- 换了 Embedding 模型 → 效果变好还是变差？**不知道**
- 调整了切片参数 → 检索质量有没有提升？**不知道**
- 上线了新版本 → 回答质量下降了吗？**不知道**

**RAGAS 四大指标**：

| 指标 | 评估对象 | 含义 |
|------|---------|------|
| Faithfulness（忠实度） | LLM 生成 | 回答是否基于检索内容，没有编造 |
| Answer Relevancy（答案相关性） | LLM 生成 | 回答是否真正解决了问题 |
| Context Precision（上下文精确率） | 检索器 | 检索到的文档中有多少是相关的 |
| Context Recall（上下文召回率） | 检索器 | 相关文档中被检索出来的比例 |

---

## 🧠 知识依赖关系图

```
embedding_examples ──→ 01_milvus_basics ──→ 02_document_chunking
       ↓                      ↓                      ↓
   理解向量              存储向量数据              准备待向量化文本
                                                    ↓
                                          03_retrieval_methods
                                                    ↓
                                              学会多种检索
                                                    ↓
                                          ┌────┬────┴────┐
                                          ↓    ↓         ↓
                                     04_rag_api  05_rag_pipeline  06_rag_advanced
                                          ↓    ↓         ↓
                                     API封装  完整流水线  高级架构
                                                    ↓
                                             06_rag_evaluation
                                                    ↓
                                              科学评估质量
```

---

## 📋 核心理论速查

### RAG 是什么？
> **RAG = 检索（Retrieval）+ 生成（Generation）**
>
> 先检索相关知识库中的文档，再让大语言模型基于这些文档生成答案。
> 解决 AI "幻觉"问题——让回答有据可查。

### Embedding 是什么？
> **Embedding = 文字 → 数字指纹**
>
> 将文本转换为向量（数字列表），语义相近的文本在向量空间中距离也近。

### Milvus 是什么？
> **Milvus = 向量数据库**
>
> 专门存储和搜索向量数据的数据库，支持亿级向量毫秒级检索。

### 索引类型对比
| 索引类型 | 适用场景 | 特点 |
|---------|---------|------|
| FLAT | 小规模测试（< 10 万条） | 精度最高，速度慢 |
| IVF_FLAT | 通用场景（10 万 ~ 千万） | 速度快，精度中 |
| HNSW | 高精度要求（千万+） | 精度高，内存大 |

### 度量类型对比
| 度量类型 | 含义 | 判断标准 |
|---------|------|---------|
| COSINE | 余弦相似度 | 越大越相似（推荐默认） |
| L2 | 欧几里得距离 | 越小越相似 |
| IP | 内积 | 越大越相似 |

---

## 💡 学习建议

### 零基础学员
1. **严格按顺序学习**，不要跳过前面的章节
2. 每个示例都要亲手运行一遍
3. 修改示例中的参数（如 chunk_size、top_k），观察效果变化
4. 遇到错误先阅读错误信息，每个示例都包含友好的错误提示

### 有基础学员
1. 可以直接跳转到感兴趣的部分
2. 重点关注代码组织、错误处理和最佳实践
3. 尝试修改示例代码，添加自己的功能
4. 学完 `05_rag_pipeline` 后，挑战 `rag_demo/` 综合实战项目

### 高效学习技巧
- **先跑通，再理解**：先让代码跑起来看到效果，再回来看每行代码的含义
- **修改参数看变化**：改一个参数运行一次，比读十遍代码更有用
- **用自己的数据测试**：把示例中的测试数据替换成你关心的内容

---

## ❓ 常见问题

### Q: 我需要付费才能学习吗？
A: 不需要！使用 Docker 启动 Milvus 可以免费在本地学习全部功能。阿里云 API 有免费额度。

### Q: 运行示例需要什么配置？
A: 每个文件开头都有"运行前检查"清单，按照提示准备即可。基础模块无需 GPU。

### Q: 代码报错怎么办？
A: 每个示例都包含错误处理代码，会显示友好的错误信息和建议。90% 的问题是 Milvus 没启动或 .env 没配置。

### Q: 学完需要多久？
A: 每天 1-2 小时，约 1-2 周可以学完全部内容。建议按自己的节奏来，不要急于求成。

---

## 📁 项目结构

```
rag_examples/
├── 00_setup/                    # 环境搭建指南
├── embedding_examples/          # Embedding 基础（先学这个！）
├── 01_milvus_basics/            # Milvus 数据库基础
├── 02_document_chunking/        # 文档切片技术
├── 03_retrieval_methods/        # 检索方法大全
├── 04_rag_api/                  # RAG API 封装
├── 05_rag_pipeline/             # 完整 RAG 流水线
├── 06_rag_advanced/             # 高级检索技术
├── 06_rag_evaluation/           # RAG 系统评估
├── utils/                       # 共享工具模块
├── milvus_config.py             # 统一 Milvus 配置入口
├── convert_py_to_ipynb.py       # .py → .ipynb 转换脚本
└── run_tests.py                 # 语法验证工具
```

---

## 🔗 参考资料

- [Milvus 官方文档](https://milvus.io/docs/zh/quickstart.md)
- [Pymilvus API 文档](https://pymilvus.readthedocs.io/)
- [LangChain 文档](https://python.langchain.com/)
- [RAGAS 评估框架](https://github.com/explodinggradients/ragas)

---

**作者**: Luke
**版本**: 2.0
**最后更新**: 2026-06-16
