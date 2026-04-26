# RAG 教学示例 - Milvus 向量库实战

> 为零基础 Python 初学者设计的 RAG 渐进式教程

## 课程说明

本教程专为 **Python 基础薄弱的 AI 初学者** 设计，用通俗易懂的语言和生活化比喻，带你从零开始学习 RAG（检索增强生成）和 Milvus 向量库。

### 学完本课程后，你将能够：

- ✅ 理解 RAG 的核心概念和工作流程
- ✅ 掌握 Milvus 向量库的基本操作
- ✅ 理解不同文档切片方法的差异
- ✅ 掌握多种检索方式（标量、语义、关键字、混合）
- ✅ 理解索引类型、度量类型、Rerank 算法的原理
- ✅ 独立构建 RAG 应用

### 前置要求

- 会基础的 Python 语法（变量、函数、循环）
- 对 AI 和大模型有基本了解
- 有好奇心和学习热情

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 Milvus 服务

**方式 1：使用 Milvus Lite（推荐，适合学习）**
```bash
# Milvus Lite 会自动启动，无需额外配置
# 首次运行时会自动下载
```

**方式 2：使用 Docker 启动完整 Milvus**
```bash
docker-compose up -d
```

### 3. 运行示例

```bash
# 从基础开始
python 01_milvus_basics/01_connect_milvus.py
```

---

## 课程目录

### 第一部分：Milvus 基础

| 文件 | 内容 | 难度 |
|------|------|------|
| [01_connect_milvus.py](01_milvus_basics/01_connect_milvus.py) | 连接 Milvus | ⭐ |
| [02_create_collection.py](01_milvus_basics/02_create_collection.py) | 创建集合 | ⭐⭐ |
| [03_insert_data.py](01_milvus_basics/03_insert_data.py) | 插入数据 | ⭐⭐ |
| [04_create_index.py](01_milvus_basics/04_create_index.py) | 创建索引 | ⭐⭐⭐ |

### 第二部分：文档切片

| 文件 | 内容 | 难度 |
|------|------|------|
| [01_fixed_chunking.py](02_document_chunking/01_fixed_chunking.py) | 固定规则切片 | ⭐⭐ |
| [02_sliding_window.py](02_document_chunking/02_sliding_window.py) | 滑动窗口切片 | ⭐⭐ |
| [03_ai_chunking.py](02_document_chunking/03_ai_chunking.py) | AI 辅助切片 | ⭐⭐⭐ |
| [04_summary_chunking.py](02_document_chunking/04_summary_chunking.py) | 概要生成切片 | ⭐⭐⭐ |
| [05_chunking_compare.py](02_document_chunking/05_chunking_compare.py) | 切片方法对比 | ⭐⭐⭐ |

### 第三部分：检索方法

| 文件 | 内容 | 难度 |
|------|------|------|
| [01_scalar_query.py](03_retrieval_methods/01_scalar_query.py) | 标量查询 | ⭐⭐ |
| [02_vector_search.py](03_retrieval_methods/02_vector_search.py) | 语义检索 | ⭐⭐⭐ |
| [03_keyword_search.py](03_retrieval_methods/03_keyword_search.py) | 关键字检索 | ⭐⭐⭐ |
| [04_hybrid_search.py](03_retrieval_methods/04_hybrid_search.py) | 混合检索 | ⭐⭐⭐⭐ |
| [05_rerank_search.py](03_retrieval_methods/05_rerank_search.py) | Rerank 重排序 | ⭐⭐⭐⭐ |

### 第四部分：API 封装

| 文件 | 内容 | 难度 |
|------|------|------|
| [01_retrieve_api.py](04_rag_api/01_retrieve_api.py) | 检索 API | ⭐⭐⭐⭐ |
| [02_rag_qa_api.py](04_rag_api/02_rag_qa_api.py) | RAG 问答 API | ⭐⭐⭐⭐ |

---

## 核心理论速查

### RAG 是什么？
> **RAG = 检索 + 生成**
>
> 先检索相关知识，再让 AI 回答，解决 AI 幻觉问题

### Milvus 是什么？
> **Milvus = 向量数据库**
>
> 专门存储和搜索向量数据的数据库

### Embedding 是什么？
> **Embedding = 文字转数字指纹**
>
> 将文本转换为向量，语义相似的文本向量也接近

### 索引类型对比
| 索引类型 | 适用场景 | 特点 |
|---------|---------|------|
| FLAT | 小规模测试 | 精度最高，速度慢 |
| IVF_FLAT | 通用场景 | 速度快，精度中 |
| HNSW | 高精度要求 | 精度高，内存大 |

### 度量类型对比
| 度量类型 | 含义 | 判断标准 |
|---------|------|---------|
| L2 | 欧几里得距离 | 越小越相似 |
| COSINE/IP | 余弦相似度 | 越大越相似 |

---

## 学习建议

### 零基础学员
1. 按顺序学习，不要跳过前面的章节
2. 每个示例都要亲手运行一遍
3. 遇到错误先阅读错误信息

### 有基础学员
1. 可以直接跳转到感兴趣的部分
2. 重点关注代码组织和最佳实践
3. 尝试修改示例代码，添加自己的功能

---

## 常见问题

### Q: 我需要付费才能学习吗？
A: 不需要！使用 Milvus Lite 可以免费在本地学习全部功能。

### Q: 运行示例需要什么配置？
A: 每个文件开头都有"运行前检查"清单，按照提示准备即可。

### Q: 代码报错怎么办？
A: 每个示例都包含错误处理代码，会显示友好的错误信息和建议。

---

## 参考资料

- [Milvus 官方文档](https://milvus.io/docs/zh/quickstart.md)
- [Pymilvus API 文档](https://pymilvus.readthedocs.io/)
- [LangChain 文档](https://python.langchain.com/)

---

**作者**: Ric
**版本**: 1.0
**最后更新**: 2026-03-28
