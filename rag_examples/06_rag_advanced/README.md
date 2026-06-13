# 06_rag_advanced — Milvus 高级检索技术

> 难度：⭐⭐⭐⭐（4 星） | 本节是从基础示例到 rag_demo 实战项目的桥梁

## 学习目标

学完本节后，你将能够：

- ✅ 理解并创建 Milvus 的 BM25 Function 实现稀疏向量检索
- ✅ 掌握 `AnnSearchRequest` 和 `hybrid_search()` API
- ✅ 理解 RRF（倒数排名融合）和加权排序的差异与选择
- ✅ 设计双 Collection 的 RAG 检索架构
- ✅ 理解从 mock 示例到生产代码的跨越

## 前置知识

- 完成 `01_milvus_basics/` — 了解 Milvus 基本操作
- 完成 `03_retrieval_methods/` — 了解基本检索方法
- 完成 `05_rag_pipeline/` — 了解 RAG 完整流程

## 文件说明

| 文件 | 内容 |
|------|------|
| `01_hybrid_search_advanced.py` | BM25 Function、混合检索 API、RRF vs 加权排序 |
| `02_dual_collection_design.py` | 双集合设计模式（文档切片 + QA 对） |
| `03_from_mock_to_real.py` | 从随机向量到真实 Embedding 的过渡指南 |

## 与 rag_demo 的关系

`rag_demo/` 是本课程的综合实战项目，它使用了本节讲授的所有高级技术：
- `rag_demo/db/vdb_init_milvus.py` — 使用 BM25 Function 创建索引
- `rag_demo/core/rag_query.py` — 使用 hybrid_search + RRF + 双集合检索

学完本节后，你应该能够完全理解 `rag_demo` 的每一行代码。
