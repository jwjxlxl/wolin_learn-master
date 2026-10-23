# Rerank 重排序 — 通俗讲解

> 📄 对应代码：`rag_examples/03_retrieval_methods/05_rerank.py`
> 
> 🎯 难度：⭐⭐⭐（3 星）
>
> 📌 核心一句话：**先粗找，再精排**——用快速模型先捞一批候选，再用精准模型重新打分排序。

---

## 一、为什么要 Rerank？

想象你去图书馆找一本书：

1. **第一步（召回）**：你根据书名关键词，快速在书架上扫了一眼，找到了 **50 本可能相关的书**。这一步要**快**，但可能混入不相关的。
2. **第二步（精排）**：你把这 50 本书一本本翻开，仔细阅读目录和摘要，**重新打分排序**，最后选出 **最相关的 5 本**。这一步要**准**，但只处理少量候选，所以慢一点也没关系。

在 RAG 系统中：
- **召回阶段**用 Bi-Encoder（向量相似度），速度快，10ms 级别，但精度有限。
- **精排阶段**用 Cross-Encoder（Rerank 模型），精度高，但计算量大，所以要等召回筛出一批候选后再用。

---

## 二、逐函数讲解

### 1. `rerank_principle()` — 原理讲解（Bi-Encoder vs Cross-Encoder）

**作用**：不依赖任何模型，用大白话讲透 Rerank 的核心原理。

**核心类比**（面试筛选）：
- **召回（Bi-Encoder）** ≈ 人事快速扫简历，先筛出 50 份「可能合适」的。
- **精排（Cross-Encoder）** ≈ 面试官逐份细读，最终定出「最合适」的 5 人。

**根本区别**：
- **Bi-Encoder**：query 和文档**各编各的向量**，再算余弦相似度 → 两边编码时互不见面，只能"隔空比划"，**快但粗**。
- **Cross-Encoder**：把 query 和文档**拼成一句话**一起送进模型 → 模型逐字交互，判断"文档到底答没答到问题"，**准但慢**。

**为什么不能只用一个**：全用 Bi 精度不够，全用 Cross 百万文档逐个过模型算不动 → 折中用两阶段（Bi 捞 Top-50，Cross 精排 Top-5）。

---

### 2. `bge_reranker_demo()` — 中文 Rerank 真实模型

**作用**：用真实的 Rerank 模型给候选文档重新打分，演示中文场景。

**核心代码**：

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('BAAI/bge-reranker-base')
scores = reranker.predict([(query, doc) for doc in candidates])
```

**逐行解释**：

1. **加载模型**：`bge-reranker-base` 是北京智源研究院（BAAI）的中文 Rerank 模型，约 278M，轻量够用。
2. **批量打分**：把所有 `(query, doc)` 对组成**列表**传给 `predict`，一次性算出相关性分数。
   ⚠️ 注意要传 `list[tuple]`，不能传生成器，否则会报错。
3. **排序**：按分数从高到低排列，输出最终排名。

**如果没装依赖 / 下载失败**：代码会捕获异常并打印提示（`pip install sentence-transformers`），不会崩溃。

**安装命令**：`pip install sentence-transformers`

---

### 3. `cross_encoder_rerank()` — 英文 Rerank 真实模型

**作用**：演示英文场景的 Rerank，用微软在 MS MARCO 数据集上训练的模型。

**核心代码**：

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
scores = model.predict([(query, doc) for doc in candidates])
```

**逐行解释**：

1. **加载模型**：`ms-marco-MiniLM-L-12-v2` 适合英文，`predict` 一次性算出所有配对的相关性分数。
2. **排序**：按分数从高到低排列。

**与示例 2 的区别**：
| 对比项 | 示例 2（bge-reranker-base） | 示例 3（ms-marco-MiniLM） |
|--------|-------------|--------------|
| 擅长语言 | 中文 | 英文 |
| 模型来源 | 北京智源（BAAI） | 微软 |
| 加载方式 | `CrossEncoder`（同一个库） | `CrossEncoder`（同一个库） |

> ⚠️ 关键点：**英文模型配英文文本，中文模型配中文文本**，别拿英文模型跑中文。

---

### 4. `milvus_rerank_pipeline()` — 两阶段实战（Milvus 召回 + Rerank 精排）

**作用**：把前面的概念串成一条真实链路——复用 `HybridRetriever`（04 混合检索）召回，再用 Rerank 模型精排，最后对比排名变化。

**流程拆解**：

| 步骤 | 做什么 | 代码对应 |
|------|--------|----------|
| ① 建库 | 复用 `HybridRetriever` 建 Collection + 插入 6 篇文档 | `retriever.create_collection()` / `insert()` |
| ② 召回 | 混合检索先捞 Top-N 候选（宁多勿漏） | `retriever.retrieve(query, top_k=N)` |
| ③ 精排 | 用 bge-reranker 对候选重新打分 | `reranker.predict(...)` |
| ④ 对比 | 打印「召回名次 → 精排名次」的变化 | 带 ↑↓ 箭头的输出 |

**关键观察**：输出中会出现形如 `4↑2`、`2↓4` 的排名变化。召回想把语义相近的都捞回来，精排再从中挑出真正"答到问题"的——例如"数据结构和算法是编程的基础"从召回第 4 提到精排第 2，这正是 Rerank 的价值所在。

**依赖**：Milvus 服务 + `ALIYUN_API_KEY`（Embedding）+ `sentence-transformers`。

---

### 5. `rerank_best_practices()` — Rerank 最佳实践

**作用**：总结 Rerank 的实战经验，告诉学生什么时候用、怎么选模型、怎么调参数。

**要点总结**：

#### 何时使用 Rerank ✅
- 对检索精度要求高的场景
- 候选集较小（<100 条）
- 有足够的计算资源

#### 何时不使用 Rerank ✗
- 实时性要求极高（Rerank 会增加 100-500ms 延迟）
- 候选集太大（Rerank 计算成本高）

#### 模型选择
| 场景 | 推荐模型 |
|------|----------|
| 中文 | `BAAI/bge-reranker-base`（轻量，代码默认）；`-large` 精度更高 |
| 英文 | `cross-encoder/ms-marco-MiniLM-L-12-v2` |
| 多语言 | `BAAI/bge-reranker-v2-m3` |
| API 方案 | Cohere Rerank API |

#### 参数建议
- 召回数量：**50-100 条**（给 Rerank 足够选择）
- 返回数量：**5-10 条**（LLM 上下文限制）
- 分数阈值：过滤掉 <0.3 的低相关结果

#### 性能优化
- 使用 fp16 推理（节省显存）
- 批量处理：一次预测多对 query-doc
- 缓存热点查询的 Rerank 结果

#### 延迟对比
| 阶段 | 模型 | 延迟（单次） |
|------|------|-------------|
| 检索 | Bi-Encoder | ~10ms |
| Rerank | CrossEncoder | ~100-500ms |
| 生成 | LLM | ~1-5s |

> 💡 **结论**：Rerank 增加的延迟通常值得，因为检索质量的提升可以显著改善最终答案质量。

---

## 三、核心概念总结

### Bi-Encoder vs Cross-Encoder

| 对比项 | Bi-Encoder（召回用） | Cross-Encoder（Rerank 用） |
|--------|---------------------|---------------------------|
| 原理 | query 和 doc 分别编码，算余弦相似度 | query 和 doc 拼在一起送入模型 |
| 速度 | 快（可预计算 doc 向量） | 慢（每对都要重新推理） |
| 精度 | 中等 | 高 |
| 适用阶段 | 从百万文档中粗筛 | 对几十条候选精排 |

### 两阶段检索架构

```
用户 query
    │
    ▼
┌─────────────────┐
│  召回（Bi-Encoder） │  ← 快，从海量数据中捞出 Top-50
└────────┬────────┘
         ▼
┌─────────────────┐
│ Rerank（CrossEncoder）│  ← 准，对 50 条精排
└────────┬────────┘
         ▼
    Top-5 结果
         │
         ▼
    交给 LLM 生成答案
```

**为什么要两阶段？** 如果直接用 CrossEncoder 扫描百万文档，太慢；如果只用 Bi-Encoder，精度不够。两阶段结合，**又快又准**。
