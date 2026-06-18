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

### 1. `mock_rerank_pipeline()` — 模拟 Rerank 流程

**作用**：不依赖任何真实模型，用假数据演示 Rerank 的完整流程。

**流程拆解**：

| 步骤 | 做什么 | 代码对应 |
|------|--------|----------|
| ① 用户提问 | `"机器学习需要什么基础？"` | `query` 变量 |
| ② 初步检索 | 返回 5 条文档，按向量相似度排序 | `retrieved_docs` 列表 |
| ③ Rerank 重打分 | 给每条文档重新打一个相关性分数 | `rerank_scores` 列表 |
| ④ 重新排序 | 按 Rerank 分数从高到低排列 | `sort()` 排序 |
| ⑤ 对比展示 | 打印 Rerank 前后的排名变化 | 带 ↑↓ 箭头的输出 |

**关键观察**：
- 第 5 句 `"机器学习需要数学基础，包括线性代数和概率。"` 在向量检索中排第 5（相似度 0.72），但 Rerank 后发现它**最直接回答了问题**，分数飙升到 0.92，跃升第 1。
- 第 1 句 `"深度学习是机器学习的子集，使用神经网络。"` 向量检索排第 1（相似度 0.85），但 Rerank 发现它讲的是"子集概念"，**没有直接回答问题**，分数降到 0.45，跌到第 3。

**这就是 Rerank 的价值**：向量相似度只看"语义距离"，而 Rerank 模型能真正理解 query 和 document 之间的**问答相关性**。

---

### 2. `bge_reranker_demo()` — 使用 BGE-Reranker 真实模型

**作用**：演示如何用真实的 BGE-Reranker 模型进行重排序。

**核心代码**：

```python
from FlagEmbedding import FlagReranker

reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=False)
scores = reranker.compute_score([query, doc] for doc in candidates)
```

**逐行解释**：

1. **导入模型**：`FlagReranker` 来自 `FlagEmbedding` 库，由北京智源研究院（BAAI）开发。
2. **加载模型**：`bge-reranker-large` 是模型名，`use_fp16=False` 表示用全精度（有 GPU 可改 True 省显存）。
3. **计算分数**：把 `(query, doc)` 配对传给模型，模型输出一个相关性分数。
4. **排序**：按分数从高到低排列。

**如果没安装模型**：代码会捕获 `ImportError`，自动降级为模拟演示（用预设的假分数），确保学生没装依赖也能看到效果。

**安装命令**：`pip install FlagEmbedding`

---

### 3. `cross_encoder_rerank()` — 使用 sentence-transformers 的 CrossEncoder

**作用**：演示另一种 Rerank 模型——CrossEncoder，常用于英文场景。

**核心代码**：

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
scores = model.predict([[query, doc] for doc in candidates])
```

**逐行解释**：

1. **导入模型**：`CrossEncoder` 来自 `sentence-transformers` 库。
2. **加载模型**：`ms-marco-MiniLM-L-12-v2` 是微软在 MS MARCO 数据集上训练的模型，适合英文。
3. **批量预测**：把所有 `(query, doc)` 对组成列表，一次性传给模型（比逐条调用快）。
4. **排序**：同样按分数从高到低排列。

**与 BGE 的区别**：
| 对比项 | BGE-Reranker | CrossEncoder |
|--------|-------------|--------------|
| 擅长语言 | 中文 | 英文 |
| 安装 | `FlagEmbedding` | `sentence-transformers` |
| 模型来源 | 北京智源 | 微软 / HuggingFace |

---

### 4. `rerank_performance_comparison()` — Rerank 效果对比

**作用**：用具体数据量化展示 Rerank 带来的效果提升。

**流程拆解**：

1. **构造场景**：查询 `"RAG 系统如何工作？"`，6 条候选文档，每条有名称、原始相似度分数、内容。
2. **打印原始排名**：按向量相似度排序的结果。
3. **打印 Rerank 后排名**：按 Rerank 分数重新排序，显示每条文档的排名变化（↑↓箭头）。
4. **计算 NDCG 指标**：用 DCG（折损累计增益）量化排序质量。

**NDCG 是什么？**
- **DCG**（Discounted Cumulative Gain）：排在前面的相关文档得分高，排在后面的得分打折。
- **NDCG**（Normalized DCG）：DCG 除以理想排序的 DCG，取值 0~1，越接近 1 越好。
- 代码中计算 `(reranked_dcg - orig_dcg) / ideal_dcg` 得到 **Rerank 带来的 NDCG 提升百分比**。

---

### 5. `complete_rag_rerank_pipeline()` — 完整 RAG + Rerank 流程

**作用**：用文字梳理完整的 RAG 检索流程，帮助学生建立全局视角。

**6 步流程**：

```
1. 用户提问
   → "机器学习和深度学习有什么区别？"

2. 生成查询向量
   → Embedding 模型把问题编码为向量

3. 向量检索（召回）
   → 从 Milvus 检索 Top-50 候选文档

4. Rerank 重排序（精排）
   → 用 CrossEncoder 对 50 个候选重新打分
   → 选取 Top-5 最终结果

5. 构建上下文
   → 拼接 Top-5 文档内容

6. 调用 LLM 生成答案
   → Prompt: 根据以下信息回答问题...
```

**关键点**：
- 召回阶段取 **Top-50**（给 Rerank 足够的选择空间）。
- 精排后只取 **Top-5**（LLM 上下文窗口有限，给太多会稀释注意力）。

---

### 6. `rerank_best_practices()` — Rerank 最佳实践

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
| 中文 | `BAAI/bge-reranker-large` |
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
