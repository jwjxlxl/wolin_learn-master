# 📝 作业：检索方法实战（03_retrieval_methods）

> **难度**：⭐⭐⭐ | **建议用时**：3-4 小时

---

## 第一部分：理论题（40 分）

### 一、简答题（每题 5 分，共 20 分）

**1. 为什么说"检索是 RAG 系统中最关键的一步"？请用"垃圾进，垃圾出"的原理结合具体场景说明。**

**2. 对比以下 3 种相似度度量类型，说明各自计算方式和适用场景：**

| 类型 | 计算方式 | 值越大/越小越相似？ | 何时使用 |
|------|---------|-------------------|---------|
| COSINE | | | |
| L2 | | | |
| IP | | | |

**3. 解释 BM25 算法的三个核心要素（TF、IDF、文档长度归一化），并说明为什么"的"、"是"等常见词的 IDF 分数低。**

**4. 画出两阶段检索架构图（召回 → 精排），标注每阶段使用的模型类型（Bi-Encoder / CrossEncoder），并解释为什么不能对百万文档直接用 CrossEncoder。**

### 二、分析与计算题（每题 10 分，共 20 分）

**5. RRF（倒数排名融合）计算：**

某次查询有 4 篇候选文档，两种检索方法的排名如下：

| 文档 | 向量检索排名 | 关键字检索排名 |
|------|------------|--------------|
| A | 1 | 3 |
| B | 2 | 2 |
| C | 4 | 1 |
| D | 3 | 4 |

使用 RRF 公式 `score = 1/(k + rank_vector) + 1/(k + rank_keyword)`，其中 k=60：
- (a) 计算每篇文档的 RRF 分数（保留 6 位小数）
- (b) 给出最终排名顺序
- (c) 解释为什么 RRF 比加权平均更稳定

**6. 场景分析题：**

某学生做了一个课程问答系统，用户提问"线性代数在机器学习中有什么用"。系统向量检索返回的 Top-5 结果如下：

| 排名 | 相似度 | 文档内容 |
|------|--------|---------|
| 1 | 0.88 | "机器学习是人工智能的核心技术" |
| 2 | 0.85 | "Python 编程语言入门教程" |
| 3 | 0.82 | "深度学习中的反向传播算法" |
| 4 | 0.79 | "线性代数是数学的基础学科，包括矩阵运算和特征值" |
| 5 | 0.76 | "机器学习需要数学基础，线性代数和概率论是关键" |

- (a) 从用户问题的角度，哪个文档才是**真正相关**的？为什么它排在了最后？
- (b) 如果加入 Rerank 阶段，排名会发生什么变化？说明原因。
- (c) 如果你只能选一种检索方法（向量/关键字/混合/Rerank），你选哪个？说明理由。

---

## 第二部分：实操题（60 分）

> **环境要求**：确保 Milvus 服务已启动（`docker ps` 能看到 milvus 容器），`pip install pymilvus python-dotenv jieba rank-bm25` 已安装。
>
> **所有代码文件请放在** `rag_examples/03_retrieval_methods/homework/` **目录下。**

### 三、基础实操：标量查询 + 向量检索综合（15 分）

**任务：创建一个"智能文档筛选器"**

编写 `homework/task1_scalar_vector.py`，完成以下功能：

1. 创建一个 Collection，包含以下字段：
   - `id`（主键，自增）
   - `title`（VARCHAR）
   - `content`（VARCHAR）
   - `category`（VARCHAR，取值：`AI`、`Product`、`Programming`、`Data`）
   - `difficulty`（INT64，取值 1-5，表示难度等级）
   - `is_free`（BOOL）
   - `embedding`（FLOAT_VECTOR，1024 维）

2. 插入至少 **12 条**模拟数据（覆盖 4 个类别、不同难度、免费/付费）

3. 实现以下查询并打印结果：
   - 查询所有 `AI` 类别中难度 ≥ 3 的文档
   - 查询所有免费（`is_free == True`）的文档
   - 用随机向量做向量检索，同时过滤 `category == 'Programming'`，返回 Top-3
   - 组合查询：`category in ['AI', 'Data'] and difficulty <= 3 and is_free == True`

**评分标准**：
- Collection 创建和 Schema 定义正确（4 分）
- 数据插入成功且覆盖所有类别（3 分）
- 4 个查询全部正确执行并输出结果（8 分）

### 四、进阶实操：实现 BM25 检索器（20 分）

**任务：从零实现一个中文 BM25 检索器**

编写 `homework/task2_bm25_searcher.py`，完成以下功能：

1. 准备文档库：从以下 5 篇文章标题+摘要构建文档库（自己编写内容，每条 50-100 字）：
   - "机器学习基础" — 介绍监督学习、无监督学习
   - "深度学习入门" — 介绍神经网络、CNN、RNN
   - "Python 数据分析" — 介绍 Pandas、NumPy
   - "自然语言处理" — 介绍分词、情感分析、机器翻译
   - "推荐系统原理" — 介绍协同过滤、内容推荐

2. 实现 `ChineseBM25` 类（继承/参考 `SimpleBM25` 的思路），要求：
   - 使用 `jieba` 进行中文分词（不要用简单字符分割）
   - 实现 `_idf()`、`_tf()`、`search()` 方法
   - 支持自定义 k1 和 b 参数

3. 执行以下检索并打印结果：
   - 查询"神经网络"，返回 Top-3
   - 查询"数据处理"，返回 Top-3
   - 查询"AI 翻译"，返回 Top-3

4. 对比实验：分别用 k1=0.5、k1=1.5、k1=2.0 查询"学习"，观察排名变化，打印对比表格。

**评分标准**：
- ChineseBM25 类实现正确，jieba 分词集成（8 分）
- TF/IDF 计算逻辑正确（4 分）
- 3 个查询结果合理（4 分）
- k1 参数对比实验及分析（4 分）

### 五、综合实操：混合检索 + Rerank 两阶段系统（25 分）

**任务：搭建一个完整的"两阶段检索问答系统"**

编写 `homework/task3_two_stage_retrieval.py`，完成以下功能：

1. **构建知识库**（10 条文档，覆盖 AI/ML/DL/NLP/CV 等主题）

2. **第一阶段 — 召回（Recall）**：
   - 用模拟向量做向量检索，返回 Top-8 候选
   - 用 BM25（可直接用 `rank-bm25` + `jieba`）做关键字检索，返回 Top-8 候选
   - 用 RRF 融合两种结果，得到 Top-8 融合排名

3. **第二阶段 — 精排（Rerank）**：
   - 对 RRF 的 Top-8 结果，模拟 CrossEncoder 打分（自己设计一个相关性评分逻辑，比如：计算 query 和文档的词重叠度 + 语义关键词匹配度）
   - 按 Rerank 分数排序，返回 Top-3 最终结果

4. **效果验证**：
   - 用 3 个不同的查询测试系统：
     - 查询 1（语义型）："什么是深度学习"
     - 查询 2（精确型）："Transformer 架构"
     - 查询 3（综合型）："机器学习需要哪些数学基础"
   - 对每个查询，打印完整流程：召回结果 → RRF 融合 → Rerank 精排 → 最终 Top-3

5. **封装**：将上述流程封装成一个 `TwoStageRetriever` 类，提供 `search(query, recall_k=8, final_k=3)` 接口。

**评分标准**：
- 召回阶段：向量检索 + BM25 各自正确实现（6 分）
- RRF 融合逻辑正确（5 分）
- 精排阶段：Rerank 打分逻辑合理（5 分）
- 3 个查询的完整流程输出（5 分）
- TwoStageRetriever 类封装规范（4 分）

---

## 附加题（+10 分，可选）

**任务：检索方法对比实验报告**

编写 `homework/bonus_comparison_report.md`，回答以下问题：

1. 对同一个文档库和同一组查询（至少 5 个），分别用以下方式检索并记录 Top-3 结果：
   - 纯向量检索
   - 纯 BM25 关键字检索
   - 混合检索（RRF 融合）
   - 混合 + Rerank

2. 人工标注每个查询的"正确答案"（哪些文档是真正相关的），计算每种方法的 **Precision@3**（Top-3 中相关文档的比例）

3. 写一份 300 字以上的分析报告，包括：
   - 各方法 Precision@3 对比表格
   - 哪种方法表现最好？为什么？
   - 在什么场景下关键字检索会优于向量检索？
   - Rerank 带来的提升是否值得额外的计算成本？

---

## 提交要求

```
homework/
├── task1_scalar_vector.py          # 任务一
├── task2_bm25_searcher.py          # 任务二
├── task3_two_stage_retrieval.py    # 任务三
└── bonus_comparison_report.md      # 附加题（可选）
```

运行验证：
```bash
cd rag_examples/03_retrieval_methods
python homework/task1_scalar_vector.py
python homework/task2_bm25_searcher.py
python homework/task3_two_stage_retrieval.py
```

所有脚本需满足：
- 使用 `from rag_examples.milvus_config import MILVUS_URI, DEFAULT_DIMENSION` 导入配置
- 不硬编码 IP、密码
- 中文注释和打印输出
- 每个示例用 `print(f"\n-- 示例 N: 标题")` 分隔
