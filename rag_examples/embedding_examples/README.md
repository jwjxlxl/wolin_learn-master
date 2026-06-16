# embedding_examples — 文本向量化（Embedding）

> 难度：⭐ | 前置：无 | 后续：01_milvus_basics 向量数据库

## 🎯 本节目标

学完本模块后，你将能够：

- ✅ 理解 Embedding 的核心概念（文本如何变成向量、语义空间、余弦相似度）
- ✅ 调用阿里云百炼 API 生成真实的 1024 维文本向量
- ✅ 使用 sentence-transformers 部署本地 Embedding 模型
- ✅ 对比不同 Embedding 模型的效果、速度和成本，做出科学选型

## 🗺️ 学习路径

```
概念理解 → 云端 API 调用 → 本地模型部署 → 对比选型
   ↑            ↑              ↑            ↑
 纯理论      需要API Key     需下载模型    科学决策
  (⭐)        (⭐⭐)           (⭐⭐)         (⭐⭐)

文件：01_embedding_basics.py → 02_aliyun_embedding.py → 03_local_embedding.py → 04_embedding_comparison.py
```

## 📁 文件说明

| # | 文件 | 内容 | 难度 | 核心收获 |
|---|------|------|------|---------|
| 1 | [01_embedding_basics.py](01_embedding_basics.py) | Embedding 基础概念 | ⭐ | 什么是向量、余弦相似度、语义空间 |
| 2 | [02_aliyun_embedding.py](02_aliyun_embedding.py) | 阿里云百炼 API | ⭐⭐ | 调用真实 API 生成向量、工具类封装 |
| 3 | [03_local_embedding.py](03_local_embedding.py) | 本地 Embedding 模型 | ⭐⭐ | sentence-transformers、语义检索演示 |
| 4 | [04_embedding_comparison.py](04_embedding_comparison.py) | 模型对比与选型 | ⭐⭐ | 效果/速度/成本对比、决策树 |

---

## 📖 学习指导

### Embedding 是什么？为什么 RAG 离不开它？

```
文字 → Embedding → 向量 → 计算机可以：
                        - 计算相似度
                        - 聚类分析
                        - 机器学习训练

生活化比喻：
  Embedding = "给文字分配 GPS 坐标"

  传统文本："苹果" 就是一串字符
  Embedding 后："苹果" → [0.8, -0.2, 0.5, ...] → 向量空间中的一个点

  语义相近的词，在向量空间中的距离也相近：

              ↑
          "橙子" ●
     "苹果" ●       ● "香蕉"
                  ● "水果"
              │
  ────────────┼───────────→
              │
```

Embedding 是 RAG 检索的基础——没有向量，就没有向量检索。

---

### 第 1 步：理解 Embedding 概念（01_embedding_basics.py）

**本节讲什么：**
- 用 4 维语义空间模拟 Embedding 过程（科技 / 水果 / 食物 / 抽象）
- 余弦相似度计算：公式、手动实现、相似度矩阵
- Embedding 的 6 大应用场景（语义搜索 / 推荐 / 聚类 / 问答 / 去重 / RAG）
- 2D 向量可视化（ASCII 艺术）
- 常见 Embedding 模型对比表

**核心概念：**

```
模拟 Embedding 过程：
  用简化的"词袋"方法给每个词分配一个 4 维向量：
  - "苹果" → 水果维度=0.7，科技维度=0.3（也可指 Apple 公司）
  - "手机" → 科技维度=0.8
  - "AI"   → 科技维度=0.9

余弦相似度公式：
              A · B
  cos(θ) = ───────────
           ‖A‖ × ‖B‖

  - 范围：[-1, 1]
  - 1: 方向完全相同（语义完全一致）
  - 0: 正交（语义无关）
  - -1: 方向完全相反（语义对立）
```

**6 大应用场景：**

| 场景 | 原理 | 示例 |
|------|------|------|
| 语义搜索 | 查询向量化 → 匹配相似文档 | 搜"苹果手机" → 找到"iPhone" |
| 推荐系统 | 用户喜欢的内容 → 找相似新内容 | 看了"ML 教程" → 推荐"DL 入门" |
| 文本聚类 | 相似向量自动分组 | 新闻分类、用户反馈分组 |
| 问答匹配 | 用户问题 → 匹配 FAQ 中最相似的问题 | "怎么退款" → 匹配"退货流程" |
| 文本去重 | 相似度 > 0.95 判定为重复 | 去除知识库中的重复文档 |
| RAG 检索 | **问题向量化 → 向量检索 → LLM 生成** | 整个 RAG 系统的基础 |

**本文件特点：**
- ⭐ **纯理论，无需 API Key、无需 GPU**，任何环境都能跑
- 所有向量都是模拟的（用规则赋值），但概念演示完整

---

### 第 2 步：阿里云百炼 API 实战（02_aliyun_embedding.py）

**本节讲什么：**
- 调用阿里云百炼 `text-embedding-v3` 模型（1024/1536 维可选）
- 单个调用 vs 批量调用
- 封装 `AliyunEmbedding` 工具类（`embed()` / `embed_batch()` / `similarity()`）
- 错误处理：重试策略（指数退避）、常见错误码
- API 调用成本控制与最佳实践

**核心概念：**

```
text-embedding-v3 特点：
  - 支持中英文
  - 支持 1024/1536 两种维度
  - 最大输入 8192 tokens
  - 输出已归一化（可直接用点积计算相似度）

3 种调用方式：
  1. DashScope SDK（推荐）
  2. OpenAI 兼容接口（本课程主要用这个）
  3. HTTP REST API
```

**工具类封装（`AliyunEmbedding`）：**

```python
emb = AliyunEmbedding(model='text-embedding-v3')

# 1. 生成单个向量
vector = emb.embed("人工智能简介")

# 2. 批量生成（内部自动分批，避免 API 限制）
vectors = emb.embed_batch(["文本 1", "文本 2", "文本 3"])

# 3. 计算相似度
sim = emb.similarity("机器学习", "深度学习")  # 返回 0.85
```

**相似度阈值参考：**

| 场景 | 推荐阈值 | 说明 |
|------|---------|------|
| 严格去重 | >0.95 | 几乎相同的内容 |
| 语义匹配 | >0.75 | 语义高度相关 |
| 主题聚合 | >0.60 | 同一主题内容 |
| 宽泛相关 | >0.40 | 有一定关联 |

**错误处理最佳实践：**

```python
# 指数退避重试策略
for attempt in range(max_retries):
    try:
        result = TextEmbedding.call(model='text-embedding-v3', input=text)
        if result.status_code == 200:
            return result.output['embeddings'][0]['embedding']
        elif result.status_code == 429:  # 频率超限
            time.sleep(2 ** attempt)  # 1s, 2s, 4s...
        else:
            raise Exception(f"API 错误：{result.code}")
    except Exception:
        if attempt == max_retries - 1:
            raise
        time.sleep(2 ** attempt)
```

---

### 第 3 步：本地模型部署（03_local_embedding.py）

**本节讲什么：**
- 使用 `sentence-transformers` 调用本地 Embedding 模型
- 中文推荐模型：`bge-large-zh-v1.5` / `bge-base-zh-v1.5` / `m3e-base`
- 批量生成、相似度计算、语义检索演示（预计算向量 + Top-K 搜索）
- GPU 加速、Half 精度、长文本分段处理
- 本地 vs API 对比

**核心概念：**

```
本地模型 vs API 对比：

| 维度 | 本地模型 | API 调用 |
|------|----------|----------|
| 成本 | 一次下载，免费 | 按调用收费 |
| 速度 | 本地推理，无网络延迟 | 网络传输延迟 |
| 隐私 | 数据不出本地 | 数据发送到云端 |
| 维护 | 需要自己管理模型 | 无需管理 |
| 效果 | 取决于选择的模型 | 通常是最新最强 |

何时选择本地模型？
  ✓ 数据敏感，不能上传到云端
  ✓ 调用量大，API 成本过高
  ✓ 需要离线运行
  ✓ 低延迟要求
```

**语义检索演示——一个"微型 RAG"：**

```python
# 1. 预计算所有文档的向量（只需做一次）
doc_embeddings = model.encode(documents, convert_to_tensor=True)

# 2. 用户提问
query = "机器学习需要什么？"

# 3. 生成查询向量
query_embedding = model.encode(query, convert_to_tensor=True)

# 4. 计算相似度，取 Top-3
cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
top_results = torch.topk(cos_scores, k=3)

# 5. 返回最相关的文档
```

这个流程就是 RAG 检索的缩影——后面 `05_rag_pipeline` 会把它完整封装。

**常用中文模型速查：**

| 模型 | 维度 | 大小 | 特点 | 适用 |
|------|------|------|------|------|
| bge-large-zh-v1.5 | 1024 | ~1.2GB | 效果最好 | **高精度 RAG** |
| bge-base-zh-v1.5 | 768 | ~500MB | 平衡性能 | 通用场景 |
| bge-small-zh-v1.5 | 512 | ~100MB | 轻量快速 | 资源受限 |
| m3e-base | 768 | ~500MB | MokaAI 出品 | 通用场景 |

**性能优化技巧：**

```python
# GPU 加速（如有）
model = SentenceTransformer('bge-large-zh-v1.5', device='cuda')

# Half 精度（节省一半内存）
model.half()

# 长文本分段取平均
def encode_long_text(model, text, max_length=512):
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    embeddings = model.encode(chunks)
    return embeddings.mean(axis=0)
```

---

### 第 4 步：模型对比与选型（04_embedding_comparison.py）

**本节讲什么：**
- 7+ 主流模型的效果对比（MTEB 中文榜单）
- 速度-效果权衡曲线
- API vs 本地部署成本分析（按 100 万次调用计算）
- **模型选择决策树**（4 步问答）
- 在自己的数据上实测的方法（Recall@K）

**核心概念：**

**模型效果对比（中文场景）：**

| 模型 | 平均分数 | 检索能力 | 维度 | 大小 | 速度 |
|------|---------|---------|------|------|------|
| bge-large-zh-v1.5 | 65.2 | 68.5 | 1024 | 1.2GB | ~50 句/秒 |
| bge-base-zh-v1.5 | 63.8 | 66.2 | 768 | 500MB | ~80 句/秒 |
| bge-small-zh-v1.5 | 59.1 | 61.8 | 512 | 100MB | ~150 句/秒 |
| m3e-base | 63.6 | 65.8 | 768 | 500MB | ~75 句/秒 |
| text-embedding-v3 | 62.2 | 64.5 | 1024 | API | API 调用 |

> 分数越高越好，检索能力是 RAG 最关注的指标

**成本对比（按 100 万次调用/年计算）：**

| 方案 | 成本 | 优点 | 缺点 |
|------|------|------|------|
| 阿里云 API | ~¥10,000/年 | 无需管理、总是最新 | 长期成本高、数据上云 |
| 本地部署 | ~¥8,000/年 | 隐私安全、无延迟 | 需维护、有前期投入 |

> 约 **50-80 万次调用/年** 是成本平衡点，超过这个量本地部署更经济

**模型选择决策树：**

```
Q1: 数据能否上传到云端？
├─ 不能 → 选择本地模型 → Q3
└─ 能 → Q2

Q2: 年调用量是否超过 50 万次？
├─ 是 → 本地部署更经济 → Q3
└─ 否 → API 更划算 → 阿里云 text-embedding-v3

Q3: 是否有 GPU 或充足内存（>2GB）？
├─ 是 → bge-large-zh-v1.5（效果最好）
└─ 否 → Q4

Q4: 是否需要实时响应（<100ms）？
├─ 是 → bge-small-zh-v1.5（速度最快）
└─ 否 → bge-base-zh-v1.5（平衡方案）
```

**快速选择指南：**

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| 企业 RAG 系统 | bge-large-zh | 效果优先 |
| 初创公司 MVP | API | 快速上线 |
| 移动端应用 | bge-small-zh | 轻量 |
| 隐私敏感数据 | 本地模型 | 数据安全 |
| 预算有限 | m3e-base | 免费 |

---

## 🧠 核心知识点总结

### Embedding 在 RAG 中的位置

```
知识库构建：
  加载文档 → 切片 → 【Embedding】→ 存入 Milvus

检索问答：
  用户提问 → 【Embedding】→ 向量检索 → Top-K 文档 → LLM 生成答案
```

两个阶段都需要 Embedding，且**必须使用同一个模型**，否则向量不在同一语义空间，相似度计算无意义。

### 关键原则

| 原则 | 说明 | 后果 |
|------|------|------|
| **模型必须一致** | 文档向量化和查询向量化用同一个模型 | 不同模型 → 不同语义空间 → 检索无效 |
| **维度必须匹配** | Collection 的 dimension = 模型输出维度 | 1024 维向量插入非 1024 维 Collection 会失败 |
| **归一化很重要** | 使用 COSINE 时向量应归一化 | 未归一化向量距离受长度影响 |
| **批量优于单次** | 批量调用减少网络开销 | 1000 条批量调用 ≈ 100 次单次调用的时间 |

---

## 🏃 快速开始

```bash
# 从概念理解开始（无需任何依赖）
python embedding_examples/01_embedding_basics.py

# 需要 API Key
python embedding_examples/02_aliyun_embedding.py

# 需要安装 sentence-transformers（首次运行会下载模型）
python embedding_examples/03_local_embedding.py

# 模型对比（模拟数据，无需额外依赖）
python embedding_examples/04_embedding_comparison.py
```

## ⚠️ 常见问题

### Q: 没有 API Key 能学这个模块吗？
A: 可以！`01_embedding_basics.py` 是纯理论演示，无需 API Key。`03_local_embedding.py` 用本地模型，下载后免费使用。

### Q: 选 API 还是本地模型？
A: 先用 API 跑通流程（入门阶段），调用量大了（>50 万次/年）再切换到本地模型。也可以直接用 `03_local_embedding.py` 的本地模型起步。

### Q: 向量维度不一致报错怎么办？
A: 确保 Milvus Collection 的 `dimension` 与 Embedding 模型输出维度一致。本课程统一使用 `text-embedding-v4`，维度为 1024。

### Q: 中文文本很长，超过模型最大长度怎么办？
A: 分段处理：将长文本切成 <= 512 tokens 的块，分别生成向量后取平均值。详见 `03_local_embedding.py` 中的 `encode_long_text()` 示例。

### Q: 怎么知道选哪个模型最好？
A: 用你的实际数据测试 2-3 个候选模型：取 50-100 条典型查询，人工标注相关文档，计算 Recall@K。详见 `04_embedding_comparison.py` 中的实测建议。

---

**作者**: Luke
**版本**: 1.0
**最后更新**: 2026-06-16
