# Neo4j 示例文件说明

## 文件结构

```
neo4j_examples/
├── neo4j_cypher_complete.cypher   # Neo4j Cypher 语法大全
├── neo4j_python_guide.py          # Python 操作 Neo4j 指南
├── data/
│   └── movies.csv                 # 批量导入示例数据
├── requirements.txt               # Python 依赖
├── .env.example                   # 环境变量配置示例
└── README.md                      # 本说明文件
```

## Neo4j 核心概念白话版

Neo4j 是图数据库，最适合表达“对象之间的关系”。如果用电影知识图谱来理解：

| 概念 | 白话解释 | 电影图中的例子 |
|------|----------|----------------|
| 节点 Node | 一个具体对象 | 一个演员、一个电影、一个导演、一个公司 |
| 标签 Label | 节点的类型 | `Person`、`Movie`、`Director`、`Company` |
| 属性 Property | 对象上的字段 | 电影的 `title`、`rating`，演员的 `name`、`age` |
| 关系 Relationship | 两个对象之间的连接 | 演员 `ACTED_IN` 电影，导演 `DIRECTED` 电影 |
| 路径 Path | 多个节点和关系连成的链路 | 演员 → 电影 → 导演 → 其他电影 |

可以把它理解成：

```text
(Alice:Person)-[:ACTED_IN {roles: ['Neo']}]->(The Matrix:Movie)<-[:DIRECTED]-(Lana:Director)
```

这句话表达的是：Alice 出演了《The Matrix》，Lana 执导了《The Matrix》。图数据库的优势就是可以沿着这些关系继续追问：“Alice 和谁合作过？”、“某个导演还拍过哪些电影？”、“两个人之间最短通过几层关系能连上？”

## 从业务问题设计图模型

学习 Neo4j 不要只背 Cypher，核心是把业务问题转成“节点 + 关系 + 属性”。

### 建模步骤

1. 先写业务问题：用户到底要查什么？
2. 找实体：哪些东西适合作为节点？
3. 找关系：实体之间是什么动作或连接？
4. 补属性：哪些字段用于过滤、排序、展示？
5. 写查询：用 Cypher 回答最初的业务问题。

### 示例 A：电商订单图谱

业务问题：
- “某个用户买过哪些商品？”
- “买过同一商品的用户还买了什么？”
- “某个订单从创建到发货经历了哪些状态？”

图模型：

```text
(User)-[:PLACED]->(Order)-[:CONTAINS]->(Product)
(Product)-[:BELONGS_TO]->(Category)
(Order)-[:PAID_BY]->(Payment)
```

适合解决：用户画像、关联推荐、订单追踪、风控排查。

### 示例 B：客服知识图谱

业务问题：
- “用户问题应该匹配哪条 FAQ？”
- “某个故障属于哪个产品模块？”
- “处理方案依赖哪些前置条件？”

图模型：

```text
(Question)-[:SIMILAR_TO]->(FAQ)-[:SOLVED_BY]->(Solution)
(FAQ)-[:BELONGS_TO]->(ProductModule)
(Solution)-[:REQUIRES]->(Condition)
```

适合解决：智能客服、售后排障、企业知识库问答、GraphRAG 上下文扩展。

## 图数据库在 GraphRAG 中的应用场景

普通 RAG 通常是“用户问题 → 向量检索相似文本 → 把文本交给大模型回答”。这种方式适合找相似内容，但它不擅长处理复杂关系，例如“某个产品故障和哪些模块、工单、解决方案有关”。GraphRAG 的核心思路是：先检索到相关节点，再沿着图关系扩展上下文，让大模型拿到更完整、更有结构的信息。

### GraphRAG 的典型流程

```text
用户问题
→ 关键词检索 / 向量检索找到相关节点
→ 沿图关系扩展上下文
→ 组织成结构化提示词
→ 交给大模型生成答案
```

在这个流程里，Neo4j 主要负责两件事：

- **存结构**：保存实体之间的关系，例如用户、订单、商品、FAQ、解决方案之间的连接。
- **扩上下文**：从一个命中的节点出发，继续查找相邻节点，补充大模型回答所需的背景信息。

### 场景 A：智能客服问答

用户问题：

```text
订单 ORD-001 为什么不能取消？
```

图模型可以设计为：

```text
(Order)-[:HAS_STATUS]->(OrderStatus)
(Order)-[:CONTAINS]->(Product)
(FAQ)-[:APPLIES_TO]->(OrderStatus)
(FAQ)-[:SOLVED_BY]->(Solution)
```

GraphRAG 检索过程：

1. 根据订单号找到 `Order` 节点。
2. 沿关系查询订单状态、商品、物流信息。
3. 根据状态找到相关 FAQ 和处理方案。
4. 把“订单状态 + 规则说明 + 解决方案”交给大模型生成客服回复。

价值：
- 回答不只依赖相似文本，还能结合真实订单状态。
- 可以解释“为什么不能取消”，而不是只返回一段泛泛的规则。

### 场景 B：企业知识库问答

用户问题：

```text
部署 Milvus 失败可能和哪些配置有关？
```

图模型可以设计为：

```text
(Error)-[:CAUSED_BY]->(Config)
(Config)-[:BELONGS_TO]->(Service)
(Error)-[:SOLVED_BY]->(Solution)
(Solution)-[:REQUIRES]->(Command)
```

GraphRAG 检索过程：

1. 用关键词或向量检索找到相关 `Error` 节点。
2. 沿关系扩展出配置项、服务模块、解决方案和命令。
3. 大模型根据结构化上下文生成排查步骤。

价值：
- 适合做运维排障、技术支持、内部知识库助手。
- 能把“错误现象 → 可能原因 → 解决步骤”串成清晰链路。

### 场景 C：推荐与关联分析

用户问题：

```text
买过机械键盘的用户还可能需要什么配件？
```

图模型可以设计为：

```text
(User)-[:BOUGHT]->(Product)
(Product)-[:BELONGS_TO]->(Category)
(Product)-[:COMPATIBLE_WITH]->(Product)
```

GraphRAG 检索过程：

1. 找到“机械键盘”对应的 `Product` 节点。
2. 查询共同购买、兼容配件、同类商品。
3. 把候选商品和推荐理由交给大模型生成自然语言推荐。

价值：
- 推荐结果可解释，例如“因为它和机械键盘兼容”。
- 比单纯向量相似更适合处理用户、商品、订单之间的复杂关系。

### Neo4j 适合放什么，不适合放什么

| 类型 | 是否适合放 Neo4j | 示例 |
|------|------------------|------|
| 实体 | 适合 | 用户、订单、商品、FAQ、错误码、解决方案 |
| 关系 | 适合 | 购买、包含、依赖、导致、解决、属于 |
| 结构化属性 | 适合 | 状态、评分、时间、类型、价格 |
| 大段原文 | 不一定适合 | 文档正文、PDF 全文、长篇知识库文章 |
| 向量 | 可选 | Neo4j 5.11+ 支持向量索引，也可以放在专门向量库中 |

实际项目中常见组合是：

```text
向量库 / 全文索引：负责召回相关文本或节点
Neo4j：负责关系扩展和结构化推理
大模型：负责理解问题、组织答案、生成自然语言回复
```

## 文件说明

### 1. neo4j_cypher_complete.cypher

Neo4j Cypher 语法大全，包含 18 个章节的循序渐进教程：

- **第 1 部分**: 清理环境
- **第 2 部分**: 创建节点 (CREATE)
- **第 3 部分**: 创建关系 (CREATE Relationship)
- **第 4 部分**: 查询节点 (MATCH + RETURN)
- **第 5 部分**: 条件查询 (WHERE)
- **第 6 部分**: 聚合函数 (Aggregation)
- **第 7 部分**: 关系查询
- **第 8 部分**: OPTIONAL MATCH (左连接)
- **第 9 部分**: UNION 合并查询结果
- **第 10 部分**: WITH 子句 (管道传递)
- **第 11 部分**: 更新数据 (SET)
- **第 12 部分**: 删除数据 (DELETE)
- **第 13 部分**: MERGE (不存在则创建)
- **第 14 部分**: 索引和约束 (Indexes & Constraints)
- **第 15 部分**: 复杂查询示例
- **第 16 部分**: 高级函数
- **第 17 部分**: 完整示例 - 电影知识图谱查询
- **第 18 部分**: 进阶实战 - 批量导入、全文检索、向量检索、GraphRAG

**使用方法**:
1. 打开 Neo4j Browser
2. 复制文件内容到查询窗口
3. 从头开始逐条执行（每条语句用 `;` 分隔）

> 注意：中文别名已使用反引号 `` 包裹，如 `RETURN p.name AS 姓名 `
> 如果使用 Neo4j 5.11 以下版本，可以先跳过向量索引相关小节。

### 2. neo4j_python_guide.py

Python 操作 Neo4j 的完整指南，包含：

- **Neo4jClient 类**: 封装的客户端，提供简洁的 API
- **CRUD 操作**: 创建、读取、更新、删除节点和关系
- **MERGE 操作**: 查找或创建
- **索引和约束**: 创建和管理
- **复杂查询**: 合作网络、统计分析、最短路径
- **进阶示例**: CSV 批量导入、全文检索、向量检索、GraphRAG 雏形

**使用方法**:

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入你的 Neo4j 连接信息和密码

# 3. 运行示例
python neo4j_python_guide.py
```

**基本用法示例**:

```python
from neo4j_python_guide import Neo4jClient

# 创建客户端
client = Neo4jClient()

# 创建节点
client.create_person("Alice", 30)
client.create_movie("The Matrix", 1999, 8.7)

# 创建关系
client.create_relationship_acted_in("Alice", "The Matrix", ["Neo"])

# 查询
persons = client.get_all_persons()
movies = client.get_movies_by_rating_range(8.5, 9.0)

# 更新
client.update_person_age("Alice", 31)

# 删除
client.delete_person("Alice")

# 关闭连接
client.close()
```

## 环境要求

- Neo4j 4.0+ (推荐使用 Neo4j 5.x)
- Python 3.8+
- neo4j Python 驱动 5.0+

## 快速启动 Neo4j (Docker)

```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_neo4j_password \
  neo4j:latest
```

启动后:
- Browser: http://localhost:7474
- Bolt: bolt://localhost:7687

如果你按上面的 Docker 命令启动，请在 `.env` 中保持一致：

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
NEO4J_DATABASE=neo4j
```

## 建议学习顺序

1. 先读本 README 的“核心概念”和“业务建模”。
2. 在 Neo4j Browser 中执行 `neo4j_cypher_complete.cypher` 的第 1-17 部分。
3. 运行 `python neo4j_python_guide.py`，理解 Python 如何封装 CRUD。
4. 学完基础后，再学习第 18 部分和 `example_advanced_usage()` 中的批量导入、全文检索、向量检索。
