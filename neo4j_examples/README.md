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
