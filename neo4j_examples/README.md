# Neo4j 示例文件说明

## 文件结构

```
neo4j_examples/
├── neo4j_cypher_complete.cypher   # Neo4j Cypher 语法大全
├── neo4j_python_guide.py          # Python 操作 Neo4j 指南
├── requirements.txt               # Python 依赖
├── .env.example                   # 环境变量配置示例
└── README.md                      # 本说明文件
```

## 文件说明

### 1. neo4j_cypher_complete.cypher

Neo4j Cypher 语法大全，包含 17 个章节的循序渐进教程：

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

**使用方法**:
1. 打开 Neo4j Browser
2. 复制文件内容到查询窗口
3. 从头开始逐条执行（每条语句用 `;` 分隔）

> 注意：中文别名已使用反引号 `` 包裹，如 `RETURN p.name AS 姓名 `

### 2. neo4j_python_guide.py

Python 操作 Neo4j 的完整指南，包含：

- **Neo4jClient 类**: 封装的客户端，提供简洁的 API
- **CRUD 操作**: 创建、读取、更新、删除节点和关系
- **MERGE 操作**: 查找或创建
- **索引和约束**: 创建和管理
- **复杂查询**: 合作网络、统计分析、最短路径

**使用方法**:

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入你的 Neo4j 连接信息

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
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

启动后:
- Browser: http://localhost:7474
- Bolt: bolt://localhost:7687
