"""知识图谱构建器 — 从 AI 术语数据构建知识图谱

作用：把 AI 领域的术语（如 Transformer、RAG、Agent）存入 Neo4j 图数据库，
      以「概念（Concept）节点 + 关系（RELATED 边）」的形式组织成知识图谱，
      演示图数据库如何表达概念之间的联系（如「BERT 基于 Transformer」）。

核心结构：
    class KnowledgeGraphBuilder  —— 封装 Neo4j 的增查操作
        - add_concept()                添加/更新一个概念节点
        - add_relation()               在两个概念之间建立带类型的关系
        - get_related_concepts()       查询与某概念直接相关的所有概念（含入向/出向）
        - get_concepts_by_category()   按分类（如「Agent架构」）批量查询概念
        - verify_connection()          检测 Neo4j 连接是否可用

运行方式（在项目根目录执行，需本地 Neo4j 已启动）：
    python neo4j_examples/generate_graph_demo1.py

Neo4j 连接信息从项目根目录 .env 读取：
    NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD / NEO4J_DATABASE
"""

import os

from dotenv import load_dotenv
from neo4j import GraphDatabase


class KnowledgeGraphBuilder:
    """知识图谱构建器 — 从术语数据构建 AI 知识图谱"""

    def __init__(self, uri=None, user=None, password=None, database=None):
        # 从 .env 读取连接配置，也可通过构造参数覆盖
        load_dotenv()
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        self.driver.close()

    def verify_connection(self) -> bool:
        """测试连接是否可用"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 AS ok")
                return result.single()["ok"] == 1
        except Exception:
            return False

    '''
        添加一个节点和属性
        实际是创建一个节点和添加属性的组合，也可以修改属性
    '''
    def add_concept(self, name: str, category: str, description: str):
        """添加一个概念节点"""
        query = """
        MERGE (c:Concept {name: $name})
        SET c.category = $category, c.description = $description
        """
        with self.driver.session(database=self.database) as session:
            session.run(query, name=name, category=category, description=description).consume()

    def add_relation(self, from_concept: str, relation_type: str, to_concept: str):
        """添加概念之间的关系"""
        query = """
        MATCH (a:Concept {name: $from})
        MATCH (b:Concept {name: $to})
        MERGE (a)-[r:RELATED {type: $rel_type}]->(b)
        """
        with self.driver.session(database=self.database) as session:
            # 注意：$from 是 Cypher 参数名，而 from 是 Python 关键字，
            # 不能用关键字参数传参，必须用字典显式传 $from
            session.run(query, {"from": from_concept, "to": to_concept, "rel_type": relation_type}).consume()

    def get_related_concepts(self, concept_name: str) -> list:
        """查找和某个概念相关的所有概念"""
        # 使用无向匹配 -[r:RELATED]-，因为关系方向表示语义（如「基于」），
        # 「相关概念」应同时包含入向和出向，否则只会查到从该概念出发的关系
        query = """
        MATCH (c:Concept {name: $name})-[r:RELATED]-(other:Concept)
        RETURN other.name AS name, r.type AS relation
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, name=concept_name)
            return [record.data() for record in result]

    def get_concepts_by_category(self, category: str) -> list:
        """查找某个分类下的所有概念"""
        query = """
        MATCH (c:Concept {category: $category})
        RETURN c.name AS name, c.description AS description
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, category=category)
            return [record.data() for record in result]


# === 构建知识图谱 ===
if __name__ == "__main__":
    builder = KnowledgeGraphBuilder()

    if not builder.verify_connection():
        print("❌ 连接失败，请检查项目根目录 .env 中的 NEO4J_* 配置")
        builder.close()
        raise SystemExit(1)

    print("🏗️ 构建 AI 知识图谱...\n")

    # 添加概念（从 `01_AI_terminology_explanations.md` 提取）
    concepts = [
        ("Transformer", "模型架构", "一眼看全局、自动划重点的架构"),
        ("BERT", "模型架构", "前后都看，像做完形填空"),
        ("GPT", "模型架构", "逐字生成的 Decoder-only 模型"),
        ("RAG", "知识增强", "AI 先查资料再回答，开卷考试"),
        ("Fine-tuning", "知识增强", "用专业知识再培训，通才变专家"),
        ("RLHF", "模型优化", "人类给 AI 打分，教它好好说话"),
        ("Agent", "Agent架构", "能自己想办法干活的 AI"),
        ("MCP", "Agent架构", "AI 的万能 USB-C 接口"),
        ("Prompt Engineering", "提示交互", "怎么提问，AI 才给你好答案"),
    ]

    for name, category, desc in concepts:
        builder.add_concept(name, category, desc)
        print(f"  ✅ 添加概念：{name}（{category}）")

    print("\n🔗 添加关系...\n")

    # 添加关系
    relations = [
        ("BERT", "基于", "Transformer"),
        ("GPT", "基于", "Transformer"),
        ("RAG", "增强", "Fine-tuning"),
        ("RLHF", "优化", "GPT"),
        ("Agent", "使用", "Prompt Engineering"),
        ("Agent", "使用", "MCP"),
        ("RAG", "需要", "Prompt Engineering"),
    ]

    for from_c, rel_type, to_c in relations:
        builder.add_relation(from_c, rel_type, to_c)
        print(f"  🔗 {from_c} --[{rel_type}]--> {to_c}")

    # 查询验证
    print("\n🔍 Transformer 相关的技术：")
    related = builder.get_related_concepts("Transformer")
    for r in related:
        print(f"  • {r['name']}（关系：{r['relation']}）")

    print("\n📂 Agent 架构下的技术：")
    agents = builder.get_concepts_by_category("Agent架构")
    for a in agents:
        print(f"  • {a['name']}: {a['description']}")

    builder.close()
    print("\n✅ 知识图谱构建完成！")
