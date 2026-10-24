"""Agent 记忆图谱 — 把经历存成图，经验可以复用

Neo4j 连接信息从项目根目录 .env 读取：
    NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD / NEO4J_DATABASE
"""

import os

from dotenv import load_dotenv
from neo4j import GraphDatabase


class MemoryGraph:
    """Agent 记忆图谱——把经历存成图，经验可以复用"""

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

    def remember(self, agent: str, problem: str, tool: str, result: str, lesson: str):
        """记录一次经历：Agent 遇到问题 → 用工具 → 得到结果 → 学到经验"""
        with self.driver.session(database=self.database) as session:
            session.run("""
            MERGE (a:Agent {name: $agent})
            MERGE (p:Problem {text: $problem})
            MERGE (t:Tool {name: $tool})
            MERGE (r:Result {text: $result})
            MERGE (l:Lesson {text: $lesson})
            MERGE (a)-[:ENCOUNTERED]->(p)
            MERGE (a)-[:USED]->(t)
            MERGE (t)-[:SOLVED]->(p)
            MERGE (t)-[:PRODUCED]->(r)
            MERGE (r)-[:TAUGHT]->(l)
            """, agent=agent, problem=problem, tool=tool, result=result, lesson=lesson).consume()

    def recall(self, keyword: str):
        """遇到新问题，顺着图谱找回类似问题用过的工具和教训"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
            MATCH (p:Problem)-[:SOLVED]-(t:Tool)-[:PRODUCED]->(r:Result)-[:TAUGHT]->(l:Lesson)
            WHERE p.text CONTAINS $kw OR t.name CONTAINS $kw
            RETURN p.text AS problem, t.name AS tool, l.text AS lesson
            """, kw=keyword)
            return [record.data() for record in result]


# === 测试 ===
if __name__ == "__main__":
    memory = MemoryGraph()

    if not memory.verify_connection():
        print("❌ 连接失败，请检查项目根目录 .env 中的 NEO4J_* 配置")
        memory.close()
        raise SystemExit(1)

    memory.remember("agent_001", "用户要查订单状态", "order_search 工具", "返回 3 条订单", "查状态先走 order_search，别硬编 SQL")
    memory.remember("agent_001", "用户问发票怎么开", "invoice_guide 文档检索", "给出开票流程", "发票类问题先查 invoice_guide")

    print("🧠 找回「订单」相关的经验：")
    for m in memory.recall("订单"):
        print(f"  • 问题：{m['problem']} → 工具：{m['tool']} → 教训：{m['lesson']}")

    memory.close()
    print("✅ 记忆图谱演示完成")
