#!/usr/bin/env python3
"""
Neo4j Python 驱动使用指南
演示如何在 Python 中执行 Neo4j 的增删改查操作
"""

import os
import csv
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase, Session, Transaction
from typing import Optional, List, Dict, Any

# 加载环境变量
load_dotenv()

# ============================================================
# 第一部分：连接配置
# ============================================================

# 从环境变量读取配置 (推荐方式)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


class Neo4jClient:
    """Neo4j 数据库客户端封装类"""

    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, password: Optional[str] = NEO4J_PASSWORD, database: str = NEO4J_DATABASE):
        """
        初始化 Neo4j 客户端

        Args:
            uri: Neo4j 连接地址，如 bolt://localhost:7687
            user: 用户名
            password: 密码
            database: 数据库名称
        """
        if not password:
            raise ValueError("请先在 .env 中设置 NEO4J_PASSWORD，或在创建 Neo4jClient 时传入 password 参数")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self._verify_connection()

    def _verify_connection(self) -> None:
        """验证连接是否成功"""
        try:
            self.driver.verify_connectivity()
            print(f"成功连接到 Neo4j 数据库：{self.database}")
        except Exception as e:
            print(f"连接失败：{e}")
            raise

    def close(self) -> None:
        """关闭驱动连接"""
        if self.driver:
            self.driver.close()
            print("Neo4j 连接已关闭")

    # --------------------------------------------------------
    # 第二部分：基本查询方法， 跟具体的某个节点无关，跟你想查电影，还是演员，都无关
    # --------------------------------------------------------

    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        执行 Cypher 查询语句

        Args:
            query: Cypher 查询语句
            parameters: 查询参数字典

        Returns:
            查询结果列表
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def execute_write(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        执行写入操作 (CREATE/MERGE/SET/DELETE)

        Args:
            query: Cypher 写入语句
            parameters: 查询参数字典

        Returns:
            操作结果统计
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            result.consume()  # 消耗结果以确保写入完成
            return {"status": "success"}

    # --------------------------------------------------------
    # 第三部分：封装的 CRUD 操作
    # --------------------------------------------------------

    # ==================== CREATE 操作 ====================

    def create_person(self, name: str, age: int) -> Dict[str, Any]:
        """
        创建一个人节点

        Args:
            name: 姓名
            age: 年龄

        Returns:
            创建的节点信息
        """
        query = """
        CREATE (p:Person {name: $name, age: $age})
        RETURN p
        """
        result = self.execute_query(query, {"name": name, "age": age})
        return result[0]["p"] if result else {}

    def create_movie(self, title: str, released: int, rating: float) -> Dict[str, Any]:
        """
        创建一个电影节点

        Args:
            title: 电影标题
            released: 发行年份
            rating: 评分

        Returns:
            创建的节点信息
        """
        query = """
        CREATE (m:Movie {title: $title, released: $released, rating: $rating})
        RETURN m
        """
        result = self.execute_query(query, {"title": title, "released": released, "rating": rating})
        return result[0]["m"] if result else {}

    def bulk_create_movies(self, movies: List[Dict[str, Any]]) -> int:
        """
        批量创建电影节点

        Args:
            movies: 电影列表，每个元素包含 title、released、rating

        Returns:
            导入的电影数量

        Cypher 说明:
            UNWIND - 把列表拆成多行，适合批量写入
            MERGE - 已存在则匹配，不存在则创建，避免重复导入
        """
        query = """
        UNWIND $movies AS row
        MERGE (m:Movie {title: row.title})
        SET m.released = row.released,
            m.rating = row.rating
        RETURN count(m) AS imported_count
        """
        result = self.execute_query(query, {"movies": movies})
        return result[0]["imported_count"] if result else 0

    def load_movies_from_csv(self, csv_path: str) -> int:
        """
        从本地 CSV 文件读取电影数据，并批量写入 Neo4j

        CSV 示例表头:
            title,released,rating

        说明:
            这里用 Python 读取 CSV，再通过 UNWIND 写入 Neo4j。
            这种方式不依赖 Neo4j 服务器的 import 目录，Windows 学员更容易跑通。
        """
        movies = []
        with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                movies.append({
                    "title": row["title"],
                    "released": int(row["released"]),
                    "rating": float(row["rating"]),
                })
        return self.bulk_create_movies(movies)

    def create_director(self, name: str, birth_year: int) -> Dict[str, Any]:
        """
        创建一个导演节点
        """
        query = """
        CREATE (d:Director {name: $name, birthYear: $birthYear})
        RETURN d
        """
        result = self.execute_query(query, {"name": name, "birthYear": birth_year})
        return result[0]["d"] if result else {}

    def create_relationship_acted_in(self, person_name: str, movie_title: str, roles: List[str]) -> Dict[str, Any]:
        """
        创建演员出演电影的关系

        Args:
            person_name: 演员姓名
            movie_title: 电影标题
            roles: 角色列表

        Returns:
            创建的关系信息

        Cypher 说明:
            MATCH - 先查找存在的演员和电影节点
            CREATE - 创建从演员到电影的 ACTED_IN 关系
            $param - 参数占位符，防止注入攻击
        """
        query = """
        MATCH (p:Person {name: $person_name})
        MATCH (m:Movie {title: $movie_title})
        CREATE (p)-[r:ACTED_IN {roles: $roles}]->(m)
        RETURN r
        """
        result = self.execute_query(
            query,
            {"person_name": person_name, "movie_title": movie_title, "roles": roles}
        )
        return result[0]["r"] if result else {}

    def create_relationship_directed(self, director_name: str, movie_title: str, year: int) -> Dict[str, Any]:
        """
        创建导演执导电影的关系
        """
        query = """
        MATCH (d:Director {name: $director_name})
        MATCH (m:Movie {title: $movie_title})
        CREATE (d)-[:DIRECTED {year: $year}]->(m)
        RETURN d, m
        """
        result = self.execute_query(
            query,
            {"director_name": director_name, "movie_title": movie_title, "year": year}
        )
        return {"director": result[0]["d"], "movie": result[0]["m"]} if result else {}

    # ==================== READ 操作 ====================

    def get_all_persons(self) -> List[Dict[str, Any]]:
        """获取所有人员节点"""
        query = "MATCH (p:Person) RETURN p.name AS name, p.age AS age"
        return self.execute_query(query)

    def get_person_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        根据姓名查询人员

        Args:
            name: 姓名

        Returns:
            人员信息或 None
        """
        query = """
        MATCH (p:Person {name: $name})
        RETURN p.name AS name, p.age AS age
        """
        result = self.execute_query(query, {"name": name})
        return result[0] if result else None

    def get_movies_by_rating_range(self, min_rating: float, max_rating: float) -> List[Dict[str, Any]]:
        """
        查询指定评分范围内的电影

        Args:
            min_rating: 最低评分
            max_rating: 最高评分

        Returns:
            电影列表

        Cypher 说明:
            WHERE - 条件过滤，只返回评分在范围内的电影
            ORDER BY DESC - 按评分降序排列
        """
        query = """
        MATCH (m:Movie)
        WHERE m.rating >= $min_rating AND m.rating <= $max_rating
        RETURN m.title AS title, m.released AS released, m.rating AS rating
        ORDER BY m.rating DESC
        """
        return self.execute_query(query, {"min_rating": min_rating, "max_rating": max_rating})

    def get_acted_in_movies(self, person_name: str) -> List[Dict[str, Any]]:
        """
        查询演员出演过的电影

        Args:
            person_name: 演员姓名

        Returns:
            电影列表

        Cypher 说明:
            -[r:ACTED_IN]-> - 匹配从演员到电影的 ACTED_IN 关系
            r.roles - 返回关系上的 roles 属性（角色名）
        """
        query = """
        MATCH (p:Person {name: $person_name})-[r:ACTED_IN]->(m:Movie)
        RETURN m.title AS title, r.roles AS roles
        """
        return self.execute_query(query, {"person_name": person_name})

    def get_director_movies(self, director_name: str) -> List[Dict[str, Any]]:
        """
        查询导演的作品

        Args:
            director_name: 导演姓名

        Returns:
            电影列表
        """
        query = """
        MATCH (d:Director {name: $director_name})-[r:DIRECTED]->(m:Movie)
        RETURN m.title AS title, m.rating AS rating, r.year AS year
        ORDER BY r.year DESC
        """
        return self.execute_query(query, {"director_name": director_name})

    def search_persons_by_age_range(self, min_age: int, max_age: int) -> List[Dict[str, Any]]:
        """
        搜索指定年龄范围的人员

        Args:
            min_age: 最小年龄
            max_age: 最大年龄

        Returns:
            人员列表
        """
        query = """
        MATCH (p:Person)
        WHERE p.age >= $min_age AND p.age <= $max_age
        RETURN p.name AS name, p.age AS age
        """
        return self.execute_query(query, {"min_age": min_age, "max_age": max_age})

    # ==================== UPDATE 操作 ====================

    def update_person_age(self, name: str, new_age: int) -> Dict[str, Any]:
        """
        更新人员年龄

        Args:
            name: 姓名
            new_age: 新年龄

        Returns:
            更新后的节点信息

        Cypher 说明:
            SET - 更新节点的属性
        """
        query = """
        MATCH (p:Person {name: $name})
        SET p.age = $new_age
        RETURN p.name AS name, p.age AS age
        """
        result = self.execute_query(query, {"name": name, "new_age": new_age})
        return result[0] if result else {}

    def remove_person_city(self, name: str) -> Dict[str, Any]:
        """
        移除人员的城市属性

        Args:
            name: 姓名

        Cypher 说明:
            REMOVE - 删除节点的属性或标签


        Returns:
            更新后的节点信息
        """
        query = """
        MATCH (p:Person {name: $name})
        REMOVE p.city
        RETURN p
        """
        result = self.execute_query(query, {"name": name})
        return result[0]["p"] if result else {}

    def add_vip_label(self, name: str) -> Dict[str, Any]:
        """
        为人员添加 VIP 标签

        Args:
            name: 姓名

        Returns:
            更新后的节点信息

        Cypher 说明:
            SET p:VIP - 给节点添加 VIP 标签
            labels(p) - 返回节点的所有标签
        """
        query = """
        MATCH (p:Person {name: $name})
        SET p:VIP
        RETURN p, labels(p) AS labels
        """
        result = self.execute_query(query, {"name": name})
        return result[0] if result else {}

    def remove_vip_label(self, name: str) -> Dict[str, Any]:
        """
        移除人员的 VIP 标签

        Args:
            name: 姓名

        Returns:
            更新后的节点信息
        """
        query = """
        MATCH (p:Person {name: $name})
        REMOVE p:VIP
        RETURN p, labels(p) AS labels
        """
        result = self.execute_query(query, {"name": name})
        return result[0] if result else {}

    # ==================== DELETE 操作 ====================

    def delete_person(self, name: str) -> bool:
        """
        删除人员节点 (使用 DETACH DELETE 同时删除关系)

        Args:
            name: 姓名

        Returns:
            是否删除成功

        Cypher 说明:
            DETACH DELETE - 先断开所有关系，再删除节点
        """
        query = """
        MATCH (p:Person {name: $name})
        DETACH DELETE p
        """
        try:
            self.execute_write(query, {"name": name})
            return True
        except Exception as e:
            print(f"删除失败：{e}")
            return False

    def delete_movie(self, title: str) -> bool:
        """
        删除电影节点

        Args:
            title: 电影标题

        Returns:
            是否删除成功
        """
        query = """
        MATCH (m:Movie {title: $title})
        DETACH DELETE m
        """
        try:
            self.execute_write(query, {"title": title})
            return True
        except Exception as e:
            print(f"删除失败：{e}")
            return False

    def delete_relationship(self, person_name: str, movie_title: str) -> bool:
        """
        删除演员与电影之间的关系

        Args:
            person_name: 演员姓名
            movie_title: 电影标题

        Returns:
            是否删除成功

        Cypher 说明:
            MATCH - 匹配演员和电影之间的 ACTED_IN 关系
            DELETE r - 删除匹配到的关系
        """
        query = """
        MATCH (p:Person {name: $person_name})-[r:ACTED_IN]->(m:Movie {title: $movie_title})
        DELETE r
        """
        try:
            self.execute_write(query, {"person_name": person_name, "movie_title": movie_title})
            return True
        except Exception as e:
            print(f"删除关系失败：{e}")
            return False

    def clear_database(self) -> bool:
        """
        清空数据库中所有节点和关系 (谨慎使用)

        Returns:
            是否清空成功
        """
        query = "MATCH (n) DETACH DELETE n"
        try:
            self.execute_write(query)
            return True
        except Exception as e:
            print(f"清空数据库失败：{e}")
            return False

    # ==================== MERGE 操作 ====================

    def merge_person(self, name: str, age: Optional[int] = None) -> Dict[str, Any]:
        """
        MERGE 操作：查找或创建人员

        Args:
            name: 姓名
            age: 年龄 (仅在创建时设置)

        Returns:
            节点信息及操作类型

        Cypher 说明:
            MERGE - 查找或创建，类似"upsert"操作
            ON CREATE SET - 仅在节点被创建时执行
            ON MATCH SET - 仅在节点已存在时执行
        """
        if age is not None:
            query = """
            MERGE (p:Person {name: $name})
            ON CREATE SET p.age = $age
            RETURN p, 'created' AS action
            """
            result = self.execute_query(query, {"name": name, "age": age})
        else:
            query = """
            MERGE (p:Person {name: $name})
            RETURN p, 'matched' AS action
            """
            result = self.execute_query(query, {"name": name})
        return result[0] if result else {}

    # ==================== 索引和约束 ====================

    def create_unique_constraint(self, label: str, property_name: str) -> bool:
        """
        创建唯一约束

        Args:
            label: 节点标签
            property_name: 属性名

        Returns:
            是否创建成功

        Cypher 说明:
            CONSTRAINT - 约束，保证数据完整性
            REQUIRE ... IS UNIQUE - 要求属性值唯一
            如果约束已存在会报错，用 try/except 捕获
        """
        # Neo4j 5.x 支持 IF NOT EXISTS，4.x 不支持
        # 使用 try/except 方式处理已存在的情况
        query = f"""
        CREATE CONSTRAINT {label}_{property_name}_unique
        FOR (n:{label}) REQUIRE n.{property_name} IS UNIQUE
        """
        try:
            self.execute_write(query)
            return True
        except Exception as e:
            # 如果约束已存在，会抛出异常，这里忽略
            if "already exists" in str(e).lower():
                print(f"约束已存在：{label}_{property_name}_unique")
                return True
            print(f"创建约束失败：{e}")
            return False

    def create_index(self, label: str, property_name: str) -> bool:
        """
        创建索引

        Args:
            label: 节点标签
            property_name: 属性名

        Returns:
            是否创建成功

        Cypher 说明:
            INDEX - 索引，加速属性查询
            ON (n.prop) - 在指定属性上创建索引
        """
        query = f"""
        CREATE INDEX {label}_{property_name}_index IF NOT EXISTS
        FOR (n:{label}) ON (n.{property_name})
        """
        try:
            self.execute_write(query)
            return True
        except Exception as e:
            print(f"创建索引失败：{e}")
            return False

    def show_constraints(self) -> List[Dict[str, Any]]:
        """显示所有约束"""
        # SHOW - 显示数据库中的约束或索引信息
        query = "SHOW CONSTRAINTS"
        return self.execute_query(query)

    def show_indexes(self) -> List[Dict[str, Any]]:
        """显示所有索引"""
        query = "SHOW INDEXES"
        return self.execute_query(query)

    def create_movie_fulltext_index(self) -> bool:
        """
        创建电影全文索引

        适用场景:
            用户输入关键词，例如 Matrix、Dark、Interstellar，
            希望按文本相关度搜索电影标题。
        """
        query = """
        CREATE FULLTEXT INDEX movie_fulltext_index IF NOT EXISTS
        FOR (m:Movie) ON EACH [m.title]
        """
        try:
            self.execute_write(query)
            return True
        except Exception as e:
            print(f"创建全文索引失败：{e}")
            return False

    def search_movies_fulltext(self, keyword: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        使用全文索引搜索电影

        Args:
            keyword: 搜索关键词
            limit: 返回数量
        """
        query = """
        CALL db.index.fulltext.queryNodes('movie_fulltext_index', $keyword)
        YIELD node, score
        RETURN node.title AS title,
               node.released AS released,
               node.rating AS rating,
               score
        ORDER BY score DESC
        LIMIT $limit
        """
        return self.execute_query(query, {"keyword": keyword, "limit": limit})

    def create_movie_vector_index(self, dimension: int = 3) -> bool:
        """
        创建电影向量索引

        教学说明:
            真实项目中 embedding 通常来自文本向量模型。
            这里用 3 维模拟向量，便于学生理解 GraphRAG 的检索思路。
        """
        query = f"""
        CREATE VECTOR INDEX movie_embedding_index IF NOT EXISTS
        FOR (m:Movie) ON (m.embedding)
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {dimension},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
        try:
            self.execute_write(query)
            return True
        except Exception as e:
            print(f"创建向量索引失败：{e}")
            return False

    def set_demo_movie_embeddings(self) -> int:
        """
        为示例电影写入模拟 embedding

        neo4j 支持向量化的数据和检索，但是实际中不常使用向量化的检索
        实际中：从向量库中做召回相似度匹配的数据，从图数据库中检索实体之间的关联关系，一起交给LLM做推理和回答


        注意:
            这是教学用的简化向量，不代表真实语义效果。
            真实 GraphRAG 会把电影简介、知识文本等送入 Embedding 模型生成向量。
        """
        embeddings = [
            {"title": "The Matrix", "embedding": [0.90, 0.10, 0.20]},
            {"title": "Inception", "embedding": [0.80, 0.25, 0.30]},
            {"title": "Interstellar", "embedding": [0.20, 0.90, 0.35]},
            {"title": "The Dark Knight", "embedding": [0.65, 0.30, 0.70]},
        ]
        query = """
        UNWIND $embeddings AS row
        MATCH (m:Movie {title: row.title})
        SET m.embedding = row.embedding
        RETURN count(m) AS updated_count
        """
        result = self.execute_query(query, {"embeddings": embeddings})
        return result[0]["updated_count"] if result else 0

    def search_similar_movies_by_vector(self, query_vector: List[float], limit: int = 3) -> List[Dict[str, Any]]:
        """
        使用向量索引查询相似电影

        GraphRAG 思路:
            先用向量召回相关节点，再沿着图关系扩展导演、演员、公司等上下文。
        """
        query = """
        CALL db.index.vector.queryNodes('movie_embedding_index', $limit, $query_vector)
        YIELD node, score
        OPTIONAL MATCH (node)<-[:DIRECTED]-(d:Director)
        RETURN node.title AS title,
               node.rating AS rating,
               d.name AS director,
               score
        ORDER BY score DESC
        """
        return self.execute_query(query, {"query_vector": query_vector, "limit": limit})

    # ==================== 复杂查询 ====================

    def get_collaboration_network(self) -> List[Dict[str, Any]]:
        """
        获取演员合作网络 (共同出演同一电影的演员)

        Returns:
            合作关系列表

        Cypher 说明:
            -[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]- - 查找共同出演同一电影的演员
            p1.name < p2.name - 避免重复 (字母顺序小的在前)
        """
        query = """
        MATCH (p1:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(p2:Person)
        WHERE p1.name < p2.name
        RETURN p1.name AS actor1, p2.name AS actor2, m.title AS movie
        ORDER BY p1.name, p2.name
        """
        return self.execute_query(query)

    def get_director_statistics(self) -> List[Dict[str, Any]]:
        """
        获取导演统计信息

        Returns:
            导演统计列表

        Cypher 说明:
            collect() - 收集所有电影标题为列表
            count() - 统计电影数量
            avg() - 计算平均评分
        """
        query = """
        MATCH (d:Director)-[:DIRECTED]->(m:Movie)
        RETURN d.name AS director,
               collect(m.title) AS movies,
               count(m) AS movie_count,
               avg(m.rating) AS avg_rating
        ORDER BY avg_rating DESC
        """
        return self.execute_query(query)

    def find_shortest_path(self, person1_name: str, person2_name: str) -> List[Dict[str, Any]]:
        """
        查找两人之间的最短路径

        Args:
            person1_name: 第一个人的姓名
            person2_name: 第二个人的姓名

        Returns:
            最短路径信息

        Cypher 说明:
            shortestPath() - 查找两个节点之间的最短路径
            [*] - 任意长度、任意类型的关系
        """
        query = """
        MATCH path = shortestPath(
            (p1:Person {name: $name1})-[*]-(p2:Person {name: $name2})
        )
        RETURN path
        """
        return self.execute_query(query, {"name1": person1_name, "name2": person2_name})


# ============================================================
# 第四部分：使用示例
# ============================================================

def example_usage():
    """演示 Neo4jClient 的各种用法"""

    # 创建客户端实例
    client = Neo4jClient()

    try:
        # 清空数据库 (可选)
        print("\n=== 清空数据库 ===")
        client.clear_database()

        # --- CREATE 示例 ---
        print("\n=== 创建数据 ===")
        client.create_person("Alice", 30)
        client.create_person("Bob", 25)
        client.create_person("Charlie", 35)
        client.create_movie("The Matrix", 1999, 8.7)
        client.create_movie("Inception", 2010, 8.8)

        # 创建关系
        client.create_relationship_acted_in("Alice", "The Matrix", ["Neo"])
        client.create_relationship_acted_in("Bob", "Inception", ["Cobb"])

        # --- READ 示例 ---
        print("\n=== 查询所有人员 ===")
        persons = client.get_all_persons()
        for p in persons:
            print(f"  {p}")

        print("\n=== 查询特定人员 ===")
        alice = client.get_person_by_name("Alice")
        print(f"  Alice: {alice}")

        print("\n=== 查询评分范围内的电影 ===")
        movies = client.get_movies_by_rating_range(8.5, 9.0)
        for m in movies:
            print(f"  {m}")

        # --- UPDATE 示例 ---
        print("\n=== 更新数据 ===")
        updated = client.update_person_age("Alice", 31)
        print(f"  更新后 Alice: {updated}")

        # --- 复杂查询 ---
        print("\n=== 导演统计 ===")
        # 先创建导演数据
        client.create_director("Lana Wachowski", 1965)
        client.create_director("Christopher Nolan", 1970)
        client.create_movie("Interstellar", 2014, 8.6)
        client.create_movie("The Dark Knight", 2008, 9.0)
        client.create_relationship_directed("Lana Wachowski", "The Matrix", 1999)
        client.create_relationship_directed("Christopher Nolan", "Inception", 2010)
        client.create_relationship_directed("Christopher Nolan", "Interstellar", 2014)
        client.create_relationship_directed("Christopher Nolan", "The Dark Knight", 2008)

        stats = client.get_director_statistics()
        for s in stats:
            print(f"  {s}")

        # --- DELETE 示例 ---
        print("\n=== 删除数据 ===")
        client.delete_person("Charlie")
        print("  已删除 Charlie")

        # --- MERGE 示例 ---
        print("\n=== MERGE 操作 ===")
        result = client.merge_person("David", 28)
        print(f"  MERGE 结果：{result}")

        # --- 索引和约束 ---
        print("\n=== 创建索引和约束 ===")
        client.create_unique_constraint("Person", "name")
        client.create_index("Movie", "title")
        print("  约束:", client.show_constraints())
        print("  索引:", client.show_indexes())

    finally:
        # 关闭连接
        client.close()


def example_advanced_usage():
    """
    进阶示例：批量导入、全文检索、向量检索/GraphRAG 雏形

    这个方法的作用:
        把前面学到的基础 CRUD，进一步连接到真实项目中常见的三类能力：

        1. 批量导入:
           企业数据通常来自 CSV、Excel、数据库导出文件，不会一条一条手动 CREATE。
           本示例用 Python 读取 data/movies.csv，再通过 UNWIND 批量写入 Neo4j。

        2. 全文检索:
           当用户输入关键词时，例如 "Matrix"，系统需要按文本相关度查找电影。
           本示例创建 FULLTEXT INDEX，并用 db.index.fulltext.queryNodes() 查询。

        3. 向量检索 / GraphRAG:
           GraphRAG 常见流程是“先用向量召回相关节点，再沿图关系扩展上下文”。
           本示例用 3 维模拟 embedding 演示语义召回，再把电影、导演等图结构信息一起返回。

    适合解决的问题:
        - 业务数据如何批量进入图数据库？
        - 用户关键词如何在 Neo4j 中搜索？
        - 图数据库如何和 RAG、Embedding、语义检索结合？

    说明:
        这部分依赖 Neo4j 5.x 的全文索引和向量索引能力。
        如果本地 Neo4j 版本较低，可以先学习 example_usage()。
    """

    client = Neo4jClient()

    try:
        # 第一步：模拟真实项目中的“数据接入”
        # 学生可以把 data/movies.csv 理解成业务系统导出的电影数据。
        print("\n=== 进阶示例：批量导入 CSV ===")
        csv_path = Path(__file__).parent / "data" / "movies.csv"
        if csv_path.exists():
            imported_count = client.load_movies_from_csv(str(csv_path))
            print(f"  从 CSV 导入/更新电影数量：{imported_count}")
        else:
            print(f"  未找到示例 CSV：{csv_path}")

        # 第二步：模拟搜索框能力
        # 适合“用户输入关键词，系统返回相关节点”的场景。
        print("\n=== 进阶示例：全文检索 ===")
        if client.create_movie_fulltext_index():
            results = client.search_movies_fulltext("Matrix")
            for item in results:
                print(f"  {item}")

        # 第三步：模拟 GraphRAG 的召回阶段
        # 真实项目中 query_vector 会来自 Embedding 模型，这里用固定向量降低学习门槛。
        print("\n=== 进阶示例：向量检索 / GraphRAG 雏形 ===")
        if client.create_movie_vector_index():
            updated_count = client.set_demo_movie_embeddings()
            print(f"  写入模拟向量数量：{updated_count}")
            results = client.search_similar_movies_by_vector([0.85, 0.15, 0.25])
            for item in results:
                print(f"  {item}")

    finally:
        client.close()


def example_transaction_usage():
    """
    演示 Neo4j Python Driver 的事务写入用法

    这个函数的作用:
        通过 session.execute_write() 执行一个写事务，在事务中创建电影节点。

    为什么要学习事务:
        1. 事务可以保证一组数据库操作要么全部成功，要么全部失败。
        2. 真实项目中经常需要在一次业务操作里同时创建多个节点和关系。
        3. 使用 execute_write() 可以让 Neo4j Driver 帮我们管理事务生命周期。

    重要注意:
        tx.run() 返回的 Result 只能在事务函数内部读取。
        不要把 Result 直接 return 到事务外再调用 result.single()，
        因为事务关闭后 Result 游标会失效，可能触发 ResultConsumedError。
    """

    client = Neo4jClient()

    try:
        with client.driver.session(database=client.database) as session:
            # 定义事务函数：所有需要放在同一个事务里的数据库操作，都写在这个函数中。
            def create_movie_tx(tx: Transaction, title: str, year: int, rating: float) -> Dict[str, Any]:
                query = """
                CREATE (m:Movie {title: $title, released: $year, rating: $rating})
                RETURN m.title AS title, m.released AS released, m.rating AS rating
                """
                result = tx.run(query, title=title, year=year, rating=rating)

                # 在事务函数内部消费 Result，并转换成普通 dict 再返回。
                # 这样事务关闭后，外部拿到的是普通 Python 数据，不会依赖已关闭的数据库游标。
                record = result.single()
                return record.data() if record else {}

            # 执行写事务：Neo4j Driver 会自动打开事务、提交事务，并在异常时回滚。
            movie = session.execute_write(create_movie_tx, "Tenet", 2020, 7.3)
            print(f"创建的电影：{movie}")

    finally:
        client.close()


if __name__ == "__main__":
    # 默认运行完整可见示例，适合第一次学习时直接执行：
    # python neo4j_python_guide.py
    example_usage()

    # 进阶示例依赖 Neo4j 5.x 的全文索引和向量索引，如需学习可取消注释：
    # example_advanced_usage()

    # 演示事务用法，如需学习可取消注释：
    # example_transaction_usage()
