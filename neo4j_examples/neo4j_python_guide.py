#!/usr/bin/env python3
"""
Neo4j Python 驱动使用指南
演示如何在 Python 中执行 Neo4j 的增删改查操作
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase, Result, Session, Transaction
from typing import Optional, List, Dict, Any

# 加载环境变量
load_dotenv()

# ============================================================
# 第一部分：连接配置
# ============================================================

# 从环境变量读取配置 (推荐方式)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j123")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


class Neo4jClient:
    """Neo4j 数据库客户端封装类"""

    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PASSWORD, database: str = NEO4J_DATABASE):
        """
        初始化 Neo4j 客户端

        Args:
            uri: Neo4j 连接地址，如 bolt://localhost:7687
            user: 用户名
            password: 密码
            database: 数据库名称
        """
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
    # 第二部分：基本查询方法
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
        CREATE INDEX {label}_{property_name}_index
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


def example_transaction_usage():
    """演示事务用法"""

    client = Neo4jClient()

    try:
        with client.driver.session(database=client.database) as session:
            # 使用事务
            def create_movie_tx(tx: Transaction, title: str, year: int, rating: float) -> Result:
                query = """
                CREATE (m:Movie {title: $title, released: $year, rating: $rating})
                RETURN m
                """
                return tx.run(query, title=title, year=year, rating=rating)

            # 执行事务
            result = session.execute_write(create_movie_tx, "Tenet", 2020, 7.3)
            print(f"创建的电影：{result.single()}")

    finally:
        client.close()


if __name__ == "__main__":
    # 运行示例
    # example_usage()
    client = Neo4jClient()
    alice = client.get_person_by_name("Alice")
    print(f"  Alice: {alice}")
    # 演示事务用法
    # example_transaction_usage()
