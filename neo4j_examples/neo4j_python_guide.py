"""
Neo4j Python 操作指南

演示如何在 Python 中连接 Neo4j 图数据库，并把常见图数据操作封装成一个客户端类：
  - 连接与断开、连通性测试
  - 节点 / 关系的增删改查（CRUD）
  - MERGE 查找或创建（幂等写入）
  - CSV 批量导入
  - 沿图关系扩展上下文（GraphRAG 雏形）

运行前准备：
  1. 在项目根目录执行：pip install -r requirements.txt（会安装 neo4j、python-dotenv）
  2. 在项目根目录 .env 中配置 Neo4j 连接信息：
       NEO4J_URI=neo4j://your-server-ip:7687
       NEO4J_USER=neo4j
       NEO4J_PASSWORD=your_neo4j_password
       NEO4J_DATABASE=neo4j
  3. 直接运行：python neo4j_python_guide.py
"""

import csv
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

# 示例数据文件：相对本脚本所在目录
MOVIES_CSV_PATH = Path(__file__).parent / "data" / "movies.csv"


class Neo4jClient:
    """Neo4j 图数据库客户端封装

    统一管理驱动连接与会话，对外提供节点、关系的常见操作，
    避免在业务代码中重复书写连接、会话管理和 Cypher 语句。

    Args:
        uri: 连接地址，默认从环境变量 NEO4J_URI 读取
        user: 用户名，默认从环境变量 NEO4J_USER 读取
        password: 密码，默认从环境变量 NEO4J_PASSWORD 读取
        database: 数据库名，默认从环境变量 NEO4J_DATABASE 读取
    """

    def __init__(self, uri=None, user=None, password=None, database=None):
        # 加载项目根目录的 .env，读取 Neo4j 连接配置
        load_dotenv()
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def __enter__(self):
        """支持 with Neo4jClient() as client: 写法"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """关闭驱动，释放连接资源"""
        self.driver.close()

    # ------------------------------------------------------------------
    # 底层通用方法
    # ------------------------------------------------------------------

    def run_query(self, query, params=None):
        """执行读查询，返回记录列表（每行是一个 dict）"""
        with self.driver.session(database=self.database) as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]

    def run_write(self, query, params=None):
        """执行写操作（CREATE / MERGE / SET / DELETE），返回写入摘要"""
        with self.driver.session(database=self.database) as session:
            result = session.run(query, params or {})
            return result.consume()

    def verify_connection(self):
        """测试连接是否可用，返回 True / False"""
        rows = self.run_query("RETURN 1 AS ok")
        return bool(rows) and rows[0]["ok"] == 1

    # ------------------------------------------------------------------
    # 通用节点操作
    # ------------------------------------------------------------------

    def create_node(self, label, properties):
        """创建节点：label 为标签名，properties 为属性字典"""
        items = [f"{key}: ${key}" for key in properties]
        query = f"CREATE (n:{label} {{{', '.join(items)}}}) RETURN n"
        return self.run_query(query, properties)

    def get_nodes(self, label, properties=None):
        """查询节点：按标签查询，可传入属性做等值过滤"""
        query = f"MATCH (n:{label})"
        if properties:
            conditions = [f"n.{key} = ${key}" for key in properties]
            query += " WHERE " + " AND ".join(conditions)
        query += " RETURN n"
        return self.run_query(query, properties or {})

    def update_node(self, label, match_props, set_props):
        """更新节点：按 match_props 匹配，用 set_props 更新属性"""
        conditions = [f"n.{key} = $match_{key}" for key in match_props]
        sets = [f"n.{key} = $set_{key}" for key in set_props]
        query = (
            f"MATCH (n:{label}) WHERE {' AND '.join(conditions)} "
            f"SET {', '.join(sets)} RETURN n"
        )
        params = {f"match_{k}": v for k, v in match_props.items()}
        params.update({f"set_{k}": v for k, v in set_props.items()})
        return self.run_query(query, params)

    def delete_node(self, label, properties):
        """删除节点：DETACH DELETE 会先断开该节点的所有关系再删除"""
        items = [f"{key}: ${key}" for key in properties]
        query = f"MATCH (n:{label} {{{', '.join(items)}}}) DETACH DELETE n"
        return self.run_write(query, properties)

    def merge_node(self, label, properties, on_create=None, on_match=None):
        """MERGE 节点：按 properties 查找，不存在则创建（幂等写入）

        on_create: 仅新建时写入的属性；on_match: 仅已存在时写入的属性。
        """
        items = [f"{key}: ${key}" for key in properties]
        query = f"MERGE (n:{label} {{{', '.join(items)}}})"
        params = dict(properties)
        if on_create:
            sets = [f"n.{key} = $oc_{key}" for key in on_create]
            query += " ON CREATE SET " + ", ".join(sets)
            params.update({f"oc_{k}": v for k, v in on_create.items()})
        if on_match:
            sets = [f"n.{key} = $om_{key}" for key in on_match]
            query += " ON MATCH SET " + ", ".join(sets)
            params.update({f"om_{k}": v for k, v in on_match.items()})
        query += " RETURN n"
        return self.run_query(query, params)

    # ------------------------------------------------------------------
    # 通用关系操作
    # ------------------------------------------------------------------

    def create_relationship(self, start_label, start_props, rel_type, end_label, end_props, rel_props=None):
        """在两个已存在节点之间创建关系

        Args:
            start_label / end_label: 起点 / 终点节点标签
            start_props / end_props: 定位起点 / 终点节点的属性字典
            rel_type: 关系类型（如 ACTED_IN）
            rel_props: 关系上的属性（可选）
        """
        start_items = [f"{key}: $s_{key}" for key in start_props]
        end_items = [f"{key}: $e_{key}" for key in end_props]
        rel_braces = ""
        params = {f"s_{k}": v for k, v in start_props.items()}
        params.update({f"e_{k}": v for k, v in end_props.items()})
        if rel_props:
            rel_items = [f"{key}: $r_{key}" for key in rel_props]
            rel_braces = " {" + ", ".join(rel_items) + "}"
            params.update({f"r_{k}": v for k, v in rel_props.items()})
        query = (
            f"MATCH (a:{start_label} {{{', '.join(start_items)}}}), "
            f"(b:{end_label} {{{', '.join(end_items)}}}) "
            f"CREATE (a)-[:{rel_type}{rel_braces}]->(b) RETURN a, b"
        )
        return self.run_query(query, params)

    def delete_relationship(self, start_label, start_props, rel_type, end_label, end_props):
        """删除两个节点之间指定类型的关系"""
        start_items = [f"{key}: $s_{key}" for key in start_props]
        end_items = [f"{key}: $e_{key}" for key in end_props]
        params = {f"s_{k}": v for k, v in start_props.items()}
        params.update({f"e_{k}": v for k, v in end_props.items()})
        query = (
            f"MATCH (a:{start_label} {{{', '.join(start_items)}}})"
            f"-[r:{rel_type}]->"
            f"(b:{end_label} {{{', '.join(end_items)}}}) DELETE r"
        )
        return self.run_write(query, params)

    # ------------------------------------------------------------------
    # 领域方法：电影知识图谱（教学演示数据）
    # ------------------------------------------------------------------

    def create_person(self, name, age):
        """创建或更新演员节点（MERGE 幂等）"""
        query = "MERGE (p:Person {name: $name}) SET p.age = $age RETURN p"
        return self.run_query(query, {"name": name, "age": age})

    def create_director(self, name, birth_year):
        """创建或更新导演节点"""
        query = "MERGE (d:Director {name: $name}) SET d.birthYear = $birth_year RETURN d"
        return self.run_query(query, {"name": name, "birth_year": birth_year})

    def create_movie(self, title, released, rating):
        """创建或更新电影节点"""
        query = (
            "MERGE (m:Movie {title: $title}) "
            "SET m.released = $released, m.rating = $rating RETURN m"
        )
        return self.run_query(query, {"title": title, "released": released, "rating": rating})

    def create_company(self, name, founded):
        """创建或更新公司节点"""
        query = "MERGE (c:Company {name: $name}) SET c.founded = $founded RETURN c"
        return self.run_query(query, {"name": name, "founded": founded})

    def create_acted_in(self, actor_name, movie_title, roles=None):
        """创建演员出演电影的关系"""
        query = (
            "MATCH (p:Person {name: $actor_name}), (m:Movie {title: $movie_title}) "
            "MERGE (p)-[r:ACTED_IN]->(m) SET r.roles = $roles RETURN p, m"
        )
        return self.run_query(query, {
            "actor_name": actor_name, "movie_title": movie_title, "roles": roles or []
        })

    def create_directed(self, director_name, movie_title, year=None):
        """创建导演执导电影的关系"""
        query = (
            "MATCH (d:Director {name: $director_name}), (m:Movie {title: $movie_title}) "
            "MERGE (d)-[r:DIRECTED]->(m) SET r.year = $year RETURN d, m"
        )
        return self.run_query(query, {
            "director_name": director_name, "movie_title": movie_title, "year": year
        })

    def create_produced_by(self, movie_title, company_name):
        """创建电影由公司制作的关系"""
        query = (
            "MATCH (m:Movie {title: $movie_title}), (c:Company {name: $company_name}) "
            "MERGE (m)-[r:PRODUCED_BY]->(c) RETURN m, c"
        )
        return self.run_query(query, {"movie_title": movie_title, "company_name": company_name})

    # ------------------------------------------------------------------
    # 领域方法：查询 / 统计
    # ------------------------------------------------------------------

    def get_all_persons(self):
        """查询所有演员"""
        query = "MATCH (p:Person) RETURN p.name AS name, p.age AS age ORDER BY p.name"
        return self.run_query(query)

    def get_movies_by_rating_range(self, min_rating, max_rating):
        """按评分范围查询电影"""
        query = (
            "MATCH (m:Movie) WHERE m.rating >= $min_rating AND m.rating <= $max_rating "
            "RETURN m.title AS title, m.released AS released, m.rating AS rating "
            "ORDER BY m.rating DESC"
        )
        return self.run_query(query, {"min_rating": min_rating, "max_rating": max_rating})

    def get_person_movies(self, actor_name):
        """查询某位演员出演过的电影及角色"""
        query = (
            "MATCH (p:Person {name: $actor_name})-[r:ACTED_IN]->(m:Movie) "
            "RETURN m.title AS title, r.roles AS roles"
        )
        return self.run_query(query, {"actor_name": actor_name})

    def get_collaborations(self):
        """查询合作过的演员（共同出演同一部电影）"""
        query = (
            "MATCH (p1:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(p2:Person) "
            "WHERE p1.name < p2.name "
            "RETURN p1.name AS actor1, p2.name AS actor2, m.title AS movie ORDER BY p1.name"
        )
        return self.run_query(query)

    def get_director_stats(self):
        """统计每个导演的作品数量与平均评分"""
        query = (
            "MATCH (d:Director)-[:DIRECTED]->(m:Movie) "
            "RETURN d.name AS director, count(m) AS movie_count, "
            "round(avg(m.rating) * 10) / 10.0 AS avg_rating ORDER BY avg_rating DESC"
        )
        return self.run_query(query)

    def expand_context(self, movie_title):
        """从一部电影出发，沿关系扩展导演、演员、公司（GraphRAG 上下文扩展雏形）"""
        query = (
            "MATCH (movie:Movie {title: $title}) "
            "OPTIONAL MATCH (movie)<-[:DIRECTED]-(director:Director) "
            "OPTIONAL MATCH (actor:Person)-[:ACTED_IN]->(movie) "
            "OPTIONAL MATCH (movie)-[:PRODUCED_BY]->(company:Company) "
            "RETURN movie.title AS movie, director.name AS director, "
            "collect(DISTINCT actor.name) AS actors, company.name AS company"
        )
        return self.run_query(query, {"title": movie_title})

    # ------------------------------------------------------------------
    # 更新 / 删除 / 批量导入
    # ------------------------------------------------------------------

    def update_person_age(self, name, new_age):
        """更新演员年龄"""
        query = "MATCH (p:Person {name: $name}) SET p.age = $new_age RETURN p"
        return self.run_query(query, {"name": name, "new_age": new_age})

    def delete_person(self, name):
        """删除演员（同时断开其所有关系）"""
        query = "MATCH (p:Person {name: $name}) DETACH DELETE p"
        return self.run_write(query, {"name": name})

    def load_movies_from_csv(self, csv_path=None):
        """从 CSV 批量导入电影（支持 UTF-8 BOM）"""
        csv_path = csv_path or MOVIES_CSV_PATH
        with open(csv_path, encoding="utf-8-sig") as f:
            rows = [
                {"title": row["title"], "released": int(row["released"]), "rating": float(row["rating"])}
                for row in csv.DictReader(f)
            ]
        query = (
            "UNWIND $rows AS row "
            "MERGE (m:Movie {title: row.title}) "
            "SET m.released = row.released, m.rating = row.rating "
            "RETURN count(m) AS imported"
        )
        return self.run_query(query, {"rows": rows})

    def clear_all(self):
        """清空数据库所有节点和关系（谨慎使用）"""
        return self.run_write("MATCH (n) DETACH DELETE n")


def main():
    """演示 Neo4jClient 的常见操作"""
    client = Neo4jClient()

    # 示例 1：测试连接并清空数据
    print("\n-- 示例 1: 连接测试 + 清空数据")
    if not client.verify_connection():
        print("连接失败，请检查项目根目录 .env 中的 NEO4J_* 配置")
        client.close()
        return
    print(f"连接成功：{client.uri} (database={client.database})")
    client.clear_all()
    print("已清空数据库")

    # 示例 2：创建节点和关系（电影知识图谱）
    print("\n-- 示例 2: 创建节点与关系")
    client.create_person("吴京", 51)
    client.create_person("沈腾", 46)
    client.create_person("张小斐", 39)
    client.create_director("郭帆", 1980)
    client.create_director("贾玲", 1982)
    client.create_movie("流浪地球", 2019, 7.9)
    client.create_movie("你好，李焕英", 2021, 7.7)
    client.create_company("华夏影视", 2010)
    client.create_acted_in("吴京", "流浪地球", ["刘培强"])
    client.create_acted_in("沈腾", "你好，李焕英", ["沈光林"])
    client.create_acted_in("张小斐", "你好，李焕英", ["李焕英"])
    client.create_directed("郭帆", "流浪地球", 2019)
    client.create_directed("贾玲", "你好，李焕英", 2021)
    client.create_produced_by("流浪地球", "华夏影视")
    print("已创建 演员 / 导演 / 电影 / 公司 节点及关系")

    # 示例 3：查询
    print("\n-- 示例 3: 查询")
    print("全部演员:", client.get_all_persons())
    print("评分 >= 7.7 的电影:", client.get_movies_by_rating_range(7.7, 10))
    print("吴京出演的电影:", client.get_person_movies("吴京"))
    print("合作过的演员:", client.get_collaborations())
    print("导演统计:", client.get_director_stats())

    # 示例 4：更新与删除
    print("\n-- 示例 4: 更新与删除")
    client.update_person_age("吴京", 52)
    print("更新吴京年龄后:", client.get_all_persons())
    client.delete_person("张小斐")
    print("删除张小斐后:", client.get_all_persons())

    # 示例 5：CSV 批量导入 + 沿图关系扩展上下文
    print("\n-- 示例 5: CSV 批量导入 + 上下文扩展")
    print("批量导入电影数:", client.load_movies_from_csv())
    print("扩展《流浪地球》上下文:", client.expand_context("流浪地球"))

    client.close()
    print("\n连接已关闭")


if __name__ == "__main__":
    main()
