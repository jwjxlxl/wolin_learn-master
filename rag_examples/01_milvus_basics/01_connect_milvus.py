# =============================================================================
# 01_connect_milvus — 连接 Milvus
# =============================================================================
# 用途：学习 Milvus 的各种连接方式
# 难度：⭐（1 星）
# =============================================================================

from pymilvus import MilvusClient
from rag_examples.milvus_config import MILVUS_URI


def demo_database_operations():
    """演示数据库级别操作

    Milvus 支持多数据库（Multi-Database），类似于 MySQL 中的数据库概念。
    每个数据库可以有自己独立的 Collection，不同数据库之间的数据完全隔离。

    本示例演示：
    - 列出所有数据库
    - 创建新数据库
    - 连接指定数据库
    - 删除数据库

    ⚠️ 注意：数据库操作仅在 Milvus 2.3+ 的 Docker/远程部署版本中支持，
    Milvus Lite 不支持多数据库功能。
    """
    from pymilvus import MilvusClient
    from rag_examples.milvus_config import MILVUS_URI, MILVUS_DB_NAME

    DEMO_DB = "wolin_learn_demo"  # 演示用的临时数据库名

    # ── 0-A：列出所有数据库 ──
    client = MilvusClient(uri=MILVUS_URI)
    databases = client.list_databases()
    print(f"当前所有数据库：{databases}")

    # ── 0-B：创建新数据库 ──
    # 如果演示库已存在则先清理（避免上次运行残留）
    if DEMO_DB in databases:
        client.drop_database(DEMO_DB)
    client.create_database(DEMO_DB)
    databases = client.list_databases()

    # ── 0-C：连接指定数据库 ──
    # 方式：创建客户端时传入 db_name 参数
    demo_client = MilvusClient(uri=MILVUS_URI, db_name=DEMO_DB)
    # 验证当前连接到了正确的数据库
    collections = demo_client.list_collections()
    print(f"当前数据库内的集合：{collections}（应为空列表）")



def connection_parameters_explained():
    """连接参数详解

    展示 Milvus 连接的各种 URI 格式和参数说明。
    """
    print("=" * 60)
    print("Milvus 连接参数详解")
    print("=" * 60)

    print("""
┌─────────────────────────────────────────────────────────┐
│ 连接参数说明                                             │
├──────────────┬──────────────────────────────────────────┤
│    参数       │    说明                                  │
├──────────────┼──────────────────────────────────────────┤
│ uri          │ 服务地址，默认 localhost:19530            │
│ user         │ 用户名（可选）                            │
│ password     │ 密码（可选）                              │
│ token        │ API Token（云服务场景）                   │
│ db_name      │ 数据库名称，默认 default                  │
└──────────────┴──────────────────────────────────────────┘

URI 格式说明：

1. 本地文件（Milvus Lite，⚠️ 不支持 Windows）
   uri = "milvus_demo.db"

2. HTTP 连接（Docker 或远程服务）
   uri = "http://localhost:19530"

3. HTTPS 连接（云服务，如 Zilliz Cloud）
   uri = "https://your-cluster.zillizcloud.com"

4. 完整格式（带认证）
   uri = "http://user:password@localhost:19530/dbname"
""")


if __name__ == "__main__":

    # 示例 0：数据库级别操作（创建/选择/删除数据库）
    demo_database_operations()

