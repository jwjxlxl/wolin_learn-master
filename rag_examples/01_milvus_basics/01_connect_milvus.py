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
    print(f"\n-- 示例 0-A: 列出所有数据库")
    client = MilvusClient(uri=MILVUS_URI)
    databases = client.list_databases()
    print(f"当前所有数据库：{databases}")
    print(f"配置文件中的 MILVUS_DB_NAME：{MILVUS_DB_NAME}")

    # ── 0-B：创建新数据库 ──
    print(f"\n-- 示例 0-B: 创建新数据库")
    # 如果演示库已存在则先清理（避免上次运行残留）
    if DEMO_DB in databases:
        print(f"数据库 '{DEMO_DB}' 已存在，先删除旧版本...")
        client.drop_database(DEMO_DB)
    client.create_database(DEMO_DB)
    print(f"✓ 数据库 '{DEMO_DB}' 创建成功")
    databases = client.list_databases()
    print(f"更新后的所有数据库：{databases}")

    # ── 0-C：连接指定数据库 ──
    print(f"\n-- 示例 0-C: 连接指定数据库")
    # 方式：创建客户端时传入 db_name 参数
    demo_client = MilvusClient(uri=MILVUS_URI, db_name=DEMO_DB)
    # 验证当前连接到了正确的数据库
    collections = demo_client.list_collections()
    print(f"已连接到数据库 '{DEMO_DB}'")
    print(f"当前数据库内的集合：{collections}（应为空列表）")

    print(f"\n💡 两种指定数据库的方式：")
    print(f"   1. 创建客户端时指定：MilvusClient(uri=..., db_name='my_db')")
    print(f"   2. 使用 use_database() 切换：client.use_database('my_db')")
    print(f"   如果都不指定，默认使用 'default' 数据库。")

    # ── 0-D：清理演示数据库 ──
    print(f"\n-- 示例 0-D: 清理演示数据库")
    demo_client.drop_database(DEMO_DB)
    print(f"✓ 演示数据库 '{DEMO_DB}' 已删除")
    databases = client.list_databases()
    print(f"当前所有数据库：{databases}")

    print(f"\n{'─' * 50}")
    print(f"💡 小结：创建 Collection 前，务必确认已连接到正确的数据库。")
    print(f"   如果在不存在的数据库中操作 Collection，会触发异常。")


def connect_local_milvus():
    """连接本地 Milvus 服务（Milvus Lite）

    ⚠️ 注意：Milvus Lite 不支持 Windows。Windows 用户请使用 Docker 方式，
    参见 MILVUS_CONFIG.md。
    """
    uri = "milvus_demo.db"
    print(f"正在连接到 {uri}...")

    client = MilvusClient(uri=uri)

    version = client.get_server_version()
    print(f"Milvus 版本：{version}")
    print("✓ 连接成功！")

    return client


def connect_remote_milvus():
    """连接远程 Milvus 服务

    使用 milvus_config.py 中配置的 MILVUS_URI（从环境变量读取）。
    """
    print(f"正在连接到 {MILVUS_URI}...")

    client = MilvusClient(uri=MILVUS_URI)

    version = client.get_server_version()
    print(f"Milvus 版本：{version}")
    print("✓ 连接成功！")

    return client


def connect_with_auth():
    """带用户名密码认证的连接

    适用于生产环境，需要权限控制的场景。
    """
    host = "localhost"
    port = "19530"
    uri = f"http://{host}:{port}"

    user = "root"
    password = "Milvus"  # Milvus 默认密码，生产环境应通过环境变量读取

    print(f"正在连接到 {uri}...")
    print(f"用户：{user}")

    client = MilvusClient(uri=uri, user=user, password=password)

    print("✓ 连接成功！")

    return client


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


def check_connection_health():
    """检查 Milvus 连接健康状态"""
    client = MilvusClient(uri=MILVUS_URI)

    checks = {
        "版本检查": lambda: client.get_server_version(),
        "连接检查": lambda: client.list_collections() is not None,
    }

    print("执行健康检查...\n")

    for name, check_func in checks.items():
        try:
            result = check_func()
            print(f"[OK] {name}: {result}")
        except Exception as e:
            print(f"[ERROR] {name}: {e}")

    print("\n[OK] 连接状态正常")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("  连接 Milvus")
    print("=" * 60)
    print(f"当前 Milvus URI: {MILVUS_URI}")
    print()

    # 示例 0：数据库级别操作（创建/选择/删除数据库）
    demo_database_operations()

    # 检查连接
    check_connection_health()
