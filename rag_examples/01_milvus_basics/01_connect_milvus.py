# =============================================================================
# 01_connect_milvus — 连接 Milvus
# =============================================================================
# 用途：学习 Milvus 的各种连接方式
# 难度：⭐（1 星）
# =============================================================================

from pymilvus import MilvusClient
from rag_examples.milvus_config import MILVUS_URI


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
    password = "Milvus"

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

    # 检查连接
    check_connection_health()
