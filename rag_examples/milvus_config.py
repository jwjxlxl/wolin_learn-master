# =============================================================================
# Milvus 连接配置
# =============================================================================
#
# 用途：统一管理 Milvus 连接配置
#
# 说明：
#   - 通过环境变量 MILVUS_URI 和 MILVUS_DB_NAME 配置连接
#   - 默认连接本地 Docker Milvus（localhost:19530）
#   - 使用前请复制 .env.example 为 .env 并填写配置
# =============================================================================

import os
from dotenv import load_dotenv

load_dotenv()

# Milvus 连接 URI（从环境变量读取，默认本地 Docker）
# 本地 Docker：http://localhost:19530
# Milvus Lite（不支持 Windows）：milvus_demo.db
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")

# Milvus 数据库名
MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME", "default")

# 默认集合名称
DEFAULT_COLLECTION_NAME = "rag_demo"

# 默认 Embedding 维度（text-embedding-v4 输出 1024 维）
DEFAULT_DIMENSION = 1024

# 默认度量类型
DEFAULT_METRIC_TYPE = "COSINE"


def get_milvus_client():
    """
    获取 Milvus 客户端

    返回：
        MilvusClient 实例
    """
    from pymilvus import MilvusClient
    return MilvusClient(uri=MILVUS_URI)


def check_connection():
    """
    检查 Milvus 连接是否正常

    返回：
        bool: 连接是否正常
    """
    try:
        client = get_milvus_client()
        version = client.get_server_version()
        print(f"[OK] Milvus 连接正常，版本：{version}")
        return True
    except Exception as e:
        print(f"[ERROR] Milvus 连接失败：{e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Milvus 连接配置检查")
    print("=" * 60)
    print(f"当前配置：MILVUS_URI = {MILVUS_URI}")
    print()
    check_connection()
