# =============================================================================
# Milvus 连接配置
# =============================================================================
#  
# 用途：统一管理 Milvus 连接配置
#
# 说明：
#   - 优先使用 Docker Milvus（已运行在 localhost:19530）
#   - 如果没有 Docker Milvus，可修改为本地文件模式
# =============================================================================

# # 设置 UTF-8 编码（Windows 专用）
# import sys
# import io
# sys.stdout = io.TextIOWrapper(
#     sys.stdout.buffer,
#     encoding='utf-8',
#     errors='replace',
#     line_buffering=True
# )

# Milvus 连接 URI
# 方案 1: Docker Milvus（推荐，已运行）
MILVUS_URI = "http://192.168.142.128:19530"

# 方案 2: Milvus Lite（本地文件模式，需要 pip install milvus-lite）
# MILVUS_URI = "milvus_demo.db"

# 默认集合名称
DEFAULT_COLLECTION_NAME = "rag_demo"

# 默认 Embedding 维度
DEFAULT_DIMENSION = 768

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
