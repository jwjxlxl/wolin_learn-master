"""rag_demo 共享配置模块

统一管理 Milvus 客户端连接和环境配置。
所有模块通过此模块获取共享的 Milvus 客户端实例。
"""

import os
from pymilvus import MilvusClient
from dotenv import load_dotenv

load_dotenv()

# Milvus 连接配置（从环境变量读取，默认本地 Docker）
MILVUS_URI = os.getenv("MILVUS_URI", "http://192.168.142.128:19530")
MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME", "default")

# 集合名称常量
DOCUMENT_CHUNKS_COLLECTION = "document_chunks"
QA_PAIRS_COLLECTION = "qa_pairs"


def get_milvus_client() -> MilvusClient:
    """获取 Milvus 客户端实例

    Returns:
        MilvusClient: 连接到配置的 Milvus 服务的客户端
    """
    return MilvusClient(uri=MILVUS_URI, db_name=MILVUS_DB_NAME)
