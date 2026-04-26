"""文件检索核心模块 - 基于文件切片向量库的检索"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import MilvusClient

# 加载环境变量
load_dotenv()

# 连接配置
SERVER_ADDR = "http://localhost:19530"

# 默认配置
DEFAULT_COLLECTION_NAME = "file_chunks"
DEFAULT_DIMENSIONS = 1536
DEFAULT_TOP_K = 5


def get_embedding_client() -> OpenAI:
    """获取阿里云 Embedding 客户端"""
    api_key = os.getenv("ALIYUN_API_KEY")
    if not api_key:
        raise ValueError("未找到 ALIYUN_API_KEY 环境变量")
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


def get_embedding(text: str, client: OpenAI = None, dimensions: int = 1536) -> List[float]:
    """获取文本的向量嵌入"""
    if client is None:
        client = get_embedding_client()
    completion = client.embeddings.create(
        model="text-embedding-v4",
        input=text,
        dimensions=dimensions,
        encoding_format="float"
    )
    return completion.data[0].embedding


def search_file_chunks(
    query: str,
    milvus_client: MilvusClient = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    top_k: int = DEFAULT_TOP_K,
    embedding_client: OpenAI = None,
    milvus_uri: str = SERVER_ADDR
) -> List[Dict[str, Any]]:
    """基于文件切片的向量检索

    根据用户的自然语言描述，在文件切片向量库中检索相关内容

    Args:
        query: 用户的自然语言查询描述
        milvus_client: Milvus 客户端实例（可选，不提供则自动创建）
        collection_name: 集合名称（默认 file_chunks）
        top_k: 返回结果数量（默认 5）
        embedding_client: Embedding 客户端实例（可选）
        milvus_uri: Milvus 连接 URI（默认 http://localhost:19530）

    Returns:
        list[dict]: 检索结果列表，每项包含:
            - chunk_id: 切片 ID
            - chunk_text: 切片内容
            - file_name: 文件名
            - file_path: 文件路径
            - chunk_index: 切片索引
            - score: 相似度得分
    """
    # 初始化客户端
    if milvus_client is None:
        milvus_client = MilvusClient(uri=milvus_uri)

    if embedding_client is None:
        embedding_client = get_embedding_client()

    # 获取查询向量
    query_embedding = get_embedding(query, embedding_client)

    # 执行向量检索
    search_results = milvus_client.search(
        collection_name=collection_name,
        data=[query_embedding],
        limit=top_k,
        anns_field="dense_embedding",
        output_fields=["chunk_id", "chunk_text", "file_name", "file_path", "chunk_index"]
    )

    # 格式化结果
    results = []
    if search_results and len(search_results) > 0:
        for hit in search_results[0]:
            entity = hit.get("entity", {})
            results.append({
                "chunk_id": entity.get("chunk_id", ""),
                "chunk_text": entity.get("chunk_text", ""),
                "file_name": entity.get("file_name", ""),
                "file_path": entity.get("file_path", ""),
                "chunk_index": entity.get("chunk_index", 0),
                "score": hit.get("distance", 0)
            })

    return results


# =============================================================================
# 示例用法
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("文件检索系统测试")
    print("=" * 60)

    # 测试查询
    test_queries = [
        "桃园三结义是哪三个人？",
        "赤壁之战谁赢了？"
    ]

    for query in test_queries:
        print(f"\n查询：{query}")
        print("-" * 60)

        results = search_file_chunks(query, top_k=3)

        for i, item in enumerate(results, 1):
            print(f"\n[{i}] 得分：{item['score']:.4f}")
            print(f"    文件：{item['file_name']}")
            print(f"    内容：{item['chunk_text'][:100]}...")
