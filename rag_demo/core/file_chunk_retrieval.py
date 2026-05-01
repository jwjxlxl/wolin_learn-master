"""文件切片检索模块

根据用户提出的问题对文件切片表进行混合检索（向量 + BM25），
返回检索出的原始文本片段列表。
"""

from openai import OpenAI
import os
from pymilvus import AnnSearchRequest, Function, FunctionType, MilvusClient
from dotenv import load_dotenv

load_dotenv()

# ── 客户端初始化 ──────────────────────────────────────────────
embedding_client = OpenAI(
    api_key=os.getenv("ALIYUN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

milvus_client = MilvusClient(
    uri="http://47.115.57.130:19530",
    db_name="ai80"
)

COLLECTION_NAME = "document_chunks"
DEFAULT_TOP_K = 5


def generate_embedding(text: str, dimensions: int = 768) -> list[float]:
    """生成文本向量"""
    completion = embedding_client.embeddings.create(
        model="text-embedding-v4",
        input=text,
        dimensions=dimensions,
        encoding_format="float"
    )
    return completion.data[0].embedding


def search_file_chunks(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """根据用户问题检索文件切片表中的原始文本片段

    Args:
        query: 用户的自然语言问题
        top_k: 返回结果数量，默认 5

    Returns:
        list[dict]: 检索出的原始文本片段列表，每项包含:
            - text: 切片文本内容
            - file_name: 文件名
            - chunk_index: 切片在文件中的索引
            - score: 混合检索相似度得分
    """
    query_vector = generate_embedding(query)

    req_dense = AnnSearchRequest(
        data=[query_vector],
        anns_field="dense_vector",
        param={"nprobe": 10},
        limit=top_k,
    )

    req_sparse = AnnSearchRequest(
        data=[query],
        anns_field="sparse_vector",
        param={"metric_type": "BM25"},
        limit=top_k,
    )

    ranker = Function(
        name="rrf",
        input_field_names=[],
        function_type=FunctionType.RERANK,
        params={"reranker": "rrf", "k": 100}
    )

    results = milvus_client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=[req_dense, req_sparse],
        ranker=ranker,
        limit=top_k,
        output_fields=["text", "file_name", "chunk_index"],
    )

    chunks = []
    for hits in results:
        for hit in hits:
            chunks.append({
                "text": hit.get("text", ""),
                "file_name": hit.get("file_name", ""),
                "chunk_index": hit.get("chunk_index", -1),
                "score": hit.get("distance", 0),
            })

    return chunks


if __name__ == "__main__":
    test_queries = ["桃园三结义是哪三个人？", "赤壁之战的经过是怎样的？"]

    for query in test_queries:
        print(f"\n问题：{query}")
        print("-" * 60)
        results = search_file_chunks(query, top_k=3)
        for i, chunk in enumerate(results, 1):
            print(f"\n[{i}] 得分：{chunk['score']:.4f}")
            print(f"    文件：{chunk['file_name']}")
            print(f"    内容：{chunk['text'][:100]}...")
