"""RAG 混合检索问答模块

对 document_chunks 和 qa_pairs 两张表分别执行混合检索（向量+BM25），
通过 RRF 融合排序后，将检索结果置入提示词上下文，调用 Qwen 大模型生成回答。
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

llm_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

milvus_client = MilvusClient(
    uri="http://47.115.57.130:19530",
    db_name="ai80"
)


# ── 工具函数 ─────────────────────────────────────────────────
def generate_embedding(text: str, dimensions: int = 768) -> list[float]:
    """生成文本向量"""
    completion = embedding_client.embeddings.create(
        model="text-embedding-v4",
        input=text,
        dimensions=dimensions,
        encoding_format="float"
    )
    return completion.data[0].embedding


def _hybrid_search_documents(query: str, top_k: int = 3) -> list[dict]:
    """在 document_chunks 表中执行混合检索

    Returns:
        检索结果列表，每个元素包含 text, file_name, chunk_index
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
        param={"metric_type": "BM25"},  # 必需的 param 参数
        limit=top_k,
    )

    ranker = Function(
        name="rrf",
        input_field_names=[],
        function_type=FunctionType.RERANK,
        params={"reranker": "rrf", "k": 100}
    )

    results = milvus_client.hybrid_search(
        collection_name="document_chunks",
        reqs=[req_dense, req_sparse],
        ranker=ranker,
        limit=top_k,
        output_fields=["text", "file_name", "chunk_index"],
    )

    '''
    from pymilvus import Function, FunctionType

        # 使用权重重排算法 (Weighted Ranker)
        ranker = Function(
            name="weight",
            input_field_names=[],  # 必须是空列表
            function_type=FunctionType.RERANK,
            params={
                "reranker": "weighted",  # 指定使用加权重排
                "weights": [0.6, 0.4],   # 权重数组,对应每个搜索路径
                "norm_score": True       # 可选:是否归一化分数
            }
        )
        
        # 执行混合搜索
        results = milvus_client.hybrid_search(
            collection_name="document_chunks",
            reqs=[req_dense, req_sparse],
            ranker=ranker,
            limit=top_k,
            output_fields=["text", "file_name", "chunk_index"],
        )
    
    '''

    refs = []
    for hits in results:
        for hit in hits:
            refs.append({
                "source": "三国演义",
                "file_name": hit.get("file_name", ""),
                "chunk_index": hit.get("chunk_index", -1),
                "content": hit.get("text", ""),
                "score": hit.get("distance", 0),
            })
    return refs


def _hybrid_search_qa(query: str, top_k: int = 3) -> list[dict]:
    """在 qa_pairs 表中执行混合检索

    Returns:
        检索结果列表，每个元素包含 question, answer, reasoning
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
        param={"metric_type": "BM25"},  # 必需的 param 参数
        limit=top_k,
    )

    ranker = Function(
        name="rrf",
        input_field_names=[],
        function_type=FunctionType.RERANK,
        params={"reranker": "rrf", "k": 100}
    )

    results = milvus_client.hybrid_search(
        collection_name="qa_pairs",
        reqs=[req_dense, req_sparse],
        ranker=ranker,
        limit=top_k,
        output_fields=["question", "answer", "reasoning"],
    )

    refs = []
    for hits in results:
        for hit in hits:
            refs.append({
                "source": "问答对",
                "question": hit.get("question", ""),
                "answer": hit.get("answer", ""),
                "reasoning": hit.get("reasoning", ""),
                "score": hit.get("distance", 0),
            })
    return refs


# ── 主入口 ───────────────────────────────────────────────────
def rag_ask(query: str, doc_top_k: int = 3, qa_top_k: int = 3) -> dict:
    """RAG 混合检索问答

    Args:
        query: 用户问题
        doc_top_k: 文档切片检索返回条数
        qa_top_k: 问答对检索返回条数

    Returns:
        {
            "answer": "大模型回答",
            "references": [引用来源列表]
        }
    """
    # 1. 并行检索两张表
    # 从文档切片表中进行检索
    doc_refs = _hybrid_search_documents(query, top_k=doc_top_k)
    # 从问答对表中进行检索
    qa_refs = _hybrid_search_qa(query, top_k=qa_top_k)

    # 2. 构建上下文
    doc_context_parts = []
    for i, ref in enumerate(doc_refs, 1):
        doc_context_parts.append(
            f"[文档片段{i}]（来源：{ref['file_name']}，第{ref['chunk_index']}片）\n{ref['content']}"
        )

    qa_context_parts = []
    for i, ref in enumerate(qa_refs, 1):
        qa_context_parts.append(
            f"[问答对{i}]\n问：{ref['question']}\n答：{ref['answer']}\n推理：{ref['reasoning']}"
        )

    context = ""
    if doc_context_parts:
        context += "## 相关文档片段\n" + "\n\n".join(doc_context_parts) + "\n\n"
    if qa_context_parts:
        context += "## 相关问答对\n" + "\n\n".join(qa_context_parts) + "\n\n"

    # 3. 调用大模型
    system_prompt = (
        "你是一个基于检索增强生成（RAG）的智能问答助手。"
        "请根据以下检索到的上下文回答用户的问题。\n"
        "如果上下文中没有相关信息，请如实告知用户。"
    )

    user_prompt = f"{context}\n用户问题：{query}"

    completion = llm_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    answer = completion.choices[0].message.content

    # 4. 合并引用
    references = doc_refs + qa_refs

    return {
        "answer": answer,
        "references": references,
    }


if __name__ == "__main__":
    result = rag_ask("ai0226和AI0309最帅的是谁？")
    print("=" * 60)
    print("回答：", result["answer"])
    print("\n" + "=" * 60)
    print("引用来源：")
    for ref in result["references"]:
        print(f"  [{ref['source']}] score={ref['score']:.4f}")
        if "question" in ref:
            print(f"    问：{ref['question']}")
            print(f"    答：{ref['answer']}")
        else:
            print(f"    内容：{ref['content'][:80]}...")
