"""RAG 问答核心模块 - 基于问答对向量库的检索增强生成"""

import os
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import MilvusClient, AnnSearchRequest, Function, FunctionType

# 加载环境变量
load_dotenv()

# 连接配置
SERVER_ADDR = "http://localhost:19530"

# 默认配置
DEFAULT_COLLECTION_NAME = "qa_pairs"
DEFAULT_DIMENSIONS = 1536
DEFAULT_TOP_K = 5
DEFAULT_LLM_MODEL = "qwen-plus"


def get_embedding_client() -> OpenAI:
    """获取阿里云 Embedding 客户端"""
    api_key = os.getenv("ALIYUN_API_KEY")
    if not api_key:
        raise ValueError("未找到 ALIYUN_API_KEY 环境变量")
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


def get_llm_client() -> OpenAI:
    """获取阿里云 LLM 客户端"""
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


def search_qa_pairs(
    query: str,
    milvus_client: MilvusClient,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    top_k: int = DEFAULT_TOP_K,
    embedding_client: OpenAI = None
) -> List[Dict[str, Any]]:
    """在问答对向量库中检索相关内容 - 混合检索（语义 + 关键字）

    Args:
        query: 查询问题
        milvus_client: Milvus 客户端实例
        collection_name: 集合名称
        top_k: 返回结果数量
        embedding_client: Embedding 客户端实例

    Returns:
        检索到的问答对列表，每项包含:
        - question: 问题
        - answer: 答案
        - reasoning: 思考过程
        - score: 相似度分数
    """
    # 获取查询向量（语义检索用）
    query_embedding = get_embedding(query, embedding_client)

    # 构建语义检索请求（dense vector）
    dense_search_request = AnnSearchRequest(
        data=[query_embedding],
        anns_field="question_dense_embedding",
        param={"nprobe": 10},
        limit=top_k
    )

    # 构建关键字检索请求（BM25 sparse）
    sparse_search_request = AnnSearchRequest(
        data=[query],
        anns_field="question_sparse_embedding",
        param={"analyzer": "standard"},
        limit=top_k
    )

    # 执行混合检索
    reqs = [dense_search_request, sparse_search_request]

    # 使用 RRF 重排序
    ranker = Function(
        name="rrf",
        input_field_names=[],
        function_type=FunctionType.RERANK,
        params={
            "reranker": "rrf",
            "k": 100  # RRF 参数
        }
    )

    # ranker = Function(
    #     name="weight",
    #     input_field_names=[],
    #     function_type=FunctionType.RERANK,
    #     params={
    #         "reranker": "weighted",
    #         "weights": [0.5, 0.5],
    #         "norm_score": True  # 可选参数
    #     }
    # )

    search_results = milvus_client.hybrid_search(
        collection_name=collection_name,
        reqs=reqs,
        ranker=ranker,
        limit=top_k,
        output_fields=["question", "answer", "reasoning"]
    )

    # 格式化结果
    results = []
    if search_results and len(search_results) > 0:
        for hit in search_results[0]:
            results.append({
                "question": hit.get("entity", {}).get("question", ""),
                "answer": hit.get("entity", {}).get("answer", ""),
                "reasoning": hit.get("entity", {}).get("reasoning", ""),
                "score": hit.get("distance", 0)
            })

    return results


def build_rag_prompt(
    question: str,
    retrieved_qas: List[Dict[str, Any]]
) -> str:
    """构建 RAG Prompt

    Args:
        question: 用户问题
        retrieved_qas: 检索到的问答对列表

    Returns:
        构建好的 Prompt 字符串
    """
    # 构建参考上下文
    context_parts = []
    for i, qa in enumerate(retrieved_qas, 1):
        context_parts.append(
            f"[参考 {i}]\n问题：{qa['question']}\n答案：{qa['answer']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""你是一个专业的智能助手，请参考以下相关问答对来回答用户的问题。

参考问答对：
{context}

用户问题：{question}

回答要求：
1. 基于上述参考问答对中的信息来组织答案
2. 如果参考内容与问题相关，优先采纳其中的答案
3. 如果参考内容不足以回答问题，可以结合你的知识进行补充
4. 回答时请标注参考来源，如"根据参考 1 中的说明..."
5. 保持回答准确、简洁、条理清晰

回答："""

    return prompt


def call_llm(
    prompt: str,
    llm_client: OpenAI = None,
    model: str = DEFAULT_LLM_MODEL
) -> str:
    """调用 LLM 生成回答

    Args:
        prompt: 输入 Prompt
        llm_client: LLM 客户端实例
        model: 模型名称

    Returns:
        LLM 生成的回答文本
    """
    if llm_client is None:
        llm_client = get_llm_client()

    response = llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个专业的智能助手，请基于参考信息回答用户问题。"},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


def qa_rag_ask(
    question: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    top_k: int = DEFAULT_TOP_K,
    llm_model: str = DEFAULT_LLM_MODEL,
    milvus_uri: str = SERVER_ADDR
) -> Tuple[str, List[Dict[str, Any]]]:
    """RAG 问答主函数 - 检索问答对向量库并生成回答

    Args:
        question: 用户问题
        collection_name: 向量库集合名称
        top_k: 检索文档数量
        llm_model: LLM 模型名称
        milvus_uri: Milvus 连接 URI

    Returns:
        tuple: (答案文本，检索结果列表)
        - 答案文本 (str): LLM 生成的回答
        - 检索结果列表 (list): 检索到的相关问答对，每项包含:
          - question: 问题
          - answer: 答案
          - reasoning: 思考过程
          - score: 相似度分数
    """
    # 初始化客户端
    milvus_client = MilvusClient(uri=milvus_uri)
    embedding_client = get_embedding_client()
    llm_client = get_llm_client()

    try:
        # 1. 检索相关问答对
        retrieved_qas = search_qa_pairs(
            query=question,
            milvus_client=milvus_client,
            collection_name=collection_name,
            top_k=top_k,
            embedding_client=embedding_client
        )

        if not retrieved_qas:
            return "未找到相关参考信息，无法回答该问题。", []

        # 2. 构建 RAG Prompt
        prompt = build_rag_prompt(question, retrieved_qas)

        # 3. 调用 LLM 生成回答
        answer = call_llm(prompt, llm_client, llm_model)

        return answer, retrieved_qas

    finally:
        # 清理资源（MilvusClient 不需要显式关闭）
        pass


# =============================================================================
# 示例用法
# =============================================================================

if __name__ == "__main__":
    # 测试问答
    test_questions = [
        "麻辣烫是谁？",
        "深度学习需要什么基础？"
    ]

    print("=" * 60)
    print("RAG 问答系统测试")
    print("=" * 60)

    for question in test_questions:
        print(f"\n用户：{question}")
        print("-" * 50)

        answer, sources = qa_rag_ask(question)

        print(f"助手：{answer}")

        if sources:
            print("\n检索到的参考:")
            for i, src in enumerate(sources[:3], 1):
                print(f"\n[参考{i}] 相似度：{src['score']:.4f}")
                print(f"  问：{src['question'][:50]}...")
                print(f"  答：{src['answer'][:50]}...")
