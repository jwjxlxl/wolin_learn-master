# =============================================================================
# 文档切片检索器 — RAG 核心模块
# =============================================================================
#
# 用途：基于文档切片表的混合检索 + LLM 回答生成
#       1. 将用户问题向量化
#       2. 在 Milvus 文档切片表中执行混合检索（稠密 + BM25 稀疏）
#       3. 将检索到的片段组装为 Prompt
#       4. 调用 Qwen LLM 生成回答
#
# 使用方法：
#   from rag_examples.sg_rag.core.doc_retriever import DocRetriever
#   retriever = DocRetriever()
#   answer = retriever.query("刘备三顾茅庐的经过是怎样的？")
# =============================================================================

import os
from dotenv import load_dotenv

from pymilvus import (
    MilvusClient,
    Function,
    FunctionType,
    AnnSearchRequest,
)

from rag_examples.milvus_config import MILVUS_URI, DEFAULT_DIMENSION

load_dotenv()

# API Key 兼容：优先 DASHSCOPE，回退 ALIYUN
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIYUN_API_KEY", "")

# 文档切片集合名
DOC_COLLECTION_NAME = os.getenv("SG_RAG_COLLECTION", "sanguo_doc_chunks")

# 默认检索数量
DEFAULT_TOP_K = 5

# Qwen 模型配置
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")

# =============================================================================
# Prompt 模板
# =============================================================================

SYSTEM_PROMPT = """你是一个基于 RAG（检索增强生成）的智能问答助手。请根据下方提供的参考资料，回答用户的问题。

要求：
1. 优先依据参考资料作答，不要凭空编造
2. 参考资料不足以回答问题时，坦诚说明并给出合理建议
3. 回答语言保持中文，简洁清晰
4. 适当引用参考资料中的关键信息
"""

USER_PROMPT_TEMPLATE = """## 参考资料

{context}

## 用户问题

{question}

请根据参考资料回答上述问题。如果参考资料包含完整答案，请据此作答；如果参考资料只有部分相关信息，请综合理解后作答。"""


# =============================================================================
# 向量化工具
# =============================================================================


def generate_embedding(text: str) -> list[float]:
    """
    调用阿里云 DashScope text-embedding-v4 生成 1024 维向量

    参数：
        text: 待向量化的文本

    返回：
        list[float]: 1024 维稠密向量
    """
    from openai import OpenAI

    if not DASHSCOPE_API_KEY:
        raise RuntimeError("未配置 DASHSCOPE_API_KEY 或 ALIYUN_API_KEY")

    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.embeddings.create(
        model="text-embedding-v4",
        input=text,
        dimensions=DEFAULT_DIMENSION,
        encoding_format="float",
    )

    return completion.data[0].embedding


# =============================================================================
# LLM 调用
# =============================================================================


def call_qwen(messages: list[dict], model: str = QWEN_MODEL) -> str:
    """
    调用 Qwen LLM 生成回答

    参数：
        messages: OpenAI 格式的 messages 列表
        model: 模型名称

    返回：
        str: LLM 生成的回答文本
    """
    from openai import OpenAI

    if not DASHSCOPE_API_KEY:
        raise RuntimeError("未配置 DASHSCOPE_API_KEY 或 ALIYUN_API_KEY")

    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
    )

    return response.choices[0].message.content


# =============================================================================
# 文档切片检索器
# =============================================================================


class DocRetriever:
    """基于 Milvus 文档切片表的 RAG 检索器

    工作流程：
        1. 将用户问题向量化
        2. 在 Milvus 中执行混合检索（稠密向量 + BM25 稀疏向量）
        3. RRF 融合排序，取 Top-K 最相似文档片段
        4. 组装 Prompt，调用 Qwen LLM 生成回答
    """

    def __init__(
        self,
        collection_name: str = DOC_COLLECTION_NAME,
        top_k: int = DEFAULT_TOP_K,
    ):
        """
        初始化检索器

        参数：
            collection_name: Milvus 文档切片集合名称
            top_k: 检索 Top-K 数量
        """
        self.collection_name = collection_name
        self.top_k = top_k
        self.client = MilvusClient(uri=MILVUS_URI)

    def search(
        self,
        query: str,
        top_k: int = None,
    ) -> list[dict]:
        """
        在文档切片表中执行混合检索

        参数：
            query: 用户问题
            top_k: 返回数量（默认使用构造时的 top_k）

        返回：
            list[dict]: 检索结果，每项包含 text、file_name、chunk_index、title 等字段
        """
        k = top_k or self.top_k

        # 1. 将问题向量化（稠密向量）
        dense_vector = generate_embedding(query)

        # 2. 构建稠密向量检索请求
        req_dense = AnnSearchRequest(
            data=[dense_vector],
            anns_field="dense_vector",
            param={"nprobe": 10},
            limit=k,
        )

        # 3. 构建 BM25 稀疏向量检索请求（直接传原始文本）
        req_sparse = AnnSearchRequest(
            data=[query],
            anns_field="sparse_vector",
            limit=k,
            param={"metric_type": "BM25"}
        )

        # 4. 构建混合检索请求列表
        reqs = [req_dense, req_sparse]

        # 5. RRF 融合排序器
        ranker = Function(
            name="rrf",
            input_field_names=[],
            function_type=FunctionType.RERANK,
            params={
                "reranker": "rrf",
                "k": 100,
            },
        )

        # 6. 执行混合检索
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=reqs,
            ranker=ranker,
            limit=k,
            output_fields=["text", "file_name", "chunk_index", "title", "category"],
        )

        # 7. 解析结果
        hits = []
        if results and len(results) > 0:
            for hit in results[0]:
                entity = hit.get("entity", {})
                hits.append({
                    "text": entity.get("text", ""),
                    "file_name": entity.get("file_name", ""),
                    "chunk_index": entity.get("chunk_index", 0),
                    "title": entity.get("title", ""),
                    "category": entity.get("category", ""),
                    "score": hit.get("distance", 0),
                })

        return hits

    def _build_context(self, hits: list[dict]) -> str:
        """
        将检索到的文档片段组装为上下文文本

        参数：
            hits: 检索结果列表

        返回：
            str: 格式化的上下文文本
        """
        if not hits:
            return "（未检索到相关参考资料）"

        context_parts = []
        for i, hit in enumerate(hits, 1):
            part = f"**参考资料 {i}**"
            if hit.get("title"):
                part += f" — {hit['title']}"
            if hit.get("file_name"):
                part += f"（来源：{hit['file_name']}）"
            part += f"\n{hit['text']}"
            context_parts.append(part)

        return "\n\n".join(context_parts)

    def query(self, question: str, top_k: int = None) -> dict:
        """
        完整 RAG 流程：检索文档片段 + LLM 生成回答

        参数：
            question: 用户问题
            top_k: 检索 Top-K 数量（可选）

        返回：
            dict: 包含 answer（回答）、hits（检索结果）、context（上下文）
        """
        # 1. 混合检索
        hits = self.search(question, top_k)

        # 2. 组装上下文
        context = self._build_context(hits)

        # 3. 构建 Prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # 4. 调用 Qwen 生成回答
        answer = call_qwen(messages)

        return {
            "answer": answer,
            "hits": hits,
            "context": context,
            "question": question,
        }


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":

    question = "张让是谁？"

    print(f"问题：{question}\n")

    if not DASHSCOPE_API_KEY:
        print("[ERROR] 未配置 DASHSCOPE_API_KEY，请在 .env 文件中设置")
        exit(1)

    retriever = DocRetriever(top_k=3)

    # 先展示检索到的文档片段
    hits = retriever.search(question)
    print(f"{'─'*50}")
    print(f"检索到 {len(hits)} 个相关片段：\n")
    for i, hit in enumerate(hits, 1):
        print(f"  [{i}] 来源：{hit['file_name']} | 第 {hit['chunk_index']} 块"
              f"{f' | {hit['title']}' if hit['title'] else ''}")
        print(f"      内容：{hit['text'][:120]}...")
        print(f"      相似度：{hit['score']:.4f}")
        print()
    print(f"{'─'*50}\n")

    # 生成 LLM 回答
    print("正在调用 Qwen LLM 生成回答...\n")
    result = retriever.query(question)
    print(f"LLM 回答：\n{result['answer']}")
