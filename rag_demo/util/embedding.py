"""共享 Embedding 模块

提供统一的文本向量化接口，供 rag_demo 各模块使用。
使用阿里云 DashScope text-embedding-v4 模型（1024 维）。
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

# Embedding 客户端（模块级单例，所有模块共享）
embedding_client = OpenAI(
    api_key=os.getenv("ALIYUN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# text-embedding-v4 默认输出维度
DEFAULT_EMBEDDING_DIMENSION = 1024


def generate_embedding(text: str, dimensions: int = DEFAULT_EMBEDDING_DIMENSION) -> list[float]:
    """生成文本的向量表示

    Args:
        text: 输入文本
        dimensions: 向量维度，默认 1024（text-embedding-v4 默认维度）

    Returns:
        向量列表（1024 维浮点数列表）
    """
    completion = embedding_client.embeddings.create(
        model="text-embedding-v4",
        input=text,
        dimensions=dimensions,
        encoding_format="float",
    )
    return completion.data[0].embedding
