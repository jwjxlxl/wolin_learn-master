import os
import json
import hashlib
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

from pymilvus import MilvusClient, DataType, Function, FunctionType
from openai import OpenAI

# 加载环境变量
load_dotenv()

# 连接配置
SERVER_ADDR = "http://localhost:19530"

# 初始化阿里云 Embedding 客户端
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


def init_collection():
    """初始化 Milvus 集合用于存储文件切片,支持混合检索"""

    # 创建 MilvusClient 实例
    client = MilvusClient(uri=SERVER_ADDR)

    # 定义集合 schema
    schema = client.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )

    # 添加主键字段
    schema.add_field(
        field_name="chunk_id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        max_length=100
    )

    # 添加文本字段 (BM25 的输入字段)
    schema.add_field(
        field_name="chunk_text",
        datatype=DataType.VARCHAR,
        max_length=65535,
        enable_analyzer=True,  # 启用分析器用于 BM25
        enable_match=True  # 启用全文匹配
    )

    # 添加密集向量字段 (用于语义搜索)
    schema.add_field(
        field_name="dense_embedding",
        datatype=DataType.FLOAT_VECTOR,
        dim=1536
    )

    # 添加稀疏向量字段 (BM25 的输出字段)
    schema.add_field(
        field_name="sparse_embedding",
        datatype=DataType.SPARSE_FLOAT_VECTOR,
        is_function_output=True
    )

    # 添加文件元数据字段
    schema.add_field(
        field_name="file_name",
        datatype=DataType.VARCHAR,
        max_length=500
    )

    schema.add_field(
        field_name="file_path",
        datatype=DataType.VARCHAR,
        max_length=1000
    )

    schema.add_field(
        field_name="chunk_index",
        datatype=DataType.INT64
    )

    schema.add_field(
        field_name="file_size",
        datatype=DataType.INT64
    )

    schema.add_field(
        field_name="created_time",
        datatype=DataType.INT64
    )

    # 创建 BM25 函数
    bm25_function = Function(
        name="bm25_function",
        function_type=FunctionType.BM25,
        input_field_names=["chunk_text"],
        output_field_names=["sparse_embedding"],
        params={}
    )
    schema.add_function(bm25_function)

    # 准备索引参数
    index_params = client.prepare_index_params()

    # 为密集向量字段创建索引
    index_params.add_index(
        field_name="dense_embedding",
        index_type="AUTOINDEX",
        metric_type="COSINE"
    )

    # 为稀疏向量字段创建索引
    index_params.add_index(
        field_name="sparse_embedding",
        index_type="AUTOINDEX",
        metric_type="BM25"
    )

    # 为标量字段创建索引
    index_params.add_index(
        field_name="file_name",
        index_type="AUTOINDEX"
    )

    index_params.add_index(
        field_name="chunk_index",
        index_type="AUTOINDEX"
    )

    # 创建集合
    client.create_collection(
        collection_name="file_chunks",
        schema=schema,
        index_params=index_params
    )

    print("✅ Collection 'file_chunks' with BM25 support created successfully!")

    return client


def create_qa_pairs_collection():
    """初始化 Milvus 集合用于存储问答对，支持问题和答案的混合检索"""

    client = MilvusClient(uri=SERVER_ADDR)

    # 定义集合 schema
    schema = client.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )

    # 添加主键字段
    schema.add_field(
        field_name="id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        max_length=100
    )

    # 添加问题文本字段
    schema.add_field(
        field_name="question",
        datatype=DataType.VARCHAR,
        max_length=65535,
        enable_analyzer=True,
        enable_match=True
    )

    # 添加答案文本字段
    schema.add_field(
        field_name="answer",
        datatype=DataType.VARCHAR,
        max_length=65535,
        enable_analyzer=True,
        enable_match=True
    )

    # 添加思考过程字段
    schema.add_field(
        field_name="reasoning",
        datatype=DataType.VARCHAR,
        max_length=65535
    )

    # 添加问题的密集向量字段
    schema.add_field(
        field_name="question_dense_embedding",
        datatype=DataType.FLOAT_VECTOR,
        dim=1536
    )

    # 添加问题的稀疏向量字段
    schema.add_field(
        field_name="question_sparse_embedding",
        datatype=DataType.SPARSE_FLOAT_VECTOR,
        is_function_output=True
    )

    # 添加答案的密集向量字段
    schema.add_field(
        field_name="answer_dense_embedding",
        datatype=DataType.FLOAT_VECTOR,
        dim=1536
    )

    # 添加答案的稀疏向量字段
    schema.add_field(
        field_name="answer_sparse_embedding",
        datatype=DataType.SPARSE_FLOAT_VECTOR,
        is_function_output=True
    )

    # 创建 BM25 函数（问题）
    question_bm25 = Function(
        name="question_bm25",
        function_type=FunctionType.BM25,
        input_field_names=["question"],
        output_field_names=["question_sparse_embedding"],
        params={}
    )
    schema.add_function(question_bm25)

    # 创建 BM25 函数（答案）
    answer_bm25 = Function(
        name="answer_bm25",
        function_type=FunctionType.BM25,
        input_field_names=["answer"],
        output_field_names=["answer_sparse_embedding"],
        params={}
    )
    schema.add_function(answer_bm25)

    # 准备索引参数
    index_params = client.prepare_index_params()

    # 为问题的密集向量创建索引
    index_params.add_index(
        field_name="question_dense_embedding",
        index_type="AUTOINDEX",
        metric_type="COSINE"
    )

    # 为问题的稀疏向量创建索引
    index_params.add_index(
        field_name="question_sparse_embedding",
        index_type="AUTOINDEX",
        metric_type="BM25"
    )

    # 为答案的密集向量创建索引
    index_params.add_index(
        field_name="answer_dense_embedding",
        index_type="AUTOINDEX",
        metric_type="COSINE"
    )

    # 为答案的稀疏向量创建索引
    index_params.add_index(
        field_name="answer_sparse_embedding",
        index_type="AUTOINDEX",
        metric_type="BM25"
    )

    # 创建集合
    client.create_collection(
        collection_name="qa_pairs",
        schema=schema,
        index_params=index_params
    )

    print("✅ Collection 'qa_pairs' created successfully!")
    return client


def insert_qa_pairs(qa_file_path: str = None) -> int:
    """插入问答对数据到 Milvus

    Args:
        qa_file_path: 问答对 JSON 文件路径，默认使用 /data/qa_pair_test_data.json

    Returns:
        插入的记录数
    """
    # 默认文件路径
    if qa_file_path is None:
        qa_file_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "qa_pair_test_data.json"
        )

    # 读取问答对数据
    with open(qa_file_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    print(f"读取到 {len(qa_data)} 条问答对数据")

    # 初始化客户端和 Embedding
    client = MilvusClient(uri=SERVER_ADDR)
    embedding_client = get_embedding_client()

    # 准备插入的数据
    data_to_insert = []
    batch_size = 10  # 批量处理，避免内存溢出

    for i, qa in enumerate(qa_data):
        question = qa.get("question", "")
        answer = qa.get("answer", "")
        reasoning = qa.get("thinking_process", "")

        # 生成唯一 ID (使用问题的 hash)
        qa_id = hashlib.md5(f"{question}_{answer}".encode('utf-8')).hexdigest()

        # 获取向量化结果
        question_embedding = get_embedding(question, embedding_client)
        answer_embedding = get_embedding(answer, embedding_client)

        data_to_insert.append({
            "id": qa_id,
            "question": question,
            "answer": answer,
            "reasoning": reasoning,
            "question_dense_embedding": question_embedding,
            "answer_dense_embedding": answer_embedding,
        })

        # 每 batch_size 条插入一次
        if len(data_to_insert) >= batch_size:
            res = client.insert(
                collection_name="qa_pairs",
                data=data_to_insert
            )
            print(f"已插入 {i + 1} 条记录")
            data_to_insert = []

    # 插入剩余数据
    if data_to_insert:
        res = client.insert(
            collection_name="qa_pairs",
            data=data_to_insert
        )
        print(f"已插入最后一批记录")

    print(f"✅ 问答对数据插入完成，共 {len(qa_data)} 条")
    return len(qa_data)


def insert_file_chunks(
    file_path: str = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> int:
    """插入文件切片数据到 Milvus

    Args:
        file_path: 源文件路径，默认使用 /data/test.txt
        chunk_size: 每个切片的最大字符数
        chunk_overlap: 切片之间的重叠字符数

    Returns:
        插入的切片数
    """
    # 导入文本切片工具
    import sys
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils"))
    from text_splitter import TextSplitter

    # 默认文件路径
    if file_path is None:
        file_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "test.txt"
        )

    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"读取文件：{file_path}")
    print(f"文件总字符数：{len(content)}")

    # 初始化客户端和 Embedding
    client = MilvusClient(uri=SERVER_ADDR)
    embedding_client = get_embedding_client()

    # 使用文本切片工具进行切片
    splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split(
        text=content,
        metadata={
            "file_name": os.path.basename(file_path),
            "file_path": file_path,
            "file_size": os.path.getsize(file_path),
            "created_time": int(time.time())
        },
        method="recursive",
        return_with_metadata=True
    )

    print(f"切片数量：{len(chunks)}")

    # 准备插入的数据
    data_to_insert = []
    batch_size = 10

    for i, chunk in enumerate(chunks):
        chunk_text = chunk["content"]
        metadata = chunk["metadata"]

        # 生成唯一 ID
        chunk_id = hashlib.md5(
            f"{metadata['file_path']}_{metadata['chunk_index']}_{chunk_text[:50]}".encode('utf-8')
        ).hexdigest()

        # 获取向量化结果
        embedding = get_embedding(chunk_text, embedding_client)

        data_to_insert.append({
            "chunk_id": chunk_id,
            "chunk_text": chunk_text,
            "dense_embedding": embedding,
            "file_name": metadata.get("file_name", ""),
            "file_path": metadata.get("file_path", ""),
            "chunk_index": metadata.get("chunk_index", 0),
            "file_size": metadata.get("file_size", 0),
            "created_time": metadata.get("created_time", 0),
        })

        # 批量插入
        if len(data_to_insert) >= batch_size:
            res = client.insert(
                collection_name="file_chunks",
                data=data_to_insert
            )
            print(f"已插入 {i + 1} 个切片")
            data_to_insert = []

    # 插入剩余数据
    if data_to_insert:
        res = client.insert(
            collection_name="file_chunks",
            data=data_to_insert
        )
        print(f"已插入最后一批切片")

    print(f"✅ 文件切片数据插入完成，共 {len(chunks)} 个切片")
    return len(chunks)


if __name__ == "__main__":

    # count = insert_qa_pairs()
    # print(f"成功插入 {count} 条问答对")

    count = insert_file_chunks(chunk_size=1500)
    print(f"成功插入 {count} 个文件切片")

