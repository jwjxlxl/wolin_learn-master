import os
import json
import time
from pathlib import Path

from pymilvus import MilvusClient, DataType, Function, FunctionType
from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 连接到远程 Milvus 并指定数据库
client = MilvusClient(
    uri="http://47.115.57.130:19530",
    db_name="ai80"
)

# 初始化向量模型客户端
embedding_client = OpenAI(
    api_key=os.getenv("ALIYUN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


def generate_embedding(text: str, dimensions: int = 768) -> list[float]:
    """生成文本的向量表示

    Args:
        text: 输入文本
        dimensions: 向量维度，默认 768

    Returns:
        向量列表
    """
    completion = embedding_client.embeddings.create(
        model="text-embedding-v4",
        input=text,
        dimensions=dimensions,
        encoding_format="float"
    )
    return completion.data[0].embedding


def create_document_chunks_collection():
    """创建文档分片集合，支持密集向量+稀疏向量(BM25)混合检索"""
    collection_name = "document_chunks"

    if client.has_collection(collection_name=collection_name):
        print(f"集合 '{collection_name}' 已存在,正在删除...")
        client.drop_collection(collection_name=collection_name)
        print(f"集合 '{collection_name}' 已删除")

    schema = client.create_schema()

    schema.add_field(
        field_name="id",
        datatype=DataType.INT64,
        is_primary=True,
        auto_id=True
    )

    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=2000,
        enable_analyzer=True,  # 第一个开关:启用分析器
        enable_match=True      # 第二个开关:启用文本匹配
    )

    schema.add_field(
        field_name="file_name",
        datatype=DataType.VARCHAR,
        max_length=200
    )

    schema.add_field(
        field_name="chunk_index",
        datatype=DataType.INT32
    )

    schema.add_field(
        field_name="timestamp",
        datatype=DataType.INT32
    )

    schema.add_field(
        field_name="dense_vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=768
    )

    schema.add_field(
        field_name="sparse_vector",
        datatype=DataType.SPARSE_FLOAT_VECTOR
    )

    bm25_function = Function(
        name="text_bm25",
        input_field_names=["text"],
        output_field_names=["sparse_vector"],
        function_type=FunctionType.BM25,
    )
    schema.add_function(bm25_function)

    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="dense_vector",
        index_type="AUTOINDEX",
        metric_type="COSINE"
    )

    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25"
    )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )

    print(f"集合 '{collection_name}' 创建成功!")


def create_qa_pairs_collection():
    """创建问答对集合，包含问题、答案、推理过程三个字段，支持问题的向量检索和BM25检索"""
    collection_name = "qa_pairs"

    if client.has_collection(collection_name=collection_name):
        print(f"集合 '{collection_name}' 已存在,正在删除...")
        client.drop_collection(collection_name=collection_name)
        print(f"集合 '{collection_name}' 已删除")

    schema = client.create_schema()

    schema.add_field(
        field_name="qa_id",
        datatype=DataType.INT64,
        is_primary=True,
        auto_id=True
    )

    schema.add_field(
        field_name="question",
        datatype=DataType.VARCHAR,
        max_length=1000,
        enable_analyzer=True,  # 第一个开关:启用分析器
        enable_match=True      # 第二个开关:启用文本匹配
    )

    schema.add_field(
        field_name="answer",
        datatype=DataType.VARCHAR,
        max_length=2000
    )

    schema.add_field(
        field_name="reasoning",
        datatype=DataType.VARCHAR,
        max_length=2000
    )

    schema.add_field(
        field_name="dense_vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=768
    )

    schema.add_field(
        field_name="sparse_vector",
        datatype=DataType.SPARSE_FLOAT_VECTOR
    )

    bm25_function = Function(
        name="text_bm25",
        input_field_names=["question"],
        output_field_names=["sparse_vector"],
        function_type=FunctionType.BM25,
    )
    schema.add_function(bm25_function)

    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="dense_vector",
        index_type="AUTOINDEX",
        metric_type="COSINE"
    )

    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25"
    )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )

    print(f"集合 '{collection_name}' 创建成功!")


def insert_document_chunks(txt_file_path: str, chunk_size: int = 500, chunk_overlap: int = 50, batch_size: int = 50):
    """读取本地 TXT 文件，切片后插入 document_chunks 集合

    Args:
        txt_file_path: 本地 TXT 文件路径
        chunk_size: 每个切片的最大字符数
        chunk_overlap: 切片之间的重叠字符数
        batch_size: 批量插入的批次大小
    """
    from rag_demo.util.text_parser import parse_txt_file
    from rag_demo.util.text_splitter import TextChunker

    # 解析文件
    text = parse_txt_file(txt_file_path)
    file_name = Path(txt_file_path).name
    print(f"文件 '{file_name}' 解析完成，总字符数: {len(text)}")

    # 切片
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.split(text, metadata={"file_name": file_name})
    print(f"切片完成，共 {len(chunks)} 个切片")

    # 生成向量并插入
    data_batch = []
    timestamp = int(time.time())

    for i, chunk in enumerate(chunks):
        dense_vector = generate_embedding(chunk["content"])
        data_batch.append({
            "text": chunk["content"],
            "file_name": file_name,
            "chunk_index": chunk["metadata"]["chunk_index"],
            "timestamp": timestamp,
            "dense_vector": dense_vector,
        })

        # 批量插入
        if len(data_batch) >= batch_size or i == len(chunks) - 1:
            client.insert(collection_name="document_chunks", data=data_batch)
            print(f"已插入 {i + 1}/{len(chunks)} 个切片")
            data_batch = []

    print(f"document_chunks 数据插入完成!")


def insert_qa_pairs(json_file_path: str, batch_size: int = 50):
    """读取 QA JSON 文件，生成向量后插入 qa_pairs 集合

    Args:
        json_file_path: 本地 JSON 文件路径
        batch_size: 批量插入的批次大小
    """
    with open(json_file_path, "r", encoding="utf-8") as f:
        qa_list = json.load(f)

    print(f"QA 数据加载完成，共 {len(qa_list)} 条")

    data_batch = []

    for i, qa in enumerate(qa_list):
        dense_vector = generate_embedding(qa["question"])
        data_batch.append({
            "question": qa["question"],
            "answer": qa["answer"],
            "reasoning": qa["reasoning"],
            "dense_vector": dense_vector,
        })

        # 批量插入
        if len(data_batch) >= batch_size or i == len(qa_list) - 1:
            client.insert(collection_name="qa_pairs", data=data_batch)
            print(f"已插入 {i + 1}/{len(qa_list)} 条 QA")
            data_batch = []

    print(f"qa_pairs 数据插入完成!")


if __name__ == "__main__":
    # create_document_chunks_collection()
    # create_qa_pairs_collection()

    data_dir = Path(__file__).parent.parent / "datas"
    # insert_document_chunks(str(data_dir / "三国演义.txt"))
    insert_qa_pairs(str(data_dir / "qa_paris_additional.json"))

    '''
        向量库中的数据，可以提供一个接口进行更新
        增量和全量数据的更新
    '''