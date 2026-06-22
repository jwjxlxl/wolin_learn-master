# =============================================================================
# Milvus 向量库初始化 — 文档切片 + 问答对存储
# =============================================================================
#
# 用途：创建两个 Milvus Collection，均支持混合检索
#       1. 文档切片表（text + BM25 稀疏向量 + 稠密向量）
#       2. 问答对表（question + BM25 稀疏向量 + 稠密向量）
#
# 使用方法：
#   python -m rag_examples.sg_rag.db.vdb_init
# =============================================================================

import os
import time
from dotenv import load_dotenv
from pymilvus import (
    MilvusClient,
    DataType,
    Function,
    FunctionType,
)

from rag_examples.milvus_config import MILVUS_URI, DEFAULT_DIMENSION

load_dotenv()

# =============================================================================
# 常量定义
# =============================================================================

COLLECTION_NAME = os.getenv("SG_RAG_COLLECTION", "sanguo_doc_chunks")
QA_COLLECTION_NAME = os.getenv("SG_RAG_QA_COLLECTION", "sanguo_qa_pairs")

# 文档切片字段最大长度
MAX_TEXT_LENGTH = 65535
MAX_FILE_NAME_LENGTH = 512
MAX_TITLE_LENGTH = 256
MAX_CATEGORY_LENGTH = 64

# 问答对字段最大长度
MAX_QUESTION_LENGTH = 2000
MAX_ANSWER_LENGTH = 65535
MAX_REASONING_LENGTH = 65535

# =============================================================================
# 集合创建
# =============================================================================


def create_collection(client: MilvusClient, collection_name: str = COLLECTION_NAME, drop_if_exists: bool = False):
    """
    创建文档切片 Collection（含 BM25 稀疏向量字段，支持混合检索）

    参数：
        client: MilvusClient 实例
        collection_name: 集合名称
        drop_if_exists: 若集合已存在是否先删除重建

    返回：
        str: 集合名称
    """
    if drop_if_exists and client.has_collection(collection_name):
        client.drop_collection(collection_name)
        print(f"[INFO] 已删除旧集合：{collection_name}")

    if client.has_collection(collection_name):
        print(f"[WARN] 集合 {collection_name} 已存在，跳过创建")
        return collection_name

    # 创建 Schema
    schema = client.create_schema()

    # --- 主键 ---
    schema.add_field(
        field_name="id",
        datatype=DataType.INT64,
        is_primary=True,
        auto_id=True,
    )

    # --- 文本字段（启用分词器 + 匹配，BM25 的输入源） ---
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=MAX_TEXT_LENGTH,
        enable_analyzer=True,   # 启用中文分词
        enable_match=True,      # 启用 BM25 匹配
    )

    # --- 元数据字段 ---
    schema.add_field(
        field_name="file_name",
        datatype=DataType.VARCHAR,
        max_length=MAX_FILE_NAME_LENGTH,
    )

    schema.add_field(
        field_name="chunk_index",
        datatype=DataType.INT32,
    )

    schema.add_field(
        field_name="timestamp",
        datatype=DataType.INT64,
    )

    schema.add_field(
        field_name="title",
        datatype=DataType.VARCHAR,
        max_length=MAX_TITLE_LENGTH,
        is_nullable=True,
    )

    schema.add_field(
        field_name="category",
        datatype=DataType.VARCHAR,
        max_length=MAX_CATEGORY_LENGTH,
        is_nullable=True,
    )

    # --- 稠密向量字段（语义检索） ---
    schema.add_field(
        field_name="dense_vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=DEFAULT_DIMENSION,  # 1024
    )

    # --- 稀疏向量字段（BM25 关键字检索，由 Function 自动填充） ---
    schema.add_field(
        field_name="sparse_vector",
        datatype=DataType.SPARSE_FLOAT_VECTOR,
    )

    # --- BM25 Function：text -> sparse_vector ---
    bm25_fn = Function(
        name="text_bm25",
        input_field_names=["text"],
        output_field_names=["sparse_vector"],
        function_type=FunctionType.BM25,
    )
    schema.add_function(bm25_fn)

    # --- 创建集合 ---
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
    )
    print(f"[OK] 集合 {collection_name} 创建成功")

    # --- 创建索引 ---
    _create_indexes(client, collection_name)

    # --- 加载集合到内存 ---
    client.load_collection(collection_name)
    print(f"[OK] 集合 {collection_name} 已加载到内存")

    return collection_name


def _create_indexes(client: MilvusClient, collection_name: str):
    """
    为集合创建索引

    - dense_vector: AUTOINDEX（COSINE 度量）
    - sparse_vector: SPARSE_INVERTED_INDEX（BM25 度量）
    """
    index_params = client.prepare_index_params()

    # 稠密向量索引 — 语义检索
    index_params.add_index(
        field_name="dense_vector",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )

    # 稀疏向量索引 — BM25 关键字检索
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
    )

    client.create_index(
        collection_name=collection_name,
        index_params=index_params,
    )
    print(f"[OK] 索引创建完成（dense: AUTOINDEX, sparse: SPARSE_INVERTED_INDEX）")


# =============================================================================
# 集合管理工具函数
# =============================================================================


def get_collection_info(client: MilvusClient, collection_name: str = COLLECTION_NAME):
    """
    打印集合的详细信息（字段、索引、统计）

    参数：
        client: MilvusClient 实例
        collection_name: 集合名称
    """
    if not client.has_collection(collection_name):
        print(f"[ERROR] 集合 {collection_name} 不存在")
        return

    info = client.get_collection_stats(collection_name)
    print(f"\n{'─'*50}")
    print(f"集合：{collection_name}")
    print(f"{'─'*50}")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # 打印字段信息
    schema = client.describe_collection(collection_name)
    print(f"\n  字段列表：")
    for field in schema.get("fields", []):
        print(f"    - {field['name']}: {field['type']}")

    print(f"{'─'*50}\n")


def drop_collection(client: MilvusClient, collection_name: str = COLLECTION_NAME):
    """
    删除集合

    参数：
        client: MilvusClient 实例
        collection_name: 集合名称
    """
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
        print(f"[OK] 集合 {collection_name} 已删除")
    else:
        print(f"[WARN] 集合 {collection_name} 不存在")


# =============================================================================
# 问答对集合
# =============================================================================


def build_qa_schema():
    """
    构建问答对 Collection 的 Schema

    返回：
        CollectionSchema 实例
    """
    schema = MilvusClient.create_schema()

    # --- 主键 ---
    schema.add_field(
        field_name="qa_id",
        datatype=DataType.INT64,
        is_primary=True,
        auto_id=True,
    )

    # --- 问题文本（启用分词器 + 匹配，BM25 的输入源） ---
    schema.add_field(
        field_name="question",
        datatype=DataType.VARCHAR,
        max_length=MAX_QUESTION_LENGTH,
        enable_analyzer=True,
        enable_match=True,
    )

    # --- 答案文本 ---
    schema.add_field(
        field_name="answer",
        datatype=DataType.VARCHAR,
        max_length=MAX_ANSWER_LENGTH,
    )

    # --- 推理过程/解析（可选） ---
    schema.add_field(
        field_name="reasoning",
        datatype=DataType.VARCHAR,
        max_length=MAX_REASONING_LENGTH,
        is_nullable=True,
    )

    # --- 稠密向量字段（问题的语义向量） ---
    schema.add_field(
        field_name="dense_vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=DEFAULT_DIMENSION,
    )

    # --- 稀疏向量字段（BM25 关键字检索，由 Function 自动填充） ---
    schema.add_field(
        field_name="sparse_vector",
        datatype=DataType.SPARSE_FLOAT_VECTOR,
    )

    # --- BM25 Function：question -> sparse_vector ---
    bm25_fn = Function(
        name="question_bm25",
        input_field_names=["question"],
        output_field_names=["sparse_vector"],
        function_type=FunctionType.BM25,
    )
    schema.add_function(bm25_fn)

    return schema


def create_qa_indexes(client: MilvusClient, collection_name: str):
    """
    为问答对集合创建索引

    - dense_vector: AUTOINDEX（COSINE 度量）
    - sparse_vector: SPARSE_INVERTED_INDEX（BM25 度量）

    参数：
        client: MilvusClient 实例
        collection_name: 集合名称
    """
    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="dense_vector",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )

    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
    )

    client.create_index(
        collection_name=collection_name,
        index_params=index_params,
    )
    print(f"[OK] 问答对索引创建完成（dense: AUTOINDEX, sparse: SPARSE_INVERTED_INDEX）")


def create_qa_collection(client: MilvusClient, collection_name: str = QA_COLLECTION_NAME, drop_if_exists: bool = False):
    """
    创建问答对 Collection（含 BM25 稀疏向量字段，支持混合检索）

    参数：
        client: MilvusClient 实例
        collection_name: 集合名称
        drop_if_exists: 若集合已存在是否先删除重建

    返回：
        str: 集合名称
    """
    if drop_if_exists and client.has_collection(collection_name):
        client.drop_collection(collection_name)
        print(f"[INFO] 已删除旧问答对集合：{collection_name}")

    if client.has_collection(collection_name):
        print(f"[WARN] 问答对集合 {collection_name} 已存在，跳过创建")
        return collection_name

    # 构建 Schema
    schema = build_qa_schema()

    # 创建集合
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
    )
    print(f"[OK] 问答对集合 {collection_name} 创建成功")

    # 创建索引
    create_qa_indexes(client, collection_name)

    # 加载集合到内存
    client.load_collection(collection_name)
    print(f"[OK] 问答对集合 {collection_name} 已加载到内存")

    return collection_name


# =============================================================================
# 向量化工具
# =============================================================================

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIYUN_API_KEY", "")


def generate_embedding(text: str) -> list[float]:
    """
    调用阿里云 DashScope text-embedding-v4 生成 1024 维向量

    参数：
        text: 待向量化的文本内容

    返回：
        list[float]: 1024 维稠密向量

    异常：
        RuntimeError: API 调用失败或无环境变量时抛出
    """
    from openai import OpenAI

    if not DASHSCOPE_API_KEY:
        raise RuntimeError(
            "未配置 DASHSCOPE_API_KEY 或 ALIYUN_API_KEY，请在 .env 文件中设置"
        )

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
# 数据插入 — 文档切片
# =============================================================================


def insert_doc_chunks(
    client: MilvusClient,
    file_path: str,
    collection_name: str = COLLECTION_NAME,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    batch_size: int = 50,
):
    """
    读取 TXT 文件 → 递归切片 → 逐块向量化 → 批量插入文档切片表

    参数：
        client: MilvusClient 实例
        file_path: TXT 文件路径
        collection_name: 目标集合名称
        chunk_size: 每块最大字符数
        chunk_overlap: 相邻块重叠字符数
        batch_size: 每次批量插入的条数

    返回：
        dict: 包含 insert_count（插入数量）等信息
    """
    from pathlib import Path
    import sys

    # 确保项目根目录在路径中
    project_root = str(Path(__file__).resolve().parents[3])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from rag_examples.sg_rag.utiles.file_utiles import parse_txt
    from rag_examples.sg_rag.utiles.text_chunker import TextChunker

    if not client.has_collection(collection_name):
        raise RuntimeError(f"集合 {collection_name} 不存在，请先调用 create_collection() 创建")

    # 1. 读取文件
    print(f"\n[INFO] 正在读取文件：{file_path}")
    text = parse_txt(file_path)
    file_name = Path(file_path).name
    print(f"[OK] 文件读取完成，共 {len(text)} 个字符")

    # 2. 递归切片
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.split(text, metadata={"file_name": file_name})
    print(f"[OK] 切片完成，共 {len(chunks)} 块")

    # 3. 批量向量化 + 插入
    timestamp = int(time.time())
    total_inserted = 0
    errors = 0

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        data_batch = []

        # 逐块向量化
        for chunk in batch_chunks:
            try:
                dense_vector = generate_embedding(chunk["content"])
            except Exception as e:
                print(f"[WARN] 第 {chunk['chunk_index']} 块向量化失败，已跳过：{e}")
                errors += 1
                continue

            data_batch.append({
                "text": chunk["content"],
                "file_name": chunk.get("file_name", file_name),
                "chunk_index": chunk["chunk_index"],
                "timestamp": timestamp,
                "title": chunk.get("file_name", file_name),
                "category": " ",
                "dense_vector": dense_vector,
            })

        # 批量插入（sparse_vector 由 BM25 Function 自动生成）
        if data_batch:
            result = client.insert(
                collection_name=collection_name,
                data=data_batch,
            )
            count = result.get("insert_count", len(data_batch))
            total_inserted += count
            print(f"[INFO] 已插入 {total_inserted}/{len(chunks)} 条"
                  f"（本批 {count} 条）")

    print(f"\n[OK] 文档切片插入完成：共插入 {total_inserted} 条"
          f"，跳过 {errors} 条（向量化失败）")

    return {
        "insert_count": total_inserted,
        "total_chunks": len(chunks),
        "errors": errors,
    }


# =============================================================================
# 数据插入 — 问答对
# =============================================================================


def insert_qa_pairs(
    client: MilvusClient,
    file_path: str,
    collection_name: str = QA_COLLECTION_NAME,
    batch_size: int = 50,
):
    """
    读取 JSON 问答文件 → 逐条向量化问题 → 批量插入问答对表

    参数：
        client: MilvusClient 实例
        file_path: JSON 文件路径（格式：list[dict(question, answer, reason)]）
        collection_name: 目标集合名称
        batch_size: 每次批量插入的条数

    返回：
        dict: 包含 insert_count（插入数量）等信息
    """
    from pathlib import Path
    import json

    if not client.has_collection(collection_name):
        raise RuntimeError(f"集合 {collection_name} 不存在，请先调用 create_qa_collection() 创建")

    # 1. 读取 JSON 文件
    print(f"\n[INFO] 正在读取问答文件：{file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        qa_list = json.load(f)

    print(f"[OK] 问答文件读取完成，共 {len(qa_list)} 条问答")

    # 2. 批量向量化 + 插入
    total_inserted = 0
    errors = 0

    for i in range(0, len(qa_list), batch_size):
        batch = qa_list[i : i + batch_size]
        data_batch = []

        for item in batch:
            question = item.get("question", "")
            answer = item.get("answer", "")
            reason = item.get("reason", "")

            if not question or not answer:
                print(f"[WARN] 跳过空问题或空答案")
                errors += 1
                continue

            try:
                dense_vector = generate_embedding(question)
            except Exception as e:
                print(f"[WARN] 问题向量化失败，已跳过：{e}")
                errors += 1
                continue

            data_batch.append({
                "question": question,
                "answer": answer,
                "reasoning": reason if reason else None,
                "dense_vector": dense_vector,
            })

        # 批量插入（sparse_vector 由 BM25 Function 自动生成）
        if data_batch:
            result = client.insert(
                collection_name=collection_name,
                data=data_batch,
            )
            count = result.get("insert_count", len(data_batch))
            total_inserted += count
            print(f"[INFO] 已插入 {total_inserted}/{len(qa_list)} 条"
                  f"（本批 {count} 条）")

    print(f"\n[OK] 问答对插入完成：共插入 {total_inserted} 条"
          f"，跳过 {errors} 条（向量化失败）")

    return {
        "insert_count": total_inserted,
        "total_qa": len(qa_list),
        "errors": errors,
    }


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path


    client = MilvusClient(uri=MILVUS_URI)
    print(f"[OK] Milvus 客户端初始化成功")

    try:
        version = client.get_server_version()
        print(f"[OK] Milvus 服务版本：{version}")
    except Exception as e:
        print(f"[ERROR] 无法连接 Milvus 服务：{e}")
        exit(1)

    print()

    # 1. 创建文档切片集合
    # create_collection(client, drop_if_exists=False)

    # 打印文档切片集合信息
    # get_collection_info(client)

    print()

    # 2. 创建问答对集合
    # create_qa_collection(client, drop_if_exists=False)

    # 打印问答对集合信息
    # get_collection_info(client, QA_COLLECTION_NAME)

    print("=" * 60)
    print("开始插入数据")
    print("=" * 60)

    # 数据文件路径
    data_dir = Path(__file__).resolve().parents[1] / "datas"
    txt_file = str(data_dir / "《三国演义》.txt")
    json_file = str(data_dir / "qa_demo.json")

    # 3. 插入文档切片
    if DASHSCOPE_API_KEY and Path(txt_file).exists():
        insert_doc_chunks(client, txt_file)
    elif not Path(txt_file).exists():
        print(f"[WARN] 文件不存在，跳过文档切片插入：{txt_file}")
    else:
        print("[WARN] DASHSCOPE_API_KEY 未配置，跳过文档切片插入")

    print()

    # 4. 插入问答对
    if DASHSCOPE_API_KEY and Path(json_file).exists():
        insert_qa_pairs(client, json_file)
    elif not Path(json_file).exists():
        print(f"[WARN] 文件不存在，跳过问答对插入：{json_file}")
    else:
        print("[WARN] DASHSCOPE_API_KEY 未配置，跳过问答对插入")

    print()
    print("[DONE] 全部操作完成")
