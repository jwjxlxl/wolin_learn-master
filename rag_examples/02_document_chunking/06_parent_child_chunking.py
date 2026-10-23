# =============================================================================
# 06_parent_child_chunking — 父子切块（Parent-Child Chunking / Small-to-Big）
# =============================================================================
# 用途：学习"父子切块"策略 — 小块负责检索精准，大块负责上下文完整（Milvus 实操版）
# 难度：⭐⭐⭐（3 星）
# =============================================================================
# 核心思想（Small-to-Big，小块检索、大块返回）：
#   切块大小是一个两难问题：
#     切太大 → 一个块混多个主题 → 向量被无关内容稀释 → 检索不准
#     切太小 → 语义不完整 → 检索准了，但上下文太碎 → 答不好
#   父子切块 = 鱼和熊掌兼得：
#     检索时用"子块"（小、聚焦）→ 找得准
#     返回时用"父块"（大、完整）→ 给得全
#
# 本文件是 Milvus 实操版：真实入库 + 真实向量化，模拟生产中的父子切块
#   单集合设计（一张 pc_chunks 表装下父块和子块，用 level 字段区分）：
#     level == "parent"：父块行 = 原文
#     level == "child" ：子块行 = 原文 + 真实向量（text-embedding-v4）+ parent_id
#
#   为什么父块行也要塞一个占位向量？
#     Milvus 要求每张 Collection 必须包含向量字段（纯标量集合建表会报
#     code=1100 "schema does not contain vector field"）。
#     占位向量永不参与检索 —— 检索时 filter='level == "child"' 只搜子块行。
#
# 工作流程（对应本文件两个核心函数）：
#   index_parent_child(client, emb, doc):
#     ① 先切父块（大，生产典型 1024 字 / overlap 100）  → 父块行入库（占位向量，level=parent）
#     ② 父块再切子块（小，生产典型 512 字 / overlap 50）→ 子块行向量化入库（level=child，记录 parent_id）
#   retrieve_with_parent(client, emb, question):
#     ③ 问题向量化 → 只搜子块行（filter 过滤父块行）→ 命中精准的小块
#     ④ 通过 parent_id 回填父块行原文并去重           → 返回完整的大块上下文
#
# 运行前提：
#   1. Milvus 服务可用（连接配置见 rag_examples/milvus_config.py 与 .env）
#   2. 已设置 ALIYUN_API_KEY（阿里云百炼，向量化用）
#
# 演示说明：只演示"子块检索 + 父块回填"的效果，不调用大模型生成答案。
# =============================================================================

from pathlib import Path
from dotenv import load_dotenv

# 先加载 rag_examples/.env（课程统一配置），再加载默认位置（不覆盖已加载的值）
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv()

from rag_examples.milvus_config import MILVUS_URI, DEFAULT_DIMENSION
from rag_examples.embedding_examples.aliyun_embedding import AliyunEmbedding
from pymilvus import MilvusClient, DataType


# =============================================================================
# 参数配置
# =============================================================================

# 演示用参数（小数量级，方便肉眼看清父子结构；生产值见文件头注释）
PARENT_SIZE = 160      # 父块大小（生产典型：1024）
PARENT_OVERLAP = 20    # 父块间重叠（生产典型：100）
CHILD_SIZE = 64        # 子块大小（生产典型：512）
CHILD_OVERLAP = 8      # 子块间重叠（生产典型：50）

# Milvus 集合名称（父块 + 子块同表，用 level 字段区分）
CHUNK_COLLECTION = "pc_chunks_demo"

# 手动分配主键：父块从 1 起，子块从 100000 起（打印时一眼区分父子）
PARENT_ID_START = 1
CHILD_ID_START = 100000
ROOT_PARENT_ID = -1    # 父块行的 parent_id 占位值（父块没有上级）

# 父块行占位向量（非零固定值；Milvus 强制要求向量字段，父块永不参与向量检索）
PLACEHOLDER_VECTOR = [0.1] * DEFAULT_DIMENSION

# 子块检索数量
TOP_K = 3


# =============================================================================
# 基础组件：split / dedupe
# =============================================================================

def split(text, chunk_size, overlap):
    """带重叠的滑动窗口切块（复用 02_sliding_window 的核心逻辑）

    参数:
        text: 输入文本
        chunk_size: 每块字符数
        overlap: 相邻块重叠字符数
    返回:
        切块列表（每块都是原文的连续子串）
    """
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")

    step = chunk_size - overlap  # 步长 = 窗口 - 重叠
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        # 当前窗口已覆盖到文本末尾，停止滑动
        if start + chunk_size >= len(text):
            break
        start += step
    return chunks


def dedupe(items):
    """去重（保持原顺序）

    top_k 个子块可能属于同一个父块，回填时只需保留一份父块
    """
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


# =============================================================================
# Milvus 操作：建集合 / 入库 / 检索回填（对照伪代码实现）
# =============================================================================

def create_collection(client):
    """在 Milvus 中创建父子块单集合（已存在则先删除，保证可重复演示）

    单集合设计：一张 pc_chunks 表同时装父块行（level=parent）和子块行（level=child）。
    Milvus 要求每张集合必须含向量字段，因此父块行用固定占位向量入库；
    检索时通过 filter='level == "child"' 保证父块行永不参与向量检索。
    """
    if client.has_collection(CHUNK_COLLECTION):
        client.drop_collection(CHUNK_COLLECTION)

    schema = client.create_schema()
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64,
                     is_primary=True, auto_id=False)  # 手动分配：父块 1..N，子块 100000..
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=8192)
    schema.add_field(field_name="parent_id", datatype=DataType.INT64)  # 子块行指向父块行；父块行占位 -1
    schema.add_field(field_name="level", datatype=DataType.VARCHAR, max_length=16)  # "parent" / "child"
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=DEFAULT_DIMENSION)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
    client.create_collection(collection_name=CHUNK_COLLECTION, schema=schema,
                             index_params=index_params)
    print(f"✓ 集合 '{CHUNK_COLLECTION}' 创建成功"
          f"（chunk_id + text + parent_id + level + {DEFAULT_DIMENSION} 维向量）")


def index_parent_child(client, emb, doc):
    """索引流程：① 先切父块（大）入库 ② 父块再切子块（小）向量化入库

    参数:
        client: MilvusClient 实例
        emb: AliyunEmbedding 实例（阿里云 text-embedding-v4）
        doc: 原始长文档文本
    返回:
        (父块数量, 子块数量)
    """
    # ① 先切父块（大），手动分配 chunk_id（1, 2, 3...），占位向量入库
    parents = split(doc, chunk_size=PARENT_SIZE, overlap=PARENT_OVERLAP)
    parent_rows = [
        {
            "chunk_id": PARENT_ID_START + i,
            "text": p,
            "parent_id": ROOT_PARENT_ID,   # 父块没有上级，占位为 -1
            "level": "parent",
            "vector": PLACEHOLDER_VECTOR,  # 占位向量：父块不真实向量化
        }
        for i, p in enumerate(parents)
    ]
    client.insert(collection_name=CHUNK_COLLECTION, data=parent_rows)
    parent_ids = [r["chunk_id"] for r in parent_rows]
    print(f"父块入库：{len(parents)} 条（level=parent，占位向量），chunk_id：{parent_ids}")

    # ② 每个父块再切子块（小），手动分配 chunk_id（100000 起），记录所属 parent_id
    child_rows = []
    next_id = CHILD_ID_START
    for parent_text, pid in zip(parents, parent_ids):
        for child_text in split(parent_text, chunk_size=CHILD_SIZE, overlap=CHILD_OVERLAP):
            child_rows.append({"chunk_id": next_id, "text": child_text,
                               "parent_id": pid, "level": "child"})
            next_id += 1

    # 子块批量向量化（阿里云 text-embedding-v4，1024 维）
    print(f"子块向量化：{len(child_rows)} 条，调用 Embedding API...")
    vectors = emb.embed_batch([r["text"] for r in child_rows])

    # 维度校验：1024 维向量插入非 1024 维集合会直接失败
    assert len(vectors[0]) == DEFAULT_DIMENSION, (
        f"向量维度 {len(vectors[0])} 与集合维度 {DEFAULT_DIMENSION} 不一致")

    for r, v in zip(child_rows, vectors):
        r["vector"] = v

    # 子块入库（文本 + 真实向量 + 归属关系）
    client.insert(collection_name=CHUNK_COLLECTION, data=child_rows)
    print(f"子块入库：{len(child_rows)} 条（level=child，真实向量 + parent_id）")

    # 落盘并加载集合，保证后续检索可见
    client.flush(CHUNK_COLLECTION)
    client.load_collection(CHUNK_COLLECTION)

    return len(parents), len(child_rows)


def retrieve_with_parent(client, emb, question, top_k=TOP_K):
    """检索流程：③ 子块行向量检索（精准）④ 回填父块行原文（完整）

    参数:
        client: MilvusClient 实例
        emb: AliyunEmbedding 实例
        question: 用户问题
        top_k: 子块检索数量
    返回:
        (children, parents)：
        children: 命中子块列表（含 chunk_id / text / parent_id / distance）
        parents:  去重后的父块文本列表（可直接作为上下文交给下游使用）
    """
    # ③ 问题向量化 → 只在子块行中检索（filter 过滤父块行，占位向量永不参与）
    query_vector = emb.embed(question)
    res = client.search(
        collection_name=CHUNK_COLLECTION,
        data=[query_vector],
        filter='level == "child"',
        limit=top_k,
        output_fields=["chunk_id", "text", "parent_id"],
    )
    children = res[0]  # top_k 个命中子块

    # ④ 收集 parent_id（保序去重）→ 按主键取回父块行原文（level=parent）
    parent_ids = dedupe([hit["entity"]["parent_id"] for hit in children])
    filter_str = (f'level == "parent" and chunk_id in '
                  f'[{", ".join(str(pid) for pid in parent_ids)}]')
    rows = client.query(
        collection_name=CHUNK_COLLECTION,
        filter=filter_str,
        output_fields=["text"],
    )
    # query 返回顺序不保证与命中顺序一致，建映射后按命中顺序重排
    row_map = {row["chunk_id"]: row["text"] for row in rows}
    parents = [row_map[pid] for pid in parent_ids if pid in row_map]
    return children, parents


# =============================================================================
# 演示数据：一篇含多个主题的长文档（约 900 字，能切出多个父块）
# =============================================================================

DEMO_DOC = (
    "Milvus 是一款开源的向量数据库，专门用于存储和管理海量高维向量。"
    "它支持向量相似度检索、关键字检索与混合检索三种检索方式，"
    "并提供 FLAT、IVF、HNSW 等多种索引类型。"
    "Milvus 采用存储与计算分离的云原生架构，能够扩展到十亿级向量规模，"
    "同时提供标量字段过滤能力，方便结合业务属性做混合查询。"
    "凭借毫秒级的检索性能，Milvus 被广泛用于图像搜索、视频推荐、"
    "智能问答与药物发现等场景，是构建现代 RAG 系统时最流行的向量存储选择之一。"

    "文档切片是构建 RAG 知识库的第一步，作用是把长文档切割成大小合适的块。"
    "常见方法包括固定字符切片、滑动窗口切片、按段落切片以及借助大模型辅助的语义切片。"
    "切片太大时一个块里混杂多个主题，向量表示被无关内容稀释，检索精度会明显下降；"
    "切片太小时语义不完整，虽然定位更精准，但缺少上下文会让生成质量变差。"
    "工程实践中通常从三百到五百字符起步，再配合百分之十到二十的重叠率，"
    "在语义完整与检索精度之间取得平衡。"

    "向量化模型负责把文字转换成计算机可以计算的向量，"
    "让语义相近的文本在向量空间中的距离也更近。"
    "常用的文本向量化模型输出维度从五百一十二到一千零二十四不等，"
    "维度越高表达能力越强，存储与计算成本也相应增加。"
    "以阿里云的向量化服务为例，一段输入文本会返回一个固定长度的浮点数组，"
    "两个向量之间的余弦相似度可以衡量文本的语义相似程度，"
    "相似度越接近一，表示两段文字的含义越接近。"

    "检索增强生成 RAG 是缓解大模型幻觉的主流方案，"
    "思路是先从外部知识库检索相关内容，再把检索结果拼进提示词交给大模型生成回答。"
    "一个完整的 RAG 流程包括文档加载、文档切片、向量化、入库、检索重排与答案生成六个环节。"
    "RAG 的优势在于知识可以随时更新而无需重新训练模型，还能在回答中引用来源，"
    "让输出可追溯可验证，因此被广泛应用于企业知识库问答、"
    "智能客服与代码助手等产品形态。"

    "深度学习通过多层神经网络自动学习数据的层级特征，是当前人工智能的核心技术。"
    "训练过程依赖反向传播算法计算梯度，再用随机梯度下降不断调整网络权重。"
    "卷积神经网络擅长处理图像任务，循环神经网络适合序列数据，"
    "而注意力机制与 Transformer 架构则是现代大语言模型的基石。"
    "随着显卡算力的增长与海量数据的积累，深度学习在语音识别、机器翻译、"
    "自动驾驶等任务上取得了接近甚至超越人类水平的表现。"
)


# =============================================================================
# 演示函数
# =============================================================================

def demo_structure(client):
    """演示父子切块的层级结构（数据从 Milvus 同一张表按 level 读回）"""
    print(f"\n-- 示例 2: 父子切块结构可视化（读取 Milvus）")

    total = client.get_collection_stats(CHUNK_COLLECTION).get("row_count", 0)
    parent_rows = client.query(collection_name=CHUNK_COLLECTION,
                               filter='level == "parent"', output_fields=["chunk_id"])
    n_parents = len(parent_rows)
    n_children = total - n_parents

    print(f"文档总长：{len(DEMO_DOC)} 字")
    print(f"父块参数：chunk_size={PARENT_SIZE}, overlap={PARENT_OVERLAP}")
    print(f"子块参数：chunk_size={CHILD_SIZE}, overlap={CHILD_OVERLAP}")
    print(f"入库结果：共 {total} 行 = {n_parents} 个父块（level=parent）"
          f"+ {n_children} 个子块（level=child）\n")

    print("父子层级结构（仅展示前 3 个父块）：")
    parents = client.query(
        collection_name=CHUNK_COLLECTION,
        filter='level == "parent"',
        output_fields=["text"],
        limit=3,
    )
    for p in parents:
        pid, text = p["chunk_id"], p["text"]
        print(f"【父块 {pid}】({len(text)} 字) {text[:30]}...")
        children = client.query(
            collection_name=CHUNK_COLLECTION,
            filter=f'level == "child" and parent_id == {pid}',
            output_fields=["text"],
        )
        for c in children:
            print(f"    └─ 子块 {c['chunk_id']} ({len(c['text'])} 字) {c['text'][:26]}...")
        print()

    print("要点：同一张表靠 level 字段区分父子；")
    print("      父块行只存原文（占位向量，永不参与检索），子块行存真实向量。")


def demo_retrieve(client, emb):
    """演示完整检索链路：子块命中（精准）→ 父块回填（完整）

    只展示检索与回填效果，不调用大模型生成答案。
    """
    print(f"\n-- 示例 3: 子块检索（精准）→ 父块回填（完整）")

    questions = [
        "Milvus 向量数据库支持哪几种检索方式？",
        "深度学习里有哪些典型的神经网络结构？",
    ]

    for q_no, question in enumerate(questions, 1):
        print(f"\n【问题 {q_no}】{question}")

        # ③ 子块检索：小而聚焦，用来精准定位
        children, parents = retrieve_with_parent(client, emb, question, top_k=TOP_K)

        print(f"第一步：子块向量检索（top {TOP_K}，filter='level == \"child\"'，小而精准）：")
        for rank, hit in enumerate(children, 1):
            entity = hit["entity"]
            # 注意：Hit 对象的主键存在"主键字段名"（chunk_id）下，不是 "id" 键
            print(f"  {rank}. 子块 {entity['chunk_id']}  score={hit['distance']:.4f}  属于父块 {entity['parent_id']}")
            print(f"     内容：{entity['text'][:40]}...")

        # ④ 回填父块：大而完整，可直接作为上下文交给下游使用
        print(f"\n第二步：按 parent_id 回填父块（去重后 {len(parents)} 个）：")
        for i, p in enumerate(parents, 1):
            print(f"  {i}. ({len(p)} 字) {p[:50]}...")

        if q_no == len(questions):
            print(f"\n对比：命中的子块只有 {CHILD_SIZE} 字左右（定位准），")
            print(f"      回填的父块有 {PARENT_SIZE} 字左右（上下文全），")
            print(f"      这就是 Small-to-Big：检索用小，返回用大。")


def parent_child_summary():
    """父子切块的要点与参数建议总结"""
    print(f"\n-- 示例 4: 父子切块总结")

    print(f"""
父子切块（Parent-Child / Small-to-Big）— Milvus 单集合实操版

  单集合设计（{CHUNK_COLLECTION}）：
    一张表装下父块行和子块行，靠 level 字段区分：
      level=parent：原文 + 占位向量（Milvus 强制要求向量字段，父块永不参与检索）
      level=child ：原文 + 真实向量（text-embedding-v4）+ parent_id 关联
    检索：filter='level == "child"' 只搜子块行
    回填：filter='level == "parent" and chunk_id in [...]' 取回父块原文

  优点：
    ✓ 检索精准：子块小、语义聚焦，向量不被稀释
    ✓ 上下文完整：回填父块，信息全面
    ✓ 两全其美：同时解决"切太大检索不准"和"切太小上下文不全"

  代价：
    ✗ 存储翻倍：父块 + 子块都要入库
    ✗ 工程复杂：需要维护 parent_id 关联与回填逻辑（本文件已完整实现）

  生产参数参考：
    父块：chunk_size=1024, overlap=100
    子块：chunk_size=512,  overlap=50

  下一步：把回填的父块拼进提示词交给大模型生成答案，见 05_rag_pipeline/
""")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":

    # ---- 示例 1: 连接 Milvus + 建集合 + 父子切块入库 ----
    print(f"\n-- 示例 1: 连接 Milvus 并完成父子切块入库")
    print(f"Milvus 地址：{MILVUS_URI}")
    client = MilvusClient(uri=MILVUS_URI)
    print("✓ Milvus 连接成功")

    emb = AliyunEmbedding()  # 阿里云 text-embedding-v4，1024 维

    create_collection(client)                                   # 建单集合（父子同表）
    index_parent_child(client, emb, DEMO_DOC)                   # 切块 → 向量化 → 入库

    # ---- 检索演示：子块命中 → 父块回填（不调用大模型）----
    demo_structure(client)         # 示例 2: 父子层级结构（读回 Milvus）
    demo_retrieve(client, emb)     # 示例 3: 子块检索 → 父块回填
    # parent_child_summary()         # 示例 4: 总结

    print("=" * 60)
    print("  小块检索精准 + 大块上下文完整 = 父子切块的核心价值")
    print("  下一步：03_retrieval_methods/（检索方法）")
    print("=" * 60)
