# =============================================================================
# RAG 分步详解 - 每一步的详细逻辑
# =============================================================================
#  
# 用途：逐步讲解 RAG 流程的每个细节
#
# 适合：想深入理解每个步骤原理的学习者
# =============================================================================

# 导入 Milvus 配置（优先使用 Docker Milvus）
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from milvus_config import MILVUS_URI


# =============================================================================
# 步骤 1: 文档加载
# =============================================================================

def step1_load_documents():
    """
    步骤 1: 加载文档

    支持的文件格式:
    - .txt  纯文本文件
    - .json JSON 格式文件
    - .md   Markdown 文件
    """
    print("=" * 60)
    print("步骤 1: 文档加载")
    print("=" * 60)

    # 示例：加载纯文本文件
    sample_docs = [
        "人工智能是模拟人类智能的计算机科学领域。",
        "机器学习通过训练数据让计算机自动学习规律。",
        "深度学习使用多层神经网络，在图像识别领域取得成功。",
        "自然语言处理让计算机理解和生成人类语言。",
        "计算机视觉让计算机能够看懂图像和视频。"
    ]

    print(f"加载了 {len(sample_docs)} 个文档")
    for i, doc in enumerate(sample_docs[:3]):
        print(f"  [{i+1}] {doc[:30]}...")

    return sample_docs


# =============================================================================
# 步骤 2: 文档切片
# =============================================================================

def step2_document_chunking(documents, chunk_size=100, chunk_overlap=20):
    """
    步骤 2: 文档切片

    将长文档切分成小片段，适合 Embedding 处理

    参数:
        documents: 原始文档列表
        chunk_size: 每个切片的大小
        chunk_overlap: 切片之间的重叠
    """
    print()
    print("=" * 60)
    print("步骤 2: 文档切片")
    print("=" * 60)
    print(f"切片参数：chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    chunks = []
    for doc in documents:
        # 简单按字符切片（实际应该按语义切片）
        for i in range(0, len(doc), chunk_size - chunk_overlap):
            chunk = doc[i:i + chunk_size]
            if chunk:
                chunks.append(chunk)

    print(f"切片完成：{len(documents)} 个文档 → {len(chunks)} 个切片")
    return chunks


# =============================================================================
# 步骤 3: Embedding 向量化
# =============================================================================

def step3_embedding(chunks):
    """
    步骤 3: Embedding 向量化

    将文本切片转换为向量表示

    注：这里用随机向量模拟，实际应该用真实的 Embedding 模型
    """
    print()
    print("=" * 60)
    print("步骤 3: Embedding 向量化")
    print("=" * 60)

    import random
    random.seed(42)

    dim = 768  # 向量维度
    embeddings = []

    for chunk in chunks:
        # 模拟 Embedding（实际应该用真实模型）
        random.seed(hash(chunk) % 10000)
        embedding = [random.uniform(-1, 1) for _ in range(dim)]
        embeddings.append(embedding)

    print(f"向量化完成：{len(chunks)} 个切片 → {dim} 维向量")
    return embeddings


# =============================================================================
# 步骤 4: 存入 Milvus
# =============================================================================

def step4_store_to_milvus(chunks, embeddings, collection_name="rag_demo"):
    """
    步骤 4: 存入 Milvus 向量数据库

    参数:
        chunks: 文本切片列表
        embeddings: 向量列表
        collection_name: Collection 名称
    """
    from pymilvus import MilvusClient

    print()
    print("=" * 60)
    print("步骤 4: 存入 Milvus")
    print("=" * 60)

    client = MilvusClient(uri=MILVUS_URI)

    # 建表前先删表
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    # 创建 Collection
    client.create_collection(
        collection_name=collection_name,
        dimension=768,
        auto_id=True,
        metric_type="COSINE"
    )

    # 批量插入
    data = []
    for chunk, emb in zip(chunks, embeddings):
        data.append({
            "content": chunk,
            "vector": emb
        })

    client.insert(collection_name=collection_name, data=data)

    print(f"存入完成：{len(chunks)} 条记录")
    return client, collection_name


# =============================================================================
# 步骤 5: 向量检索
# =============================================================================

def step5_vector_search(client, collection_name, query, top_k=3):
    """
    步骤 5: 向量检索

    参数:
        client: MilvusClient
        collection_name: Collection 名称
        query: 查询文本
        top_k: 返回最相似的 K 个结果
    """
    import random

    print()
    print("=" * 60)
    print("步骤 5: 向量检索")
    print("=" * 60)
    print(f"查询：{query}")

    # 模拟查询向量
    random.seed(hash(query) % 10000)
    query_vector = [random.uniform(-1, 1) for _ in range(768)]

    # 执行检索
    results = client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=top_k,
        output_fields=["content"]
    )

    print(f"\n检索结果（Top-{top_k}）：")
    for i, hit in enumerate(results[0]):
        print(f"  [{i+1}] 相似度：{hit['distance']:.4f}")
        print(f"      内容：{hit['entity']['content'][:50]}...")

    return results


# =============================================================================
# 步骤 6: RAG 问答
# =============================================================================

def step6_rag_qna(contexts, question):
    """
    步骤 6: RAG 问答

    基于检索到的上下文生成答案

    参数:
        contexts: 检索到的相关文档
        question: 用户问题
    """
    print()
    print("=" * 60)
    print("步骤 6: RAG 问答")
    print("=" * 60)
    print(f"问题：{question}")

    # 构建 Prompt
    context_text = ""
    for i, ctx in enumerate(contexts):
        if isinstance(ctx, dict):
            content = ctx.get('entity', {}).get('content', str(ctx))
        else:
            content = str(ctx)
        context_text += f"[{i+1}] {content}\n"

    prompt = f"""
请根据以下信息回答问题：

相关信息：
{context_text}

问题：{question}

要求：
1. 基于上述信息回答
2. 如果信息不足，说明无法回答
"""

    print("\n构建的 Prompt:")
    print("-" * 40)
    print(prompt[:200] + "...")

    # 注：实际应该调用 LLM 生成答案
    # 这里只是演示流程
    print("\n[注：实际应该调用 LLM 生成答案]")

    return prompt


# =============================================================================
# 完整流程总结
# =============================================================================

def full_pipeline_summary():
    """
    完整流程总结
    """
    print()
    print("=" * 60)
    print("RAG 流程总结")
    print("=" * 60)

    print("""
┌─────────────────────────────────────────────────────────┐
│ RAG 完整流程                                             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  文档加载 → 文档切片 → Embedding → 存入 Milvus          │
│                                            ↓            │
│  用户问题 → Embedding → 向量检索 → 检索结果            │
│                                            ↓            │
│  构建 Prompt → LLM 生成 → 返回答案                      │
│                                                         │
└─────────────────────────────────────────────────────────┘

关键步骤说明:

1. 文档加载：读取各种格式的文档
2. 文档切片：切分成适合处理的小片段
3. Embedding: 将文本转换为向量
4. 存入 Milvus: 保存到向量数据库
5. 向量检索：根据问题查找相关文档
6. RAG 问答：基于上下文生成答案
""")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  RAG 分步详解 - 逐步讲解每个步骤")
    print("  说明：深入理解 RAG 流程的每个环节")
    print("=" * 70 + "\n")

    # 运行所有步骤
    documents = step1_load_documents()

    chunks = step2_document_chunking(documents)

    embeddings = step3_embedding(chunks)

    client, collection_name = step4_store_to_milvus(chunks, embeddings)

    results = step5_vector_search(client, collection_name, "什么是机器学习？")

    step6_rag_qna(results[0], "什么是机器学习？")

    full_pipeline_summary()

    print("\n" + "=" * 70)
    print("  RAG 分步学习完成！")
    print("=" * 70 + "\n")
