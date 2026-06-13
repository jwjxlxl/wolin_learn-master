# =============================================================================
# RAG 最小可运行版 - 50 行代码理解 RAG
# =============================================================================
#  
# 用途：用最简单的代码演示 RAG 核心流程
#
# 核心流程：
#   文档 → 切片 → Embedding → Milvus → 检索 → 问答
# =============================================================================

# 从项目根目录（wolin_learn-master/）运行本文件：
#   python rag_examples/05_rag_pipeline/rag_minimal.py
from rag_examples.milvus_config import MILVUS_URI, DEFAULT_DIMENSION


# =============================================================================
# RAG 最小封装类（约 50 行核心代码）
# =============================================================================

class SimpleRAG:
    """最简单的 RAG 实现

    ⚠️ 重要说明：本类使用随机模拟向量（mock embedding）仅用于演示 RAG 流程。
    在实际项目中，必须替换为真实的 Embedding 模型。详见：

    - embedding_examples/02_aliyun_embedding.py — 阿里云百炼 API 的真实 Embedding 调用
    - embedding_examples/03_local_embedding.py — 本地 sentence-transformers 模型
    - 05_rag_pipeline/rag_full_pipeline.py — 使用了真实 Embedding 的完整 RAG 实现
    """

    def __init__(self, milvus_uri=MILVUS_URI, collection_name="simple_rag"):
        from pymilvus import MilvusClient

        self.client = MilvusClient(uri=milvus_uri)
        self.collection_name = collection_name

        # 使用配置文件中的默认维度（当前：text-embedding-v4 = 1024 维）
        self.dim = DEFAULT_DIMENSION

    def _random_embedding(self, text):
        """模拟 Embedding 生成（仅供演示，实际项目请替换为真实 Embedding 模型）

        ⚠️ 此处使用随机向量仅为演示 RAG 流程，检索结果无实际语义意义。
        替换方法：将本函数替换为调用真实 Embedding API，例如：

            from rag_examples.embedding_examples.02_aliyun_embedding import AliyunEmbedding
            embedder = AliyunEmbedding()
            return embedder.embed(text)

        或使用阿里云 DashScope：

            from openai import OpenAI
            client = OpenAI(api_key="...", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
            return client.embeddings.create(model="text-embedding-v4", input=text).data[0].embedding
        """
        import random
        random.seed(hash(text) % 10000)
        return [random.uniform(-1, 1) for _ in range(self.dim)]

    def add_documents(self, documents):
        """
        添加文档到知识库

        参数:
            documents: 文档列表，每个文档是字符串
        """
        # 确保 Collection 存在
        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.dim,
                auto_id=True,
                metric_type="COSINE"
            )

        # 生成向量并插入
        data = []
        for doc in documents:
            data.append({
                "content": doc,
                "vector": self._random_embedding(doc)
            })

        self.client.insert(
            collection_name=self.collection_name,
            data=data
        )

        print(f"[OK] 已添加 {len(documents)} 个文档")

    def ask(self, question, top_k=3):
        """
        问答接口

        参数:
            question: 用户问题
            top_k: 返回最相似的 K 个文档
        返回:
            最相似的文档列表
        """
        # 生成查询向量
        query_vector = self._random_embedding(question)

        # 向量检索
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=top_k,
            output_fields=["content"]
        )

        # 提取结果
        contexts = []
        for hit in results[0]:
            contexts.append({
                "content": hit["entity"]["content"],
                "distance": hit["distance"]
            })

        return contexts


# =============================================================================
# 示例用法
# =============================================================================

def demo_basic_usage():
    """演示基本用法"""
    print("=" * 60)
    print("RAG 最小可运行版 - 基本演示")
    print("=" * 60)

    # 初始化
    rag = SimpleRAG(milvus_uri=MILVUS_URI)

    # 测试文档
    documents = [
        "人工智能是模拟人类智能的计算机科学领域。",
        "机器学习通过训练数据让计算机自动学习规律。",
        "深度学习使用多层神经网络，在图像识别领域取得成功。",
        "自然语言处理让计算机理解和生成人类语言。",
        "计算机视觉让计算机能够看懂图像和视频。"
    ]

    # 添加文档
    print("\n1. 添加文档到知识库")
    rag.add_documents(documents)

    # 提问
    print("\n2. 提问：什么是机器学习？")
    results = rag.ask("什么是机器学习？")

    print("\n检索到的相关文档：")
    for i, result in enumerate(results):
        print(f"  [{i+1}] 相似度：{result['distance']:.4f}")
        print(f"      内容：{result['content']}")

    return rag


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  RAG 最小可运行版")
    print("  说明：50 行代码理解 RAG 核心流程")
    print("=" * 70 + "\n")

    demo_basic_usage()

    print("\n" + "=" * 70)
    print("  RAG 学习完成！")
    print("=" * 70 + "\n")
