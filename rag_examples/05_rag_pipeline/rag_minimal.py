# =============================================================================
# RAG 最小可运行版 - 50 行代码理解 RAG
# =============================================================================
#  
# 用途：用最简单的代码演示 RAG 核心流程
#
# 核心流程：
#   文档 → 切片 → Embedding → Milvus → 检索 → 问答
# =============================================================================

# 导入 Milvus 配置（优先使用 Docker Milvus）
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from milvus_config import MILVUS_URI


# =============================================================================
# RAG 最小封装类（约 50 行核心代码）
# =============================================================================

class SimpleRAG:
    """最简单的 RAG 实现"""

    def __init__(self, milvus_uri=MILVUS_URI, collection_name="simple_rag"):
        from pymilvus import MilvusClient

        self.client = MilvusClient(uri=milvus_uri)
        self.collection_name = collection_name

        # 简化：用随机向量模拟 Embedding（实际应该用真实模型）
        import random
        random.seed(42)
        self.dim = 768

    def _random_embedding(self, text):
        """模拟 Embedding（实际应该用真实模型）"""
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
