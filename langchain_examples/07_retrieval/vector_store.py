# =============================================================================
# 向量存储 — Embedding（文字→数字指纹）+ 语义搜索
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解 Embedding：把文字转成数字向量，语义相近→向量相近
#   ✅ 使用 FAISS 向量数据库做语义检索
# =============================================================================

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)


"""
什么是 Embedding（嵌入）？

  传统搜索: 找包含"苹果"这个词的文章（关键词匹配）
  向量搜索: 找和"苹果"语义接近的内容（包括"水果"、"iPhone"）

  原理: Embedding 模型把文字转成数字向量
    "苹果" → [0.8, 0.2, -0.5, 0.3, ...]
    "香蕉" → [0.7, 0.3, -0.4, 0.2, ...]  ← 向量相近（都是水果）
    "手机" → [0.3, 0.8,  0.5,-0.2, ...]  ← 向量差异大（不同类别）

  生活化比喻: Embedding = "文字指纹"
    每个人的指纹独一无二 → 每个词的含义对应独特的数字向量
    相似的人的指纹也有相似特征 → 语义相近的词向量也接近
"""


# =============================================================================
# 示例 1: 用模拟向量理解 Embedding 概念
# =============================================================================

def understand_embedding():
    """
    用模拟数据直观展示"语义相近→向量相近"的原理。

    不需要安装任何额外依赖，直接运行即可理解核心概念。
    """
    print(f"\n-- 示例 1: 理解 Embedding 概念（模拟演示）")

    # 模拟的向量（实际中由 Embedding 模型生成）
    mock = {
        "苹果": [0.8, 0.2, -0.5, 0.3],
        "香蕉": [0.7, 0.3, -0.4, 0.2],
        "手机": [0.3, 0.8, 0.5, -0.2],
        "水果": [0.9, 0.1, -0.6, 0.4],
    }

    def similarity(v1, v2):
        """简化的向量相似度计算（余弦相似度原理）。"""
        return sum(a * b for a, b in zip(v1, v2))

    print("向量表示:")
    for word, vec in mock.items():
        print(f"  {word:4} → {vec}")

    print('\n"苹果"与其他词的相似度:')
    for word, vec in mock.items():
        sim = similarity(mock["苹果"], vec)
        print(f"  苹果 ↔ {word:4} = {sim:.2f}")

    print('\n结论: "苹果"和"香蕉"(都是水果) 相似度 > "苹果"和"手机"(不同类别)')


# =============================================================================
# 示例 2: 使用 FAISS 向量数据库做真实检索
# =============================================================================

def faiss_vector_search():
    """
    用 FAISS（Facebook 开源的向量搜索库）做真正的语义检索。

    需要安装: pip install faiss-cpu langchain-huggingface

    流程: Document → Embedding(向量化) → FAISS 存储 → 语义搜索
    """
    print(f"\n-- 示例 2: FAISS 向量检索（需要额外依赖）")

    try:
        from langchain_core.documents import Document
        from langchain_community.embeddings import DashScopeEmbeddings
        from langchain_community.vectorstores import FAISS
        import os
        from dotenv import load_dotenv

        load_dotenv()
        os.environ["DASHSCOPE_API_KEY"] = os.getenv("ALIYUN_API_KEY", "")

        if not os.environ["DASHSCOPE_API_KEY"]:
            print("⚠️ 需要设置 ALIYUN_API_KEY 环境变量")
            return

        # 准备知识库文档
        docs = [
            Document(page_content="Python 是一种高级编程语言", metadata={"source": "prog"}),
            Document(page_content="机器学习让计算机从数据中学习", metadata={"source": "ai"}),
            Document(page_content="深度学习使用神经网络", metadata={"source": "ai"}),
            Document(page_content="苹果是一种常见水果，富含维生素", metadata={"source": "food"}),
            Document(page_content="香蕉含有丰富的钾元素", metadata={"source": "food"}),
        ]

        embeddings = DashScopeEmbeddings(model="text-embedding-v3")
        store = FAISS.from_documents(docs, embeddings)

        # 语义搜索: "人工智能相关的技术" 应该找到 ai 来源的文档
        query = "人工智能相关的技术"
        results = store.similarity_search(query, k=3)

        print(f"搜索: '{query}'\n")
        for i, doc in enumerate(results):
            print(f"  [{i + 1}] 来源: {doc.metadata['source']}")
            print(f"      内容: {doc.page_content}\n")

    except ImportError:
        print("⚠️ 需要安装: pip install faiss-cpu langchain-huggingface langchain-community")
        print("  跳过此示例（示例 1 已演示核心概念）")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n>>> 07_retrieval/vector_store — 向量存储与语义搜索\n")

    understand_embedding()
    faiss_vector_search()

    # 接下来学习: rag_basic.py（RAG 基础）
