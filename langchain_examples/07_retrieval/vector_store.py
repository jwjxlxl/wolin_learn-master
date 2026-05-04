# =============================================================================
# 向量存储
# =============================================================================
#  
# 用途：教学演示 - 使用向量数据库存储和检索文档
#
# 核心概念：
#   - Embedding = "文字转数字指纹"
#   - 向量数据库 = "按相似度搜索"
#   - 相似度 = "语义接近程度"
# =============================================================================

# -----------------------------------------------------------------------------
# 运行前检查
# -----------------------------------------------------------------------------
# 本示例主要演示概念，使用 FAISS 作为向量数据库
# 如需实际运行，需要安装：
#   pip install faiss-cpu langchain-huggingface
# -----------------------------------------------------------------------------

# 设置 UTF-8 编码（Windows 专用）
import sys
import io

from langchain_community.embeddings import DashScopeEmbeddings

sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer,
    encoding='utf-8',
    errors='replace',
    line_buffering=True
)



# =============================================================================
# 第一部分：理解向量存储
# =============================================================================
"""
什么是向量存储？

📊 Embedding（嵌入）
   把文字转换成数字向量
   "苹果" → [0.1, 0.5, -0.3, 0.8, ...]

   💡 生活化比喻
   Embedding = "文字指纹"
   每个人的指纹独一无二 → 每个词的含义对应独特向量

🔍 向量搜索
   传统搜索：找包含"苹果"这个词的文章
   向量搜索：找和"苹果"语义接近的内容
            （包括"水果"、"iPhone"、"科技"等）

   💡 生活化比喻
   向量搜索 = "以词搜词"
   就像找"相似的人"，不是找"穿同样衣服的人"

📦 向量数据库
   存储大量文档的向量表示
   支持快速相似度搜索
"""


# =============================================================================
# 示例 1: 理解 Embedding
# =============================================================================

def understand_embedding():
    """
    理解 Embedding 的概念

    演示如何将文本转换为向量
    """
    print("=" * 60)
    print("示例 1: 理解 Embedding（文字转数字指纹）")
    print("=" * 60)

    # 模拟 Embedding 输出
    # 实际应用中会调用嵌入模型生成
    mock_embeddings = {
        "苹果": [0.8, 0.2, -0.5, 0.3],
        "香蕉": [0.7, 0.3, -0.4, 0.2],
        "手机": [0.3, 0.8, 0.5, -0.2],
        "电脑": [0.2, 0.9, 0.4, -0.3],
        "水果": [0.9, 0.1, -0.6, 0.4],
        "科技": [0.1, 0.7, 0.6, -0.1],
    }

    print("模拟的文本向量表示：\n")

    for text, vector in mock_embeddings.items():
        print(f"  {text:6} → {vector}")

    print("\n💡 观察：")
    print("  - '苹果'和'香蕉'的向量比较接近（都是水果）")
    print("  - '苹果'和'水果'的向量也很接近（语义相关）")
    print("  - '苹果'和'手机'的向量差异较大（不同类别）")
    print()

    # 计算简单相似度（余弦相似度简化版）
    def simple_similarity(v1, v2):
        """计算两个向量的简单相似度"""
        return sum(a * b for a, b in zip(v1, v2))

    print("计算'苹果'与其他词的相似度：")
    apple_vec = mock_embeddings["苹果"]

    for text, vec in mock_embeddings.items():
        sim = simple_similarity(apple_vec, vec)
        print(f"  苹果 ↔ {text:6} = {sim:.4f}")

    print("\n相似度越高，表示语义越接近！")
    print()


# =============================================================================
# 示例 2: 使用 DashScope 模型 生成 Embedding
# =============================================================================

def generate_embeddings():
    """
    """
    print("=" * 60)
    print("示例 2: 生成真实的 Embedding")
    print("=" * 60)

    try:
        from sentence_transformers import SentenceTransformer

        # 加载模型
        print("加载模型...（首次运行会自动下载）")
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        # 要编码的文本
        texts = [
            "我喜欢吃苹果",
            "香蕉是一种水果",
            "手机是科技产品",
            "苹果电脑很贵",
            "今天天气不错",
        ]

        # 生成向量
        print("生成向量表示...\n")
        embeddings = model.encode(texts)

        for i, (text, emb) in enumerate(zip(texts, embeddings), 1):
            print(f"{i}. {text}")
            print(f"   向量维度：{len(emb)}")
            print(f"   向量前 10 维：{emb[:10].round(3)}")
            print()

        # 计算相似度
        from sklearn.metrics.pairwise import cosine_similarity

        print("计算相似度矩阵：")
        sim_matrix = cosine_similarity([embeddings[0]], embeddings)[0]

        for i, sim in enumerate(sim_matrix):
            print(f"  '我喜欢吃苹果' ↔ '{texts[i]}' = {sim:.4f}")

    except ImportError:
        print("需要安装依赖：")
        print("  pip install sentence-transformers scikit-learn")
        print("\n跳过此示例，继续学习其他部分")

    print()


# =============================================================================
# 示例 3: 使用 FAISS 存储向量
# =============================================================================

def faiss_vector_store():
    """
    使用 FAISS 向量数据库存储和检索

    FAISS = Facebook AI Similarity Search
    """
    print("=" * 60)
    print("示例 3: FAISS 向量存储")
    print("=" * 60)

    try:
        import faiss, os
        import numpy as np
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_core.documents import Document
        from langchain_community.vectorstores import FAISS

        # 准备文档
        documents = [
            Document(page_content="Python 是一种高级编程语言", metadata={"source": "prog"}),
            Document(page_content="机器学习让计算机从数据中学习", metadata={"source": "ai"}),
            Document(page_content="深度学习使用神经网络", metadata={"source": "ai"}),
            Document(page_content="大语言模型基于大量文本训练", metadata={"source": "ai"}),
            Document(page_content="苹果是一种常见水果", metadata={"source": "food"}),
            Document(page_content="香蕉含有丰富的钾元素", metadata={"source": "food"}),
        ]

        print("加载嵌入模型...")
        # embeddings = HuggingFaceEmbeddings(
        #     model_name="paraphrase-multilingual-MiniLM-L12-v2"
        # )

        # 从.env文件中读取API密钥
        # 把秘钥写入环境变量
        # 实例化一个DashScopeEmbeddings对象
        os.environ["DASHSCOPE_API_KEY"] = os.getenv("ALIYUN_API_KEY")
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v3"
        )

        print("创建向量存储...")
        vectorstore = FAISS.from_documents(documents, embeddings)

        print(f"已存储 {len(documents)} 个文档\n")

        # 相似度搜索
        query = "人工智能相关的技术"
        print(f"搜索：'{query}'\n")

        results = vectorstore.similarity_search(query, k=3)

        print(f"找到 {len(results)} 个最相关的文档：\n")
        for i, doc in enumerate(results, 1):
            print(f"{i}. 来源：{doc.metadata['source']}")
            print(f"   内容：{doc.page_content}")
            print()

        # 保存和加载
        print("保存向量库到本地...")
        vectorstore.save_local("faiss_demo")
        print("已保存到 faiss_demo/ 目录\n")

        print("从本地加载向量库...")
        loaded_store = FAISS.load_local(
            "faiss_demo",
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("加载成功！")

    except ImportError as e:
        print("需要安装依赖：")
        print("  pip install faiss-cpu langchain-huggingface langchain-community")
        print(f"\n错误详情：{e}")
        print("\n跳过此示例")

    print()


# =============================================================================
# 示例 4: 向量搜索 vs 关键词搜索
# =============================================================================

def vector_vs_keyword_search():
    """
    对比向量搜索和关键词搜索的区别

    理解语义搜索的优势
    """
    print("=" * 60)
    print("示例 4: 向量搜索 vs 关键词搜索")
    print("=" * 60)

    # 文档库
    documents = [
        "苹果是一种水果，富含维生素 C",
        "苹果公司发布了新款 iPhone 手机",
        "香蕉是热带水果，口感软糯",
        "机器学习需要大量数据训练模型",
        "Python 编程语言广泛应用于 AI 领域",
        "智能手机是现代人的必备品",
    ]

    print("文档库：\n")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc}")
    print()

    # 关键词搜索
    query = "苹果"
    print(f"【关键词搜索】查找包含'{query}'的文档：\n")

    keyword_results = [doc for doc in documents if query in doc]
    for i, doc in enumerate(keyword_results, 1):
        print(f"  {i}. {doc}")

    if not keyword_results:
        print("  （无匹配结果）")

    print()

    # 向量搜索（模拟）
    print(f"【向量搜索】查找与'{query}'语义相关的文档：\n")

    # 模拟向量搜索结果（基于语义相似度）
    vector_results = [
        "苹果公司发布了新款 iPhone 手机",      # 都指苹果公司
        "苹果是一种水果，富含维生素 C",        # 字面匹配
        "智能手机是现代人的必备品",           # 产品相关
        "香蕉是热带水果，口感软糯",           # 水果相关
    ]

    for i, doc in enumerate(vector_results, 1):
        print(f"  {i}. {doc}")

    print("\n💡 向量搜索的优势：")
    print("  - 能找到语义相关但不包含关键词的文档")
    print("  - 区分多义词（苹果=水果 or 公司）")
    print("  - 支持模糊匹配和概念联想")
    print()


# =============================================================================
# 示例 5: 实用的向量检索函数
# =============================================================================

def practical_vector_search():
    """
    实用的向量检索封装

    展示完整的 RAG 检索流程
    """
    print("=" * 60)
    print("示例 5: 实用的向量检索函数")
    print("=" * 60)

    from langchain_core.documents import Document

    # 模拟向量检索结果
    def mock_similarity_search(query, documents, k=3):
        """
        模拟相似度搜索

        实际应用中会用真实的向量数据库
        """
        # 简单关键词匹配模拟
        results = []
        for doc in documents:
            score = sum(1 for word in query if word in doc.page_content)
            if score > 0:
                results.append((score, doc))

        # 按相似度排序
        results.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in results[:k]]

    # 知识库文档
    knowledge_base = [
        Document(page_content="Qwen 是阿里云开发的大语言模型系列"),
        Document(page_content="Ollama 是本地运行大模型的工具"),
        Document(page_content="LangChain 是构建 AI 应用的框架"),
        Document(page_content="深度学习使用神经网络模拟人脑"),
        Document(page_content="Python 是 AI 领域常用的编程语言"),
    ]

    # 检索函数
    def retrieve(query, k=2):
        """检索与问题最相关的文档"""
        return mock_similarity_search(query, knowledge_base, k)

    # 测试
    queries = [
        "阿里云的模型",
        "本地运行模型",
        "AI 框架",
    ]

    for query in queries:
        print(f"问题：{query}")
        results = retrieve(query)
        print(f"检索到 {len(results)} 个相关文档：")
        for doc in results:
            print(f"  - {doc.page_content}")
        print()


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  向量存储 - Vector Store")
    print("  说明：Embedding 和向量检索")
    print("=" * 70 + "\n")

    print("【说明】")
    print("  本示例主要演示概念，部分功能需要额外依赖")
    print()

    # 运行示例
    # understand_embedding()
    # generate_embeddings()  # 需要依赖可取消注释
    # faiss_vector_store()   # 需要依赖可取消注释
    # vector_vs_keyword_search()
    # practical_vector_search()

    faiss_vector_store()

    print("=" * 70)
    print("  接下来学习：rag_basic.py（RAG 基础示例）")
    print("=" * 70 + "\n")
