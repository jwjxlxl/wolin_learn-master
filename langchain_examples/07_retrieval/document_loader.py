# =============================================================================
# 文档加载器
# =============================================================================
#  
# 用途：教学演示 - 使用 DocumentLoader 加载各种格式的文档
#
# 核心概念：
#   - 如何读取 PDF、Word、TXT 文档
#   - 文档分块（Chunking）
# =============================================================================

# -----------------------------------------------------------------------------
# 运行前检查
# -----------------------------------------------------------------------------
# 本示例主要演示概念，不需要 Ollama 服务
# 如需实际加载文档，需要安装：
#   pip install pypdf python-docx
# -----------------------------------------------------------------------------

# 设置 UTF-8 编码（Windows 专用）
import sys
import io
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer,
    encoding='utf-8',
    errors='replace',
    line_buffering=True
)


# =============================================================================
# 第一部分：理解文档加载
# =============================================================================
"""
为什么要加载文档？

📚 场景
   你有一份 PDF 产品手册，想让 AI 基于手册内容回答用户问题

🔧 步骤
   1. 加载文档（读取文件）
   2. 分割文档（切成小块）
   3. 存储到向量数据库
   4. 检索相关片段
   5. 交给 AI 生成答案

💡 生活化比喻
   文档加载 = "把书读进电脑"
   文档分块 = "把厚书拆成章节"
"""


# =============================================================================
# 示例 1: 加载 TXT 文档
# =============================================================================

def load_txt_document():
    """
    加载 TXT 文本文档

    最简单的文档格式
    """
    print("=" * 60)
    print("示例 1: 加载 TXT 文档")
    print("=" * 60)

    from langchain_community.document_loaders import TextLoader
    import os

    # 创建一个测试文件（使用绝对路径）
    test_file = os.path.join(os.path.dirname(__file__), "test_doc.txt")
    try:
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("""
人工智能（Artificial Intelligence，简称 AI）是计算机科学的一个分支，
致力于让计算机具有人类的智能特征，如学习、推理、感知、理解语言等。

机器学习是人工智能的核心技术之一，它让计算机通过数据学习规律，
而不需要显式编程。深度学习是机器学习的一个子领域，使用神经网络
模拟人脑的工作方式。

AI 的应用非常广泛，包括：
- 图像识别：人脸识别、医学影像分析
- 自然语言处理：机器翻译、智能客服
- 推荐系统：电商推荐、新闻推荐
- 自动驾驶：无人驾驶汽车
""")
        print(f"已创建测试文件：{test_file}")
    except Exception as e:
        print(f"创建测试文件失败：{e}")
        return

    # 加载文档
    try:
        loader = TextLoader(test_file, encoding='utf-8')
        documents = loader.load()

        print(f"\n加载成功！")
        print(f"文档数量：{len(documents)}")
        print(f"第一个文档页数：{len(documents[0].page_content)} 字符")
        print(f"\n文档内容（前 100 字）：")
        print(documents[0].page_content[:100])
        print("...")

    except Exception as e:
        print(f"加载失败：{e}")
    print()


# =============================================================================
# 示例 2: 文档分块（Chunking）
# =============================================================================

def chunk_document():
    """
    使用 RecursiveCharacterTextSplitter 分割文档

    长文档需要切成小块，方便后续检索
    """
    print("=" * 60)
    print("示例 2: 文档分块（Chunking）")
    print("=" * 60)

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # 创建测试文本
    text = """
人工智能（Artificial Intelligence，简称 AI）是计算机科学的一个分支，
致力于让计算机具有人类的智能特征。机器学习是人工智能的核心技术之一，
它让计算机通过数据学习规律。深度学习是机器学习的一个子领域，
使用神经网络模拟人脑的工作方式。

AI 的应用非常广泛，包括图像识别、自然语言处理、推荐系统、自动驾驶等。
图像识别用于人脸识别和医学影像分析。自然语言处理用于机器翻译和智能客服。
推荐系统用于电商和新闻推荐。自动驾驶用于无人驾驶汽车。

人工智能的发展经历了几个阶段：符号主义、连接主义、深度学习。
符号主义认为智能可以通过符号操作实现。连接主义认为应该模拟神经网络。
深度学习结合了两者优点，在多个领域取得突破性进展。

未来 AI 将继续发展，在医疗、教育、交通等领域发挥更大作用。
同时也带来了一些挑战，如就业影响、隐私保护、算法偏见等。
我们需要在发展和治理之间找到平衡。
"""

    # 创建文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,      # 每块最大 100 字符
        chunk_overlap=20,    # 块之间重叠 20 字符（保持上下文）
        length_function=len
    )

    # 分割文本
    chunks = text_splitter.split_text(text)

    print(f"原文长度：{len(text)} 字符")
    print(f"分割后块数：{len(chunks)}")
    print()

    for i, chunk in enumerate(chunks, 1):
        print(f"【块 {i}】({len(chunk)} 字符)")
        print(chunk.strip())
        print()


# =============================================================================
# 示例 3: 从字符串创建文档
# =============================================================================

def create_documents_from_text():
    """
    直接从字符串创建文档对象

    适合硬编码知识内容
    """
    print("=" * 60)
    print("示例 3: 从字符串创建文档")
    print("=" * 60)

    from langchain_core.documents import Document

    # 创建文档列表（模拟知识库）
    documents = [
        Document(
            page_content="Python 是一种高级编程语言，由 Guido van Rossum 于 1989 年发明。",
            metadata={"source": "python_intro", "topic": "programming"}
        ),
        Document(
            page_content="机器学习是让计算机从数据中学习规律的技术，无需显式编程。",
            metadata={"source": "ml_intro", "topic": "ai"}
        ),
        Document(
            page_content="深度学习使用神经网络模拟人脑，在图像识别等领域效果很好。",
            metadata={"source": "dl_intro", "topic": "ai"}
        ),
        Document(
            page_content="大语言模型（LLM）是基于大量文本训练的深度学习模型。",
            metadata={"source": "llm_intro", "topic": "ai"}
        ),
    ]

    print(f"创建了 {len(documents)} 个文档：\n")

    for i, doc in enumerate(documents, 1):
        print(f"{i}. 来源：{doc.metadata['source']}")
        print(f"   主题：{doc.metadata['topic']}")
        print(f"   内容：{doc.page_content[:50]}...")
        print()


# =============================================================================
# 示例 4: 实用的文档加载函数
# =============================================================================

def practical_document_loader():
    """
    实用的文档加载函数

    封装常用功能
    """
    print("=" * 60)
    print("示例 4: 实用的文档加载函数")
    print("=" * 60)

    from langchain_core.documents import Document

    def load_knowledge_base():
        """
        加载知识库

        实际应用中可以从文件/数据库加载
        """
        return [
            Document(
                page_content="Qwen 是阿里云开发的大语言模型系列，包括 Qwen-Plus、Qwen-Max 等版本。",
                metadata={"source": "qwen"}
            ),
            Document(
                page_content="DeepSeek 是深度求索开发的大语言模型，以高性价比著称。",
                metadata={"source": "deepseek"}
            ),
            Document(
                page_content="Ollama 是本地运行大模型的工具，支持多种开源模型。",
                metadata={"source": "ollama"}
            ),
            Document(
                page_content="LangChain 是构建 AI 应用的框架，提供标准化工具。",
                metadata={"source": "langchain"}
            ),
        ]

    def search_documents(documents, query):
        """
        简单搜索（关键词匹配）

        实际应用中会用向量检索
        """
        results = []
        for doc in documents:
            if query.lower() in doc.page_content.lower():
                results.append(doc)
        return results

    # 加载知识库
    kb = load_knowledge_base()
    print(f"知识库包含 {len(kb)} 个文档\n")

    # 搜索
    queries = ["阿里", "本地", "框架"]
    for query in queries:
        results = search_documents(kb, query)
        print(f"搜索 '{query}': 找到 {len(results)} 个结果")
        for doc in results:
            print(f"  - {doc.metadata['source']}: {doc.page_content[:30]}...")
    print()


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  文档加载器 - Document Loader")
    print("  说明：加载和分割文档")
    print("=" * 70 + "\n")

    print("【说明】")
    print("  本示例主要演示概念，部分功能需要额外依赖")
    print()

    # 运行示例
    load_txt_document()
    chunk_document()
    create_documents_from_text()
    practical_document_loader()

    print("=" * 70)
    print("  接下来学习：vector_store.py（向量存储）")
    print("=" * 70 + "\n")
