# =============================================================================
# 文档加载与分块 — 把长文档变成可检索的小片段
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解文档分块（Chunking）的意义
#   ✅ 使用 RecursiveCharacterTextSplitter 做递归切片
#   ✅ 从字符串创建 LangChain Document 对象
# =============================================================================

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)


"""
为什么需要文档分块（Chunking）？

  原始文档可能有几万字 → 直接塞给 LLM 会超出上下文限制
  解决: 切成小块，每块 200-500 字 → 只检索最相关的几块给 LLM

  生活化比喻: Chunking = 把厚书拆成章节
    要找"赤壁之战"的内容 → 不需要翻整本书
    → 只翻到相关章节 → 快速精准
"""


# =============================================================================
# 示例 1: RecursiveCharacterTextSplitter 递归分块
# =============================================================================

def chunk_document():
    """
    使用 LangChain 的 RecursiveCharacterTextSplitter 做智能分块。

    它按优先级尝试切分: 段落(\n\n) → 行(\n) → 句子(。)→ 字级别
    这样能尽量保持语义完整性，不会在句子中间断开。
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    print(f"\n-- 示例 1: 递归文本分块")

    text = """人工智能是计算机科学的一个分支，致力于让计算机具有人类智能。
机器学习是人工智能的核心技术之一，让计算机通过数据学习规律。
深度学习是机器学习的一个子领域，使用神经网络模拟人脑的工作方式。
AI的应用非常广泛，包括图像识别、自然语言处理、推荐系统、自动驾驶等。"""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=60,        # 每块最多 60 字符
        chunk_overlap=10,     # 块之间重叠 10 字符（保持上下文连贯）
        separators=["\n", "。", "，", " "],
    )

    chunks = text_splitter.split_text(text)

    print(f"原文 {len(text)} 字符 → {len(chunks)} 个块:\n")
    for i, chunk in enumerate(chunks):
        print(f"  [块{i + 1}] ({len(chunk)}字) {chunk.strip()}")


# =============================================================================
# 示例 2: 从字符串创建 Document 对象
# =============================================================================

def create_documents_from_text():
    """
    直接从字符串创建 Document 对象——适合硬编码知识库。

    Document 有两个核心属性:
    - page_content: 文档正文（会被向量化和检索）
    - metadata:     附加信息（来源、作者等，可用于过滤）
    """
    from langchain_core.documents import Document

    print(f"\n-- 示例 2: 从字符串创建 Document 对象")

    documents = [
        Document(
            page_content="Python 是一种高级编程语言，由 Guido van Rossum 于 1989 年发明。",
            metadata={"source": "python_intro", "topic": "programming"},
        ),
        Document(
            page_content="机器学习是让计算机从数据中学习规律的技术。",
            metadata={"source": "ml_intro", "topic": "ai"},
        ),
        Document(
            page_content="深度学习使用神经网络模拟人脑，在图像识别等领域效果很好。",
            metadata={"source": "dl_intro", "topic": "ai"},
        ),
    ]

    for doc in documents:
        print(f"  来源: {doc.metadata['source']} | 主题: {doc.metadata['topic']}")
        print(f"  内容: {doc.page_content}\n")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n>>> 07_retrieval/document_loader — 文档加载与分块\n")

    chunk_document()
    create_documents_from_text()

    # 接下来学习: vector_store.py（向量存储）
