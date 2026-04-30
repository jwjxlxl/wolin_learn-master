"""文本切片工具类，使用 LangChain 的 RecursiveCharacterTextSplitter 实现"""

from typing import List, Dict, Optional, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:
    """基于 LangChain 的文本切片工具类

    使用 RecursiveCharacterTextSplitter 进行递归切片，
    优先按段落、行、句子层级切分，保持语义完整性。
    """

    DEFAULT_CHUNK_SIZE = 500
    DEFAULT_CHUNK_OVERLAP = 50

    SEPARATORS = [
        "\n\n",   # 段落
        "\n",     # 行
        "。",     # 中文句号
        ".",      # 英文句号
        "！",     # 中文感叹号
        "!",      # 英文感叹号
        "？",     # 中文问号
        "?",      # 英文问号
        "；",     # 中文分号
        ";",      # 英文分号
        "，",     # 中文逗号
        ",",      # 英文逗号
        " ",      # 空格
        "",       # 字符级
    ]

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        """初始化文本切片器

        Args:
            chunk_size: 每个切片的最大字符数
            chunk_overlap: 切片之间的重叠字符数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        """对输入文本进行切片

        Args:
            text: 要切分的文本内容（字符串）
            metadata: 可选的元数据字典，会合并到每个切片中

        Returns:
            list[dict]，每个元素包含：
                - 'content': 切片文本
                - 'metadata': 元数据（包含 chunk_index, total_chunks 及传入的 metadata）
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.SEPARATORS,
            length_function=len,
        )

        docs = splitter.create_documents(
            texts=[text],
            metadatas=[metadata or {}],
        )

        return [
            {
                "content": doc.page_content,
                "metadata": {
                    "chunk_index": i,
                    "total_chunks": len(docs),
                    **(doc.metadata or {}),
                },
            }
            for i, doc in enumerate(docs)
        ]


def split_text(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    chunk_size: int = TextChunker.DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = TextChunker.DEFAULT_CHUNK_OVERLAP,
) -> List[Dict]:
    """便捷函数：使用默认参数对文本进行切片

    Args:
        text: 要切分的文本内容（字符串）
        metadata: 可选的元数据字典
        chunk_size: 每个切片的最大字符数
        chunk_overlap: 切片之间的重叠字符数

    Returns:
        list[dict]，每个元素包含 'content' 和 'metadata'
    """
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.split(text, metadata)

if __name__ == "__main__":
    text = """
    # 测试文本
    这是一个测试文本，用于测试文本切片功能。
    """
    chunks = split_text(text)
    print(chunks)
