# =============================================================================
# 文本切片工具类
# =============================================================================
#
# 用途：对文件内容进行递归切片，返回带元数据的切片列表
# =============================================================================

from typing import List, Dict, Optional, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:
    """基于 LangChain 的文本切片工具类

    使用 RecursiveCharacterTextSplitter 进行递归切片，
    优先按段落、行、句子层级切分，保持语义完整性。
    """

    DEFAULT_CHUNK_SIZE = 500
    DEFAULT_CHUNK_OVERLAP = 50

    # 中文优先的分隔符优先级
    SEPARATORS = [
        "\n\n",       # 段落
        "\n",         # 行
        "。",         # 中文句号
        ".",          # 英文句号
        "！",         # 中文感叹号
        "!",          # 英文感叹号
        "？",         # 中文问号
        "?",          # 英文问号
        "；",         # 中文分号
        ";",          # 英文分号
        "，",         # 中文逗号
        ",",          # 英文逗号
        " ",          # 空格
        "",           # 字符级别兜底
    ]

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        """
        初始化切片器

        参数：
            chunk_size: 每块最大字符数
            chunk_overlap: 相邻块之间的重叠字符数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        """
        对文本进行递归切片

        参数：
            text: 待切分的完整文本内容（str）
            metadata: 附加元数据（如 file_name、source 等），会合并到每个切片中

        返回：
            list[dict]: 每个切片为一个字典，包含：
                - content: 切片文本（str）
                - chunk_index: 切片序号，从 0 开始
                - total_chunks: 切片总数
                - char_count: 字符数
                - 以及用户传入的 metadata 字段
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.SEPARATORS,
            length_function=len,
        )

        docs = splitter.create_documents(texts=[text], metadatas=[metadata or {}])

        total = len(docs)

        return [
            {
                "content": doc.page_content,
                "chunk_index": i,
                "total_chunks": total,
                "char_count": len(doc.page_content),
                **(doc.metadata or {}),
            }
            for i, doc in enumerate(docs)
        ]

    def split_by_paragraph(self, text: str) -> List[str]:
        """
        按段落简单切分（不重叠，不分块）

        参数：
            text: 待切分的完整文本

        返回：
            list[str]: 段落文本列表
        """
        return [p.strip() for p in text.split("\n\n") if p.strip()]


if __name__ == "__main__":
    # 简单自测
    sample = (
        "第一段内容。这是第一段话，用来测试文本切片功能。"
        "它包含了一些简单的中文句子。\n\n"
        "第二段内容。这是第二段话，用来展示重叠切片的效果。"
        "当 overlap 设置不为零时，相邻块会有部分文字重叠。\n\n"
        "第三段内容。第三段也很短，只是为了凑够测试长度。"
    )

    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.split(sample, metadata={"file_name": "test.txt"})

    print(f"[OK] 切片完成，共 {len(chunks)} 块\n")
    for chunk in chunks:
        print(f"  第 {chunk['chunk_index']}/{chunk['total_chunks']} 块"
              f"（{chunk['char_count']} 字符）：{chunk['content'][:30]}...")
