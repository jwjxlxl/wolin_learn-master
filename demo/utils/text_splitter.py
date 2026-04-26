"""文本切片工具类模块，使用 LangChain 进行文档切片"""

from typing import List, Union, Dict, Optional, Any

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
)


class TextSplitter:
    """文本切片工具类

    提供多种文本切片策略，基于 LangChain 实现。
    支持返回纯文本列表或带元数据的字典列表。
    """

    # 默认切片参数
    DEFAULT_CHUNK_SIZE = 500
    DEFAULT_CHUNK_OVERLAP = 50

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        separator: str = "\n\n",
        separators: Optional[List[str]] = None,
    ):
        """初始化文本切片器

        Args:
            chunk_size: 每个切片的最大字符数
            chunk_overlap: 切片之间的重叠字符数
            separator: 主要分隔符
            separators: 递归切片的分隔符列表 (按优先级排序)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.separators = separators or [
            "\n\n",      # 段落
            "\n",        # 行
            "。",        # 中文句号
            ".",         # 英文句号
            "！",        # 中文感叹号
            "!",         # 英文感叹号
            "？",        # 中文问号
            "?",         # 英文问号
            "；",        # 中文分号
            ";",         # 英文分号
            "，",        # 中文逗号
            ",",         # 英文逗号
            " ",         # 空格
            "",          # 字符级
        ]

    def split_recursive(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        return_with_metadata: bool = True,
    ) -> Union[List[Dict], List[str]]:
        """使用递归字符切片器进行文本切片

        这是推荐的切片方法，会尝试按段落、行、句子等层级进行切片，
        尽可能保持语义完整性。

        Args:
            text: 要切分的文本
            metadata: 可选的元数据，会添加到每个切片中
            return_with_metadata: 是否返回带元数据的字典列表

        Returns:
            如果 return_with_metadata 为 True，返回 List[Dict]，每个 dict 包含：
                - 'content': 切片文本
                - 'metadata': 元数据 (包含 chunk_index, total_chunks 等)
            否则返回 List[str]
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
        )

        if return_with_metadata:
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
        else:
            return splitter.split_text(text)

    def split_character(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        return_with_metadata: bool = True,
    ) -> Union[List[Dict], List[str]]:
        """使用简单字符切片器进行文本切片

        按指定分隔符进行简单切分，适合结构规整的文本。

        Args:
            text: 要切分的文本
            metadata: 可选的元数据
            return_with_metadata: 是否返回带元数据的字典列表

        Returns:
            如果 return_with_metadata 为 True，返回 List[Dict]，否则返回 List[str]
        """
        splitter = CharacterTextSplitter(
            separator=self.separator,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )

        if return_with_metadata:
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
        else:
            return splitter.split_text(text)

    def split_by_token(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        return_with_metadata: bool = True,
        encoding_name: str = "cl100k_base",
    ) -> Union[List[Dict], List[str]]:
        """使用 Token 切片器进行文本切片

        按 token 数量进行切分，适合需要精确控制 token 数量的场景
        (如调用 LLM API)。

        Args:
            text: 要切分的文本
            metadata: 可选的元数据
            return_with_metadata: 是否返回带元数据的字典列表
            encoding_name: tiktoken 编码器名称，可选：
                - cl100k_base (GPT-4, GPT-3.5-Turbo)
                - p50k_base (Codex, GPT-3)
                - r50k_base (GPT-2, GPT-3)

        Returns:
            如果 return_with_metadata 为 True，返回 List[Dict]，否则返回 List[str]
        """
        splitter = TokenTextSplitter(
            encoding_name=encoding_name,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        if return_with_metadata:
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
                        "token_encoding": encoding_name,
                        **(doc.metadata or {}),
                    },
                }
                for i, doc in enumerate(docs)
            ]
        else:
            return splitter.split_text(text)

    def split(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        method: str = "recursive",
        return_with_metadata: bool = True,
    ) -> Union[List[Dict], List[str]]:
        """统一的文本切片接口

        Args:
            text: 要切分的文本
            metadata: 可选的元数据
            method: 切片方法，可选：
                - 'recursive': 递归字符切片 (推荐)
                - 'character': 简单字符切片
                - 'token': Token 切片
            return_with_metadata: 是否返回带元数据的字典列表

        Returns:
            如果 return_with_metadata 为 True，返回 List[Dict]，否则返回 List[str]

        Raises:
            ValueError: 当 method 参数无效时
        """
        if method == "recursive":
            return self.split_recursive(text, metadata, return_with_metadata)
        elif method == "character":
            return self.split_character(text, metadata, return_with_metadata)
        elif method == "token":
            return self.split_by_token(text, metadata, return_with_metadata)
        else:
            raise ValueError(
                f"无效的切片方法：{method}，可选值：'recursive', 'character', 'token'"
            )


def split_text(
    text: str,
    chunk_size: int = TextSplitter.DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = TextSplitter.DEFAULT_CHUNK_OVERLAP,
    metadata: Optional[Dict[str, Any]] = None,
    method: str = "recursive",
) -> List[Dict]:
    """便捷函数：使用默认参数进行文本切片

    Args:
        text: 要切分的文本
        chunk_size: 每个切片的最大字符数
        chunk_overlap: 切片之间的重叠字符数
        metadata: 可选的元数据
        method: 切片方法 ('recursive', 'character', 'token')

    Returns:
        List[Dict]，每个 dict 包含 'content' 和 'metadata' 字段
    """
    splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split(
        text=text,
        metadata=metadata,
        method=method,
        return_with_metadata=True,
    )


if __name__ == "__main__":
    # 使用示例
    sample_text = """
    这是第一段。这是一个完整的段落，包含多个句子。
    这是第二段的开始。第二段也有多个句子。

    这是第三段，空行分隔的段落。

    这是第四段。
    """

    # 使用类的方式
    splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
    result = splitter.split(sample_text, method="recursive")
    print("=== 递归切片结果 ===")
    for chunk in result:
        print(f"切片 {chunk['metadata']['chunk_index']}: {chunk['content'][:30]}...")

    # 使用便捷函数
    result_with_meta = split_text(
        sample_text,
        chunk_size=50,
        chunk_overlap=10,
        metadata={"source": "example.txt", "author": "test"},
    )
    print("\n=== 带元数据的切片结果 ===")
    for chunk in result_with_meta:
        print(f"切片 {chunk['metadata']['chunk_index']}:")
        print(f"  内容：{chunk['content'][:30]}...")
        print(f"  元数据：{chunk['metadata']}")

    # 仅返回文本列表
    result_text_only = splitter.split(
        sample_text,
        method="recursive",
        return_with_metadata=False,
    )
    print(f"\n=== 纯文本列表 (共 {len(result_text_only)} 个切片) ===")
