"""测试 text_splitter.py — 文本切片功能"""

import pytest
from rag_demo.util.text_splitter import TextChunker, split_text


class TestTextChunker:
    """测试 TextChunker 类"""

    def test_default_parameters(self):
        """测试默认参数初始化"""
        chunker = TextChunker()
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 50

    def test_custom_parameters(self):
        """测试自定义参数初始化"""
        chunker = TextChunker(chunk_size=200, chunk_overlap=30)
        assert chunker.chunk_size == 200
        assert chunker.chunk_overlap == 30

    def test_split_basic(self, sample_chinese_text):
        """测试基本切片功能"""
        chunker = TextChunker(chunk_size=80, chunk_overlap=10)
        chunks = chunker.split(sample_chinese_text)
        assert len(chunks) > 0
        assert all("content" in chunk for chunk in chunks)
        assert all("metadata" in chunk for chunk in chunks)

    def test_split_metadata_includes_chunk_index(self, sample_chinese_text):
        """测试切片元数据包含 chunk_index"""
        chunker = TextChunker(chunk_size=80, chunk_overlap=10)
        chunks = chunker.split(sample_chinese_text)
        for i, chunk in enumerate(chunks):
            assert chunk["metadata"]["chunk_index"] == i

    def test_split_metadata_includes_total_chunks(self, sample_chinese_text):
        """测试元数据包含 total_chunks"""
        chunker = TextChunker(chunk_size=80, chunk_overlap=10)
        chunks = chunker.split(sample_chinese_text)
        total = chunks[0]["metadata"]["total_chunks"]
        assert total == len(chunks)
        for chunk in chunks:
            assert chunk["metadata"]["total_chunks"] == total

    def test_split_with_custom_metadata(self, sample_chinese_text):
        """测试自定义元数据会合并到每个切片"""
        chunker = TextChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.split(
            sample_chinese_text,
            metadata={"file_name": "test.txt", "author": "tester"},
        )
        for chunk in chunks:
            assert chunk["metadata"]["file_name"] == "test.txt"
            assert chunk["metadata"]["author"] == "tester"

    def test_split_empty_text(self):
        """测试空文本"""
        chunker = TextChunker()
        chunks = chunker.split("")
        # 空文本应该返回空列表或单个空切片
        assert isinstance(chunks, list)

    def test_split_single_sentence(self):
        """测试单句文本"""
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.split("这是一句测试。")
        assert len(chunks) >= 1

    def test_split_content_not_empty(self, sample_chinese_text):
        """测试每个切片的内容不应为空"""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.split(sample_chinese_text)
        for chunk in chunks:
            assert len(chunk["content"]) > 0

    def test_overlap_produces_more_chunks(self, large_chinese_text):
        """测试更大的 overlap 会产生更多切片"""
        chunker_small = TextChunker(chunk_size=100, chunk_overlap=10)
        chunker_large = TextChunker(chunk_size=100, chunk_overlap=80)

        chunks_small = chunker_small.split(large_chinese_text)
        chunks_large = chunker_large.split(large_chinese_text)

        # overlap 更大 → 切片更多
        assert len(chunks_large) >= len(chunks_small)

    def test_chunks_reassemble_to_original(self, sample_chinese_text):
        """测试切片拼接后包含原文关键内容"""
        chunker = TextChunker(chunk_size=50, chunk_overlap=0)
        chunks = chunker.split(sample_chinese_text)
        combined = "".join(c["content"] for c in chunks)
        # 无 overlap 时，所有 chunk 拼接应包含原文（可能有少量丢失在分隔符）
        assert "人工智能" in combined
        assert len(combined) > 0


class TestSplitTextFunction:
    """测试 split_text 便捷函数"""

    def test_split_text_basic(self, sample_chinese_text):
        """测试便捷函数基本功能"""
        chunks = split_text(sample_chinese_text, chunk_size=80, chunk_overlap=10)
        assert len(chunks) > 0
        assert all("content" in chunk for chunk in chunks)

    def test_split_text_default_params(self, sample_chinese_text):
        """测试默认参数"""
        chunks = split_text(sample_chinese_text)
        assert len(chunks) > 0

    def test_split_text_with_metadata(self, sample_chinese_text):
        """测试带元数据的调用"""
        chunks = split_text(
            sample_chinese_text,
            metadata={"source": "unit_test"},
            chunk_size=100,
            chunk_overlap=20,
        )
        for chunk in chunks:
            assert chunk["metadata"]["source"] == "unit_test"


class TestEdgeCases:
    """边界条件测试"""

    def test_very_small_chunk_size(self, sample_chinese_text):
        """测试极小的 chunk_size"""
        chunker = TextChunker(chunk_size=5, chunk_overlap=1)
        chunks = chunker.split(sample_chinese_text)
        # 应该能正常处理
        assert len(chunks) > 0

    def test_overlap_larger_than_chunk(self):
        """测试 overlap > chunk_size 会抛出 ValueError"""
        import pytest
        chunker = TextChunker(chunk_size=50, chunk_overlap=100)
        # LangChain 会拒绝不合理的参数：overlap 必须小于 chunk_size
        with pytest.raises(ValueError, match="larger chunk overlap"):
            chunker.split("这是一个测试文本，用于验证边界条件。")

    def test_very_long_single_line(self):
        """测试超长无分隔符文本"""
        long_text = "A" * 5000
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.split(long_text)
        assert len(chunks) > 1  # 应该被切分成多个 chunk

    def test_english_text(self, sample_english_text):
        """测试英文文本切片"""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.split(sample_english_text)
        assert len(chunks) > 0
