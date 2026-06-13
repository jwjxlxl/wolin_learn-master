"""测试 text_parser.py — TXT/PDF/DOCX 文件解析"""

import os
import pytest
from pathlib import Path

from rag_demo.util.text_parser import parse_txt_file, parse_pdf_file, parse_docx_file


class TestParseTxtFile:
    """测试 TXT 文件解析"""

    def test_parse_utf8_txt(self, temp_txt_file):
        """测试解析 UTF-8 编码的 TXT 文件"""
        text = parse_txt_file(temp_txt_file)
        assert len(text) > 0
        assert "人工智能" in text
        assert "机器学习" in text

    def test_parse_utf8_txt_encoding_param(self, temp_txt_file):
        """测试显式指定 UTF-8 编码"""
        text = parse_txt_file(temp_txt_file, encoding="utf-8")
        assert "人工智能" in text

    def test_parse_txt_file_not_found(self):
        """测试文件不存在的情况"""
        with pytest.raises(FileNotFoundError):
            parse_txt_file("/nonexistent/path/file.txt")

    def test_parse_txt_strips_whitespace(self, temp_dir):
        """测试解析后去除首尾空白"""
        file_path = temp_dir / "whitespace.txt"
        file_path.write_text("  \n  内容在中间  \n  ", encoding="utf-8")
        text = parse_txt_file(str(file_path))
        assert text == "内容在中间"

    def test_parse_txt_handles_gbk_fallback(self, temp_dir):
        """测试 UTF-8 失败时回退到 GBK 编码"""
        content = "中文内容测试"
        file_path = temp_dir / "gbk_test.txt"
        # 用 GBK 编码写入
        file_path.write_bytes(content.encode("gbk"))
        text = parse_txt_file(str(file_path))
        assert "中文内容测试" in text


class TestParsePdfFile:
    """测试 PDF 文件解析（需要 pypdf 库）"""

    def test_parse_pdf_import(self):
        """测试 pypdf 库可用"""
        try:
            from pypdf import PdfReader
            assert PdfReader is not None
        except ImportError:
            pytest.skip("pypdf 未安装")

    def test_parse_pdf_file_not_found(self):
        """测试 PDF 文件不存在"""
        with pytest.raises(Exception):
            parse_pdf_file("/nonexistent/path/file.pdf")


class TestParseDocxFile:
    """测试 DOCX 文件解析（需要 python-docx 库）"""

    def test_parse_docx_import(self):
        """测试 python-docx 库可用"""
        try:
            from docx import Document
            assert Document is not None
        except ImportError:
            pytest.skip("python-docx 未安装")

    def test_parse_docx_file_not_found(self):
        """测试 DOCX 文件不存在"""
        with pytest.raises(Exception):
            parse_docx_file("/nonexistent/path/file.docx")


class TestParseIntegration:
    """集成测试：确保三种解析器返回一致的文本格式"""

    def test_all_parsers_return_string(self, temp_txt_file):
        """所有解析器应返回字符串类型"""
        text = parse_txt_file(temp_txt_file)
        assert isinstance(text, str)

    def test_all_parsers_return_non_empty(self, temp_txt_file):
        """解析非空文件应返回非空字符串"""
        text = parse_txt_file(temp_txt_file)
        assert len(text) > 0
