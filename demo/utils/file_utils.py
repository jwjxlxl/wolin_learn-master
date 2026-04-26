"""文件处理工具函数模块"""

import os
import subprocess

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None


def parse_txt_file(file_path: str) -> str:
    """解析 .txt 文件并返回文本内容

    Args:
        file_path: .txt 文件的路径

    Returns:
        文件内容的字符串

    Raises:
        FileNotFoundError: 当文件不存在时
        UnicodeDecodeError: 当文件编码无法识别时
    """
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(
        f"无法解码文件 {file_path}，尝试的编码格式：{encodings}"
    )


def parse_pdf_file(file_path: str) -> str:
    """解析 .pdf 文件并返回文本内容

    Args:
        file_path: .pdf 文件的路径

    Returns:
        文件内容的字符串

    Raises:
        FileNotFoundError: 当文件不存在时
        ImportError: 当 pypdf2 未安装时
    """
    if PdfReader is None:
        raise ImportError("请先安装 pypdf2: pip install pypdf2")

    reader = PdfReader(file_path)
    text_parts = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_parts.append(text)

    return "\n\n".join(text_parts)


def parse_doc_file(file_path: str) -> str:
    """解析 .doc 文件 (Word 97-2003) 并返回文本内容

    Args:
        file_path: .doc 文件的路径

    Returns:
        文件内容的字符串

    Raises:
        FileNotFoundError: 当文件不存在时
        ImportError: 当 antiword 未安装时
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}")

    try:
        result = subprocess.run(
            ["antiword", file_path],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except FileNotFoundError:
        raise ImportError(
            "请先安装 antiword: "
            "Linux: sudo apt-get install antiword | "
            "macOS: brew install antiword | "
            "Windows: 下载二进制文件并添加到 PATH"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"解析 .doc 文件失败：{e.stderr}")


def parse_docx_file(file_path: str) -> str:
    """解析 .docx 文件 (Word 2007+) 并返回文本内容

    Args:
        file_path: .docx 文件的路径

    Returns:
        文件内容的字符串

    Raises:
        FileNotFoundError: 当文件不存在时
        ImportError: 当 python-docx 未安装时
    """
    if Document is None:
        raise ImportError("请先安装 python-docx: pip install python-docx")

    doc = Document(file_path)
    text_parts = []

    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            text_parts.append(paragraph.text)

    return "\n\n".join(text_parts)


if __name__ == "__main__":
    print(parse_pdf_file(r"C:\Ric\Resumes\赵雅坤简历V1.pdf"))