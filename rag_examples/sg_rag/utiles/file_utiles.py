# =============================================================================
# 文件处理工具
# =============================================================================
#
# 用途：提供各类文件（txt / doc / docx / pdf / md 等）解析函数的统一入口
# =============================================================================

import os
from pathlib import Path


def parse_txt(file_path: str) -> str:
    """
    解析 TXT 文件，返回完整文本内容

    参数：
        file_path: TXT 文件的绝对或相对路径

    返回：
        str: 文件的完整文本内容

    异常：
        FileNotFoundError: 文件不存在时抛出
        ValueError: 文件格式不是 .txt 时抛出
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")

    if path.suffix.lower() != ".txt":
        raise ValueError(f"不支持的文件格式，期望 .txt，实际为 {path.suffix}")

    # 尝试常见编码，优先 UTF-8，回退 GBK
    for encoding in ("utf-8", "gbk", "gb2312", "utf-8-sig"):
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue

    raise ValueError(f"无法解析文件编码：{file_path}（尝试了 utf-8 / gbk / gb2312 / utf-8-sig）")


def parse_docx(file_path: str) -> str:
    """
    解析 DOCX 文件，按段落顺序提取文本内容

    参数：
        file_path: DOCX 文件的绝对或相对路径

    返回：
        str: 文件的完整文本内容（段落之间用换行符分隔）

    异常：
        FileNotFoundError: 文件不存在时抛出
        ValueError: 文件格式不是 .docx 时抛出
    """
    from docx import Document

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")

    if path.suffix.lower() != ".docx":
        raise ValueError(f"不支持的文件格式，期望 .docx，实际为 {path.suffix}")

    doc = Document(file_path)

    paragraphs = []
    for para in doc.paragraphs:
        if para.text.strip():
            paragraphs.append(para.text)

    return "\n".join(paragraphs)


def parse_doc(file_path: str) -> str:
    """
    解析 DOC 文件（旧版 Word 格式），通过 win32com 调用 Word 应用
    先转为临时 DOCX，再提取文本

    参数：
        file_path: DOC 文件的绝对或相对路径

    返回：
        str: 文件的完整文本内容

    异常：
        FileNotFoundError: 文件不存在时抛出
        ValueError: 文件格式不是 .doc 时抛出
        RuntimeError: 转换失败或系统未安装 Word 时抛出
    """
    from win32com import client as win32_client

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")

    if path.suffix.lower() != ".doc":
        raise ValueError(f"不支持的文件格式，期望 .doc，实际为 {path.suffix}")

    abs_path = str(path.resolve())
    temp_docx = abs_path + ".temp.docx"

    try:
        # 调用 Word 转换
        word = win32_client.Dispatch("Word.Application")
        word.Visible = False
        word.DisplayAlerts = False

        doc = word.Documents.Open(abs_path)
        doc.SaveAs2(temp_docx, FileFormat=16)  # 16 = wdFormatXMLDocument
        doc.Close()
        word.Quit()

        # 复用 parse_docx 提取文本
        text = parse_docx(temp_docx)
        return text

    except Exception as e:
        raise RuntimeError(f"DOC 文件转换失败：{e}")

    finally:
        # 清理临时文件
        if os.path.exists(temp_docx):
            try:
                os.remove(temp_docx)
            except Exception:
                pass


def parse_pdf(file_path: str) -> str:
    """
    解析 PDF 文件，提取文本内容

    参数：
        file_path: PDF 文件的绝对或相对路径

    返回：
        str: 文件的完整文本内容（页面之间用换行符分隔）

    异常：
        FileNotFoundError: 文件不存在时抛出
        ValueError: 文件格式不是 .pdf 时抛出
    """
    from PyPDF2 import PdfReader

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"不支持的文件格式，期望 .pdf，实际为 {path.suffix}")

    reader = PdfReader(file_path)

    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            pages_text.append(text)

    return "\n".join(pages_text)


def parse_markdown(file_path: str) -> str:
    """
    解析 Markdown 文件，返回原始文本内容（含 Markdown 标记）

    参数：
        file_path: MD 文件的绝对或相对路径

    返回：
        str: 文件的完整文本内容

    异常：
        FileNotFoundError: 文件不存在时抛出
        ValueError: 文件格式不是 .md / .markdown 时抛出
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")

    if path.suffix.lower() not in (".md", ".markdown"):
        raise ValueError(f"不支持的文件格式，期望 .md / .markdown，实际为 {path.suffix}")

    # 尝试常见编码
    for encoding in ("utf-8", "gbk", "gb2312", "utf-8-sig"):
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue

    raise ValueError(f"无法解析文件编码：{file_path}（尝试了 utf-8 / gbk / gb2312 / utf-8-sig）")


def parse_file(file_path: str) -> str:
    """
    根据文件后缀自动选择解析函数，统一入口

    参数：
        file_path: 文件的绝对或相对路径

    返回：
        str: 文件的完整文本内容

    异常：
        FileNotFoundError: 文件不存在时抛出
        ValueError: 文件格式不支持时抛出
    """
    suffix = Path(file_path).suffix.lower()

    dispatch = {
        ".txt": parse_txt,
        ".doc": parse_doc,
        ".docx": parse_docx,
        ".pdf": parse_pdf,
        ".md": parse_markdown,
        ".markdown": parse_markdown,
    }

    parser = dispatch.get(suffix)
    if parser is None:
        raise ValueError(f"不支持的文件格式：{suffix}（支持 .txt / .doc / .docx / .pdf / .md）")

    return parser(file_path)


if __name__ == "__main__":
    text = parse_file("../datas/《三国演义》.txt")
    print(f"[OK] 解析成功，共 {len(text)} 个字符")
    print(f"     前 100 字符预览：{text[:100]}...")
