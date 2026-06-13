def parse_txt_file(file_path: str, encoding: str = "utf-8") -> str:
    """
    从本地路径解析 TXT 文件内容

    Args:
        file_path: 本地 TXT 文件的路径
        encoding: 文件编码，默认 utf-8，可尝试 gbk 等

    Returns:
        解析后的文本内容（字符串）
    """
    try:
        with open(file_path, "r", encoding=encoding) as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="gbk", errors="ignore") as f:
            text = f.read()

    return text.strip()


def parse_pdf_file(file_path: str) -> str:
    """
    从本地路径解析 PDF 文件内容

    Args:
        file_path: 本地 PDF 文件的路径

    Returns:
        解析后的文本内容（字符串）
    """
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    return "\n".join(pages).strip()


def parse_docx_file(file_path: str) -> str:
    """
    从本地路径解析 Word (.docx) 文件内容

    Args:
        file_path: 本地 .docx 文件的路径

    Returns:
        解析后的文本内容（字符串）
    """
    from docx import Document

    doc = Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

    return "\n".join(paragraphs).strip()


if __name__ == '__main__':
    # 使用说明：
    #   1. 取消注释并替换为你的文件路径
    #   2. 支持 TXT、PDF、DOCX 三种格式

    # 示例：解析 TXT 文件
    # print(parse_txt_file("path/to/your/file.txt"))

    # 示例：解析 PDF 文件
    # print(parse_pdf_file("path/to/your/file.pdf"))

    # 示例：解析 DOCX 文件
    # print(parse_docx_file("path/to/your/file.docx"))

    print("text_parser 模块加载成功。请取消注释上面的示例代码并替换为你的文件路径。")