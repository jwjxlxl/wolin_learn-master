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
    # print(parse_txt_file("C:\\沃林AI课程\\Dify\\教育服务\\相关资料\\三国演义.txt"))
    print(parse_pdf_file(r"C:\沃林AI课程\Dify\教育服务\相关资料\测试简历\曹文栋-简历.pdf"))