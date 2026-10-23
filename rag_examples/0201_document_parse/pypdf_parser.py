from pathlib import Path

from pypdf import PdfReader


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "rag_examples" / "data"
OUTPUT_DIR = PROJECT_DIR / "rag_examples" / "data" / "output"


def save_images(page, page_number, image_dir):
    """保存当前 PDF 页中的图片并返回数量。"""
    count = 0
    for index, image in enumerate(page.images, start=1):
        image_dir.mkdir(parents=True, exist_ok=True)
        name = "page_{:03d}_{:02d}_{}".format(page_number, index, Path(image.name).name)
        (image_dir / name).write_bytes(image.data)
        count += 1
    return count


def parse_pdf(file_path, output_dir=OUTPUT_DIR):
    """返回普通 PDF 的文本，并将图片保存到 output。"""
    file_path = Path(file_path)
    reader = PdfReader(str(file_path))
    pages = []
    image_dir = output_dir / file_path.stem / "images"

    # 保留页码，方便后续切分和检索
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        pages.append("--- 第 {} 页 ---\n{}".format(number, text))
        save_images(page, number, image_dir)

    return "\n\n".join(pages)


def main():
    # 批量测试 datas 目录下的 PDF 文件
    for file_path in DATA_DIR.rglob("扫描件_检索增强生成综述.pdf"):
        text = parse_pdf(file_path)
        print("解析出的文本的内容:", text)
        print("PyPDF 解析完成：{}，共 {} 个字符".format(file_path.name, len(text)))


if __name__ == "__main__":
    main()
