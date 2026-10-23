from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "rag_examples" / "data"


def read_txt(file_path):
    """兼容常见的中文文本编码。"""
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeError("无法识别文本编码：{}".format(file_path))


def parse_txt(file_path):
    """解析 TXT 并直接返回文本。"""
    return read_txt(Path(file_path))


def main():
    # 批量测试 datas 目录下的 TXT 文件
    for file_path in DATA_DIR.rglob("ai_intro.txt"):
        text = parse_txt(file_path)
        print("TXT 解析完成：{}，共 {} 个字符".format(file_path.name, len(text)))


if __name__ == "__main__":
    main()
