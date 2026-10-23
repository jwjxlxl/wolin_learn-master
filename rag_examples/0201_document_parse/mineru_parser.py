"""MinerU 解析 PDF 的核心优势（对比 pypdf 等纯文本提取工具）。

1. 版面分析：自动识别标题、正文、表格、图片、公式等区域，而非把整页文字线性倒出；
2. 阅读顺序还原：对双栏、混排等复杂版面，能恢复正确的阅读顺序；
3. OCR 识别：图片型（扫描件、无文本层）的 PDF 也能识别出文字；
4. 公式识别：数学公式可转换为 LaTeX；
5. 结构化输出：解析结果以 Markdown 形式返回，图片、表格单独外置保存。
"""

import os
from pathlib import Path, PurePosixPath

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "rag_examples" / "data"
OUTPUT_DIR = PROJECT_DIR / "rag_examples" / "data" / "output"


def get_vlm_config():
    """从 .env 读取 OpenAI 兼容的 Qwen 视觉模型配置。

    MinerU 的 VLM 阶段需要视觉模型做版面识别与 OCR，
    因此 model 必须是 qwen-vl 系列，不能用纯文本的 qwen-plus。
    """
    from mineru.config import VlmConfig

    load_dotenv()
    values = {
        "server_url": os.getenv(
            "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).strip(),
        "api_key": os.getenv("ALIYUN_API_KEY", "").strip(),
        "model": os.getenv("QWEN_MODEL", "qwen-vl-ocr").strip(),
    }
    if not all(values.values()):
        raise RuntimeError("请在 .env 中配置 ALIYUN_API_KEY（可选：QWEN_MODEL、QWEN_BASE_URL）")
    return VlmConfig(**values)


def parse_pdf(file_path, output_dir=OUTPUT_DIR):
    """使用 MinerU 返回复杂 PDF 文本，并保存解析出的图片。"""
    # 当前 Windows 环境使用 Torch，避免 ONNX 原生运行库兼容问题
    os.environ.setdefault("MINERU_MODEL_SMALL_BACKEND", "torch")
    try:
        from docvortex.export import materialize_middle
        from mineru.parser import MinerUParser, ParseResult
    except ImportError as error:
        raise RuntimeError("请先运行：uv sync --extra mineru") from error

    file_path = Path(file_path)
    parser = MinerUParser(tier="standard", vlm_config=get_vlm_config())
    try:
        result = parser.parse(file_path)
    except Exception as error:
        if "invalid_api_key" in str(error):
            raise RuntimeError("Qwen API Key 无效，请检查 .env 中的 ALIYUN_API_KEY") from error
        raise

    # 外置图片后返回 Markdown，不写出 Markdown 或 JSON 文件
    middle_json, images = materialize_middle(result.middle_json)
    document_dir = output_dir / file_path.stem
    for image_path, image_data in images.items():
        relative_path = PurePosixPath(image_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("MinerU 返回了不安全的图片路径：{}".format(image_path))
        target = document_dir.joinpath(*relative_path.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(image_data)

    # 图片链接用文档目录名做相对基路径，避免绝对路径/中文空格被 URL 编码成乱码
    return ParseResult(middle_json).markdown(asset_base_url=file_path.stem)


def main():
    # 批量测试 datas 目录下的复杂 PDF 文件
    for file_path in DATA_DIR.rglob("扫描件_检索增强生成综述.pdf"):
        text = parse_pdf(file_path)
        print("解析出的文本的内容:", text)
        print("MinerU 解析完成：{}，共 {} 个字符".format(file_path.name, len(text)))


if __name__ == "__main__":
    main()
