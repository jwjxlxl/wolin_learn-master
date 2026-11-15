"""示例共用的环境检查与模型客户端。"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[1] / ".env")

MODEL_NAME = os.getenv("LANGFUSE_DEMO_MODEL", "qwen-plus")
QWEN_BASE_URL = os.getenv(
    "ALIYUN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)


def setup_console_encoding() -> None:
    """在 Windows 终端中以 UTF-8 输出中文。"""
    if hasattr(sys.stdout, "reconfigure") and (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def require_setting(name: str) -> str:
    """检查必需的环境变量，拒绝课程模板里的占位值。"""
    value = os.getenv(name, "").strip()
    if not value or value.startswith(("your_", "pk-lf-your", "sk-lf-your", "sk-your")):
        raise RuntimeError(f"请在项目根目录 .env 中配置有效的 {name}")
    return value


def get_langfuse_client():
    """连接 Langfuse；服务地址可由 LANGFUSE_BASE_URL 指定。"""
    require_setting("LANGFUSE_PUBLIC_KEY")
    require_setting("LANGFUSE_SECRET_KEY")
    from langfuse import get_client

    client = get_client()
    try:
        authenticated = client.auth_check()
    except Exception as exc:
        raise RuntimeError("无法连接 Langfuse，请检查网络和 LANGFUSE_BASE_URL") from exc
    if not authenticated:
        raise RuntimeError("Langfuse 鉴权失败，请检查密钥和 LANGFUSE_BASE_URL")
    return client


def create_qwen_client():
    """创建普通的通义千问 OpenAI 兼容客户端，供手动埋点示例使用。"""
    api_key = require_setting("ALIYUN_API_KEY")
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=QWEN_BASE_URL)
