"""通用辅助函数

提供跨模块共享的工具函数，减少重复代码。
"""

import os
from dotenv import load_dotenv

# 跟踪是否已加载 .env
_env_loaded = False


def ensure_env_loaded():
    """确保环境变量已加载（幂等操作，只加载一次）"""
    global _env_loaded
    if not _env_loaded:
        load_dotenv()
        _env_loaded = True


def get_api_key(key_names: list = None) -> str:
    """安全获取 API Key

    按优先级尝试多个环境变量名，返回第一个存在的。

    Args:
        key_names: 环境变量名列表，默认 ["DASHSCOPE_API_KEY", "ALIYUN_API_KEY"]

    Returns:
        API Key 字符串，如果未找到返回空字符串
    """
    if key_names is None:
        key_names = ["DASHSCOPE_API_KEY", "ALIYUN_API_KEY"]

    ensure_env_loaded()

    for name in key_names:
        key = os.getenv(name)
        if key:
            return key

    return ""


def safe_milvus_operation(operation_fn, error_msg: str = "Milvus 操作失败"):
    """安全执行 Milvus 操作，带异常处理

    Args:
        operation_fn: 无参数的 lambda 或 callable
        error_msg: 失败时的提示信息

    Returns:
        operation_fn 的返回值，失败时返回 None
    """
    try:
        return operation_fn()
    except Exception as e:
        print(f"[WARN] {error_msg}: {e}")
        return None


def format_score(score: float, precision: int = 4) -> str:
    """格式化分数为可读字符串"""
    return f"{score:.{precision}f}"


def truncate_text(text: str, max_len: int = 80, suffix: str = "...") -> str:
    """截断文本到指定长度

    Args:
        text: 原始文本
        max_len: 最大长度
        suffix: 超出时追加的后缀

    Returns:
        截断后的文本
    """
    if len(text) <= max_len:
        return text
    return text[:max_len - len(suffix)] + suffix
