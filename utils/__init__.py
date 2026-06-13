"""项目共享工具模块

提供跨子项目的通用工具函数，包括模型客户端、Embedding 等。
"""

from utils.model_utils import get_qwen_client, get_model

__all__ = ["get_qwen_client", "get_model"]
