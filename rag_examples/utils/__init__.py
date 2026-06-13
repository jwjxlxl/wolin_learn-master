"""rag_examples 共享工具包

提供跨模块的通用功能：
- 环境配置加载
- 通用 Embedding 封装（mock + real 双模式）
- Milvus 客户端辅助函数
"""

from rag_examples.utils.helpers import (
    ensure_env_loaded,
    get_api_key,
    safe_milvus_operation,
)


__all__ = [
    "ensure_env_loaded",
    "get_api_key",
    "safe_milvus_operation",
]
