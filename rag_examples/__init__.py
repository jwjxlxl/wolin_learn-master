# RAG Examples Package
# 导出 Milvus 配置供子模块使用

import sys
import os

# 将 milvus_config.py 添加到路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _current_dir)

# 导入并重新导出配置
from milvus_config import MILVUS_URI, DEFAULT_COLLECTION_NAME, DEFAULT_DIMENSION, DEFAULT_METRIC_TYPE

__all__ = ['MILVUS_URI', 'DEFAULT_COLLECTION_NAME', 'DEFAULT_DIMENSION', 'DEFAULT_METRIC_TYPE']
