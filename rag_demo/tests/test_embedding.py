"""测试 embedding.py — 向量生成模块"""

import os
import pytest
from unittest.mock import patch, MagicMock

from rag_demo.util.embedding import (
    generate_embedding,
    embedding_client,
    DEFAULT_EMBEDDING_DIMENSION,
)


class TestEmbeddingModule:
    """测试 Embedding 模块基本属性"""

    def test_default_dimension_is_1024(self):
        """验证默认维度为 1024"""
        assert DEFAULT_EMBEDDING_DIMENSION == 1024

    def test_embedding_client_exists(self):
        """验证 embedding_client 已初始化"""
        assert embedding_client is not None

    def test_generate_embedding_returns_list(self):
        """使用 mock 测试 generate_embedding 返回列表类型"""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * DEFAULT_EMBEDDING_DIMENSION)
        ]

        with patch.object(embedding_client.embeddings, 'create', return_value=mock_response):
            result = generate_embedding("测试文本")
            assert isinstance(result, list)

    def test_generate_embedding_correct_dimension(self):
        """使用 mock 测试返回的向量维度正确"""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * DEFAULT_EMBEDDING_DIMENSION)
        ]

        with patch.object(embedding_client.embeddings, 'create', return_value=mock_response):
            result = generate_embedding("测试文本")
            assert len(result) == DEFAULT_EMBEDDING_DIMENSION

    def test_generate_embedding_calls_correct_model(self):
        """测试调用了正确的模型名"""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * DEFAULT_EMBEDDING_DIMENSION)
        ]

        with patch.object(embedding_client.embeddings, 'create', return_value=mock_response) as mock_create:
            generate_embedding("测试文本")
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["model"] == "text-embedding-v4"

    def test_generate_embedding_custom_dimension(self):
        """测试自定义维度参数"""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 512)
        ]

        with patch.object(embedding_client.embeddings, 'create', return_value=mock_response) as mock_create:
            result = generate_embedding("测试", dimensions=512)
            assert len(result) == 512
            assert mock_create.call_args.kwargs["dimensions"] == 512

    def test_generate_embedding_default_dimension_param(self):
        """测试默认维度参数传入 API"""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * DEFAULT_EMBEDDING_DIMENSION)
        ]

        with patch.object(embedding_client.embeddings, 'create', return_value=mock_response) as mock_create:
            generate_embedding("测试")
            assert mock_create.call_args.kwargs["dimensions"] == DEFAULT_EMBEDDING_DIMENSION

    def test_generate_embedding_encoding_format(self):
        """测试 encoding_format 参数为 float"""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * DEFAULT_EMBEDDING_DIMENSION)
        ]

        with patch.object(embedding_client.embeddings, 'create', return_value=mock_response) as mock_create:
            generate_embedding("测试")
            assert mock_create.call_args.kwargs["encoding_format"] == "float"


class TestEmbeddingIntegration:
    """集成测试：验证 Embedding 的语义特性（需要 API Key）"""

    @pytest.mark.skipif(
        not os.getenv("ALIYUN_API_KEY"),
        reason="未设置 ALIYUN_API_KEY，跳过需要 API 的测试"
    )
    def test_real_embedding_dimension(self):
        """真实 API 测试：验证返回 1024 维向量"""
        result = generate_embedding("人工智能")
        assert len(result) == DEFAULT_EMBEDDING_DIMENSION

    @pytest.mark.skipif(
        not os.getenv("ALIYUN_API_KEY"),
        reason="未设置 ALIYUN_API_KEY，跳过需要 API 的测试"
    )
    def test_real_embedding_all_floats(self):
        """真实 API 测试：验证返回的是浮点数列表"""
        result = generate_embedding("测试")
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.skipif(
        not os.getenv("ALIYUN_API_KEY"),
        reason="未设置 ALIYUN_API_KEY，跳过需要 API 的测试"
    )
    def test_real_embedding_semantic_similarity(self):
        """真实 API 测试：语义相似的文本向量应更接近"""
        import numpy as np

        vec1 = generate_embedding("人工智能是计算机科学的前沿领域")
        vec2 = generate_embedding("AI技术正在改变世界")
        vec3 = generate_embedding("今天天气很好适合出去玩")

        # 计算余弦相似度
        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        sim_ai = cosine_sim(vec1, vec2)
        sim_unrelated = cosine_sim(vec1, vec3)

        # AI 相关的两个文本应该比不相关的更接近
        assert sim_ai > sim_unrelated, (
            f"语义相似的文本应更接近！AI相似度={sim_ai:.4f}, 无关相似度={sim_unrelated:.4f}"
        )
