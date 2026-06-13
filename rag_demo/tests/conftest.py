"""rag_demo 测试配置和共享 fixtures"""

import os
import sys
import pytest
import tempfile
from pathlib import Path

# 确保 rag_demo 在 Python 路径中
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def sample_chinese_text():
    """提供一段标准中文测试文本"""
    return (
        "人工智能（AI）是模拟人类智能的计算机科学领域。"
        "机器学习通过训练数据让计算机自动学习规律。"
        "深度学习使用多层神经网络模拟人脑结构。"
        "自然语言处理让计算机理解和生成人类语言。"
        "RAG（检索增强生成）结合检索和生成技术，先检索相关知识再生成答案。"
    )


@pytest.fixture
def sample_english_text():
    """提供一段标准英文测试文本"""
    return (
        "Artificial intelligence is the simulation of human intelligence. "
        "Machine learning enables computers to learn from data. "
        "Deep learning uses neural networks with multiple layers. "
        "Natural language processing helps computers understand human language."
    )


@pytest.fixture
def temp_dir():
    """创建临时目录，测试结束后自动清理"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_txt_file(temp_dir, sample_chinese_text):
    """创建一个临时 TXT 文件用于测试"""
    file_path = temp_dir / "test_doc.txt"
    file_path.write_text(sample_chinese_text, encoding="utf-8")
    return str(file_path)


@pytest.fixture
def large_chinese_text():
    """提供一段较长的中文文本用于切片测试"""
    paragraphs = [
        "人工智能的发展历程可以追溯到20世纪50年代。早期的AI研究集中在逻辑推理和问题求解上。",
        "机器学习是人工智能的核心技术之一。它通过统计学方法让计算机从数据中学习模式和规律。",
        "深度学习是机器学习的一个重要分支。它使用多层人工神经网络来模拟人脑的学习过程。",
        "自然语言处理（NLP）是AI与语言学交叉的领域。它致力于让计算机理解、解释和生成人类语言。",
        "计算机视觉让机器能够'看懂'图像和视频。这一技术在自动驾驶、医疗影像分析等领域有广泛应用。",
        "推荐系统是AI在商业领域最成功的应用之一。从电商到短视频，推荐算法无处不在。",
        "大语言模型（LLM）是近年来AI领域最重要的突破。GPT、Claude、Qwen等模型展现了强大的语言能力。",
        "RAG（检索增强生成）解决了大模型的知识截止和幻觉问题。通过检索外部知识库，RAG可以给出更准确的答案。",
        "向量数据库是RAG系统的核心组件。Milvus是最流行的开源向量数据库之一，支持亿级向量检索。",
        "Embedding是将文本转换为向量的过程。好的Embedding模型应该让语义相似的文本在向量空间中距离相近。",
    ]
    return "\n\n".join(paragraphs)
