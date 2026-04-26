# Embedding 示例 - 文本向量化

本模块演示如何使用 Embedding API 将文本转换为向量。

## 课程目录

| 文件 | 内容 | 难度 |
|------|------|------|
| [01_embedding_basics.py](01_embedding_basics.py) | Embedding 基础概念 | ⭐ |
| [02_aliyun_embedding.py](02_aliyun_embedding.py) | 阿里云百炼 Embedding API | ⭐⭐ |
| [03_local_embedding.py](03_local_embedding.py) | 本地 Embedding 模型 | ⭐⭐ |
| [04_embedding_comparison.py](04_embedding_comparison.py) | Embedding 模型对比 | ⭐⭐ |

## 什么是 Embedding？

Embedding 是将文本、图像等数据转换为数字向量的技术。

```
文本："人工智能是计算机科学的一个分支"
          ↓ Embedding 模型
向量：[0.12, -0.45, 0.78, 0.23, -0.56, ...] (768 维)
```

## 常见 Embedding 模型

| 模型 | 维度 | 语言 | 提供商 |
|------|------|------|--------|
| text-embedding-v3 | 1024/1536 | 中英文 | 阿里云 |
| bge-large-zh-v1.5 | 1024 | 中文 | 北京智源 |
| m3e-base | 768 | 中文 | MokaAI |
| text-embedding-3-small | 1536 | 多语言 | OpenAI |
| all-MiniLM-L6-v2 | 384 | 英文 | SentenceTransformers |

## 运行前准备

```bash
pip install -r ../requirements.txt
```

## 阿里云百炼 API 配置

1. 访问 https://bailian.console.aliyun.com/
2. 开通模型服务（text-embedding-v3）
3. 获取 API Key
4. 设置环境变量：`DASHSCOPE_API_KEY=your_key`
