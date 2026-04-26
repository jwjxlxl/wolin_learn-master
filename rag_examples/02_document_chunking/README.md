# 02_document_chunking - 文档切片

本模块演示如何对文档进行切片（Chunking），这是 RAG 系统中的关键步骤。

## 文档切片的重要性

文档切片的质量直接影响 RAG 系统的检索效果：
- **切太大**：包含过多无关信息，影响检索精度
- **切太小**：语义不完整，检索不到相关内容

## 课程目录

| 文件 | 内容 | 难度 |
|------|------|------|
| [01_fixed_chunking.py](01_fixed_chunking.py) | 固定规则切片 | ⭐ |
| [02_sliding_window.py](02_sliding_window.py) | 滑动窗口切片 | ⭐⭐ |
| [03_ai_chunking.py](03_ai_chunking.py) | AI 辅助切片 | ⭐⭐⭐ |
| [04_summary_chunking.py](04_summary_chunking.py) | 概要生成切片 | ⭐⭐⭐ |
| [05_chunking_comparison.py](05_chunking_comparison.py) | 切片方法对比 | ⭐⭐ |

## 切片方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| 固定规则 | 简单快速 | 可能切断语义 | 快速原型、对精度要求不高 |
| 滑动窗口 | 保持上下文 | 有数据冗余 | 通用场景 |
| AI 辅助 | 语义完整 | 需要额外 API | 高质量 RAG 系统 |
| 概要生成 | 检索效率高 | 可能丢失细节 | 长文档检索 |

## 运行前准备

```bash
pip install -r ../requirements.txt
```

## 学习建议

1. 先学习 `01_fixed_chunking.py` 理解切片的基本概念
2. 依次学习更复杂的方法
3. 最后运行 `05_chunking_comparison.py` 对比效果
