# =============================================================================
# 02_sliding_window — 滑动窗口切片
# =============================================================================
# 用途：学习保持上下文连续性的切片方法 — 固定大小的窗口在文本上滑动
# 难度：⭐⭐（2 星）
# =============================================================================
# 核心概念：
#   - 固定大小的窗口滑动，每次移动一个步长
#   - 相邻切片有重叠，保持语义连续性
# 核心参数：
#   - window_size（窗口大小）：每个切片包含多少字符
#   - step_size（步长）：每次滑动多少字符
#   - overlap（重叠）：window_size - step_size
# =============================================================================

import re


# =============================================================================
# 核心切片函数
# =============================================================================

def sliding_window_chunking(text, window_size=100, step_size=50):
    """
    基础滑动窗口切片

    参数:
        text: 输入文本
        window_size: 窗口大小（每个切片的字符数）
        step_size: 步长（每次滑动的字符数）
    返回:
        切片列表
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + window_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += step_size
        if end >= len(text):
            break
    return chunks


def sliding_window_by_words(text, window_words=15, step_words=8):
    """
    按单词的滑动窗口切片（适合英文）

    参数:
        text: 输入文本
        window_words: 窗口包含的单词数
        step_words: 滑动步长的单词数
    返回:
        切片列表
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + window_words, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += step_words
        if end >= len(words):
            break
    return chunks


def sliding_window_with_sentence_boundary(text, window_sentences=3, step_sentences=2):
    """
    滑动窗口 + 句子边界

    窗口按句子数量滑动，确保句子完整

    参数:
        text: 输入文本
        window_sentences: 窗口包含的句子数
        step_sentences: 滑动步长的句子数
    返回:
        切片列表
    """
    sentences = re.split(r'[.!?。！？]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks = []
    start = 0
    while start < len(sentences):
        end = min(start + window_sentences, len(sentences))
        chunk = '。'.join(sentences[start:end]) + '。'
        chunks.append(chunk)
        start += step_sentences
        if end >= len(sentences):
            break
    return chunks


def visualize_sliding_window(text, window_size=20, step_size=10, max_windows=5):
    """
    可视化滑动窗口过程

    参数:
        text: 输入文本
        window_size: 窗口大小
        step_size: 滑动步长
        max_windows: 最多显示的窗口数
    """
    print(f"文本长度：{len(text)}, 窗口：{window_size}, 步长：{step_size}")
    start = 0
    i = 1
    while start < len(text) and i <= max_windows:
        end = min(start + window_size, len(text))
        print(f"窗口{i}: {text[start:end]}")
        start += step_size
        i += 1


def compare_sliding_window_params(text):
    """
    对比不同滑动窗口参数的效果

    参数:
        text: 输入文本
    """
    print(f"\n-- 示例 5: 参数对比实验")

    params_list = [
        (100, 80, "小重叠 (20%)"),
        (100, 50, "中重叠 (50%)"),
        (100, 20, "大重叠 (80%)"),
    ]

    print(f"文本长度：{len(text)} 字符")
    print(f"窗口大小：固定 100 字符")
    print(f"对比不同步长的效果\n")

    results = []

    for window_size, step_size, label in params_list:
        chunks = sliding_window_chunking(text, window_size, step_size)
        overlap = window_size - step_size
        overlap_ratio = overlap / window_size * 100

        results.append({
            "label": label,
            "window": window_size,
            "step": step_size,
            "overlap": overlap,
            "chunks": len(chunks)
        })

        print(f"{label}:")
        print(f"  步长={step_size}, 重叠={overlap} ({overlap_ratio:.0f}%)")
        print(f"  切片数量：{len(chunks)}")
        print(f"  总覆盖字符：{len(chunks) * window_size} (有重复)")
        print()

    print("选择建议：")
    print("  - 小重叠 (20%)：节省空间，适合粗略检索")
    print("  - 中重叠 (50%)：平衡性能和精度（推荐）")
    print("  - 大重叠 (80%)：保持上下文，适合精细检索")


def sliding_window_summary():
    """滑动窗口切片的优缺点总结"""
    print(f"\n-- 示例 6: 滑动窗口切片总结")

    print("""
滑动窗口切片 vs 固定切片

滑动窗口优势：
  - 保持上下文连续性
  - 减少语义切断的影响
  - 检索时召回更完整的信息

滑动窗口劣势：
  - 数据冗余（重叠部分重复存储）
  - 切片数量更多，存储成本增加
  - 检索时可能有重复结果

使用建议:

1. 文档较短（<1000 字符）
   -> 用小步长（step=window/4），保持完整上下文

2. 文档中等（1000-5000 字符）
   -> 用中等步长（step=window/2），平衡性能和精度

3. 文档很长（>5000 字符）
   -> 用大步长（step=window*3/4），减少冗余

参数建议:

| 场景    | 窗口大小 | 步长  | 重叠率 | 适用        |
|---------|----------|-------|--------|-------------|
| 短文档  | 200      | 100   | 50%    | 通用        |
| 中档文档 | 500      | 250   | 50%    | 通用        |
| 长文档  | 1000     | 500   | 50%    | 通用        |
| 高精度  | 300      | 75    | 75%    | 法律/医疗   |
| 省空间  | 500      | 400   | 20%    | 快速原型    |

注意事项:

1. 重叠率 = (window_size - step_size) / window_size
2. 重叠率越高，存储成本越大，但检索效果通常越好
3. 建议从 50% 重叠率开始，根据效果调整
""")


# =============================================================================
# 演示函数
# =============================================================================

def demo_basic_sliding_window():
    """演示基础滑动窗口"""
    print(f"\n-- 示例 1: 基础滑动窗口")

    text = '这是一段用于演示滑动窗口切片的测试文本。' * 5
    chunks = sliding_window_chunking(text, window_size=30, step_size=10)
    print(f"切片数量：{len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"切片 {i+1}: {chunk}")


def demo_visualize_window():
    """演示可视化滑动过程"""
    print(f"\n-- 示例 2: 可视化滑动过程")

    text = 'ABCDEFGHIJ' * 6
    visualize_sliding_window(text, window_size=20, step_size=10)


def demo_word_sliding_window():
    """演示按单词滑动窗口"""
    print(f"\n-- 示例 3: 按单词滑动窗口")

    text = """Machine learning is a subset of artificial intelligence.
It enables computers to learn from data without being explicitly programmed.
Deep learning uses neural networks with multiple layers.
Natural language processing helps computers understand human language.
Computer vision allows machines to interpret and analyze images."""

    print(f"原始文本单词数：{len(text.split())}")
    print(f"窗口大小：15 单词")
    print(f"步长：8 单词\n")

    chunks = sliding_window_by_words(text)
    print(f"切片数量：{len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"切片{i+1}: {chunk}")


def demo_sentence_boundary_window():
    """演示句子边界滑动窗口"""
    print(f"\n-- 示例 4: 滑动窗口 + 句子边界")

    text = """人工智能是计算机科学的一个分支。它试图理解智能的本质。
机器学习是 AI 的核心技术。深度学习是机器学习的重要分支。
自然语言处理让计算机理解人类语言。计算机视觉让机器看懂图像。
推荐系统根据用户偏好推荐内容。知识图谱用图结构表示知识。"""

    print(f"窗口大小：3 句子")
    print(f"步长：2 句子\n")

    chunks = sliding_window_with_sentence_boundary(text)

    print(f"切片数量：{len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"切片{i+1}: {chunk}")


def demo_parameter_comparison():
    """运行参数对比实验"""
    text = """
    人工智能（AI）是模拟人类智能的计算机科学领域。
    机器学习通过训练数据让计算机自动学习规律。
    深度学习使用多层神经网络模拟人脑。
    自然语言处理让计算机理解和生成人类语言。
    计算机视觉让计算机能够"看懂"图像和视频。
    推荐系统根据用户偏好推荐相关内容。
    知识图谱用图结构存储和表示知识。
    大语言模型是基于海量文本训练的深度学习模型。
    """.strip() * 3

    compare_sliding_window_params(text)


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  滑动窗口切片（Sliding Window Chunking）")
    print("=" * 60)

    demo_basic_sliding_window()
    demo_visualize_window()
    demo_word_sliding_window()
    demo_sentence_boundary_window()
    demo_parameter_comparison()
    sliding_window_summary()
