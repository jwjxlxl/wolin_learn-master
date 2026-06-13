# =============================================================================
# 05_chunking_comparison — 切片方法对比
# =============================================================================
# 用途：对比四种文档切片方法的效果，帮助学生选择合适的方法
# 难度：⭐⭐（2 星）
# =============================================================================
# 核心概念：
#   - 固定规则切片：简单快速，可能切断语义
#   - 滑动窗口切片：保持上下文连续性，有数据冗余
#   - AI 辅助切片：语义完整，需要 API 调用
#   - 概要生成切片：检索精准，额外生成成本
#
# 对比维度：切片质量、处理速度、存储成本、检索效果、实现复杂度
# =============================================================================

import re
import statistics
from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# 四种切片方法
# =============================================================================

def fixed_chunking(text, chunk_size=100):
    """固定字符切片"""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks


def sliding_window_chunking(text, window_size=100, step_size=50):
    """滑动窗口切片"""
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + window_size]
        chunks.append(chunk)
        start += step_size
        if start >= len(text):
            break
    return chunks


def ai_chunking(text, max_chunk_size=150):
    """模拟 AI 切片（按段落）"""
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = [p.strip() for p in paragraphs if p.strip()]
    return chunks


def summary_chunking(text):
    """模拟概要切片（返回原文 + 摘要）"""
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    summaries = []
    for p in paragraphs:
        p = p.strip()
        if p:
            chunks.append(p)
            first_sentence = re.split(r'[.!?!.!?。！？]+', p)[0]
            summaries.append(
                first_sentence[:30] + "..."
                if len(first_sentence) > 30
                else first_sentence
            )
    return chunks, summaries


# =============================================================================
# 质量评估与检索模拟
# =============================================================================

def evaluate_chunk_quality(chunks):
    """评估切片质量的简单指标

    指标包括：
    - 平均长度
    - 长度标准差（越小说明越均匀）
    - 完整性评分（基于是否有完整句子）
    """
    if not chunks:
        return None

    lengths = [len(c) for c in chunks]
    avg_length = statistics.mean(lengths)
    std_length = statistics.stdev(lengths) if len(lengths) > 1 else 0
    complete_ratio = sum(1 for c in chunks if '。' in c or '.' in c) / len(chunks)

    return {
        "avg_length": avg_length,
        "std_length": std_length,
        "complete_ratio": complete_ratio,
        "total_chunks": len(chunks),
    }


def simulate_search(query, chunks):
    """模拟检索：基于关键词重叠度"""
    query_words = set(query.lower())
    scores = []
    for i, chunk in enumerate(chunks):
        chunk_words = set(chunk.lower())
        overlap = len(query_words & chunk_words)
        scores.append((i, overlap, chunk))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:3]


# =============================================================================
# 演示函数
# =============================================================================

def demo_all_methods():
    """演示四种切片方法"""
    print(f"\n-- 示例 1: 四种切片方法对比")

    text = """人工智能（AI）是模拟人类智能的计算机科学领域。它包括机器学习、深度学习、自然语言处理等多个分支。AI 从 1956 年诞生至今，已经经历了多次发展浪潮。

机器学习是 AI 的核心技术之一。它通过训练数据让计算机自动学习规律，无需显式编程。监督学习、无监督学习、强化学习是三种主要的学习范式。

深度学习是机器学习的子集，使用多层神经网络模拟人脑。它在图像识别、语音识别等领域取得了突破性进展。卷积神经网络（CNN）和循环神经网络（RNN）是两种经典架构。

自然语言处理（NLP）让计算机理解和生成人类语言。应用包括机器翻译、情感分析、智能客服等。近年来，大语言模型（LLM）在 NLP 领域引发了革命。

计算机视觉（CV）让计算机能够"看懂"图像和视频。应用包括人脸识别、物体检测、医学图像分析等。深度学习大幅提升了 CV 的性能。"""

    print(f"原始文本长度：{len(text)} 字符\n")
    print(f"原始文本:\n{text[:200]}...\n")

    fixed = fixed_chunking(text, chunk_size=100)
    print(f"\n1. 固定字符切片 (chunk_size=100)")
    print(f"   切片数量：{len(fixed)}")
    for i, chunk in enumerate(fixed[:4]):
        print(f"   [{i+1}] {len(chunk)}字：{chunk[:40]}...")

    sliding = sliding_window_chunking(text, window_size=100, step_size=50)
    print(f"\n2. 滑动窗口切片 (window=100, step=50)")
    print(f"   切片数量：{len(sliding)}")
    for i, chunk in enumerate(sliding[:4]):
        print(f"   [{i+1}] {len(chunk)}字：{chunk[:40]}...")

    ai = ai_chunking(text, max_chunk_size=150)
    print(f"\n3. AI 辅助切片 (按段落)")
    print(f"   切片数量：{len(ai)}")
    for i, chunk in enumerate(ai[:4]):
        print(f"   [{i+1}] {len(chunk)}字：{chunk[:40]}...")

    summary_chunks, summary_list = summary_chunking(text)
    print(f"\n4. 概要生成切片")
    print(f"   切片数量：{len(summary_chunks)}")
    for i, (chunk, summ) in enumerate(zip(summary_chunks[:4], summary_list[:4])):
        print(f"   [{i+1}] 摘要：{summ}")


def demo_quality_evaluation():
    """演示切片质量评估"""
    print(f"\n-- 示例 2: 切片质量评估")

    text = """人工智能（AI）是模拟人类智能的计算机科学领域。它包括机器学习、深度学习、自然语言处理等多个分支。

机器学习是 AI 的核心技术之一。它通过训练数据让计算机自动学习规律，无需显式编程。

深度学习使用多层神经网络模拟人脑。它在图像识别、语音识别等领域取得了突破性进展。

自然语言处理让计算机理解和生成人类语言。应用包括机器翻译、情感分析、智能客服等。"""

    methods = {
        "固定字符 (100)": fixed_chunking(text, 100),
        "固定字符 (150)": fixed_chunking(text, 150),
        "滑动窗口 (100/50)": sliding_window_chunking(text, 100, 50),
        "AI 辅助 (段落)": ai_chunking(text),
    }

    print(f"文本长度：{len(text)} 字符\n")
    print("质量评估结果:\n")

    results = []
    for name, chunks in methods.items():
        metrics = evaluate_chunk_quality(chunks)
        results.append((name, metrics))
        print(f"【{name}】")
        print(f"  切片数：{metrics['total_chunks']}")
        print(f"  平均长度：{metrics['avg_length']:.1f} 字")
        print(f"  长度波动：{metrics['std_length']:.1f} (越小越均匀)")
        print(f"  完整句比例：{metrics['complete_ratio']*100:.0f}%")
        print()

    best = max(results, key=lambda x: x[1]['complete_ratio'])
    print(f"推荐：语义完整性最佳 → {best[0]}")


def demo_search_comparison():
    """演示检索效果对比"""
    print(f"\n-- 示例 3: 检索效果模拟对比")

    text = """人工智能（AI）是模拟人类智能的计算机科学领域。它包括机器学习、深度学习、自然语言处理等多个分支。

机器学习是 AI 的核心技术之一。它通过训练数据让计算机自动学习规律，无需显式编程。监督学习、无监督学习、强化学习是三种主要的学习范式。

深度学习是机器学习的子集，使用多层神经网络模拟人脑。它在图像识别、语音识别等领域取得了突破性进展。卷积神经网络（CNN）和循环神经网络（RNN）是两种经典架构。

自然语言处理（NLP）让计算机理解和生成人类语言。应用包括机器翻译、情感分析、智能客服等。近年来，大语言模型（LLM）在 NLP 领域引发了革命。"""

    fixed_chunks = fixed_chunking(text, 100)
    sliding_chunks = sliding_window_chunking(text, 100, 50)
    ai_chunks = ai_chunking(text)

    queries = [
        "机器学习有几种学习方式？",
        "深度学习在图像识别的应用",
        "自然语言处理包括什么",
    ]

    for query in queries:
        print(f"\n查询：{query}")
        print("-" * 50)

        for method_name, chunks in [
            ("固定字符", fixed_chunks),
            ("滑动窗口", sliding_chunks),
            ("AI 辅助", ai_chunks),
        ]:
            results = simulate_search(query, chunks)
            top_score = results[0][1] if results else 0
            print(f"  {method_name}: 最高匹配度={top_score}")
        print()


def demo_file_comparison():
    """演示文件切片对比"""
    print(f"\n-- 示例 4: 文件切片对比")

    import os
    file_path = os.path.join(os.path.dirname(__file__), "..", "data", "txt", "milvus_intro.txt")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"文件不存在：{file_path}")
        return

    print(f"文件：milvus_intro.txt")
    print(f"文本长度：{len(text)} 字符\n")

    methods = {
        "固定字符 (200)": fixed_chunking(text, 200),
        "滑动窗口 (200/100)": sliding_window_chunking(text, 200, 100),
        "AI 辅助 (段落)": ai_chunking(text),
    }

    print("切片结果对比：\n")
    for name, chunks in methods.items():
        print(f"【{name}】")
        print(f"  切片数量：{len(chunks)}")
        print(f"  平均长度：{sum(len(c) for c in chunks)/len(chunks):.1f} 字")
        print(f"  预览：")
        for i, chunk in enumerate(chunks[:2]):
            preview = chunk[:50].replace('\n', ' ')
            print(f"    [{i+1}] {preview}...")
        print()


def comprehensive_comparison_table():
    """打印综合对比表"""
    print(f"\n-- 示例 5: 综合评分表")

    print("""
┌────────────────────────────────────────────────────────────────────┐
│ 四种切片方法综合对比                                                │
├─────────────────┬──────────┬──────────┬──────────┬─────────────────┤
│      维度        │ 固定规则  │ 滑动窗口  │  AI 辅助   │   概要生成       │
├─────────────────┼──────────┼──────────┼──────────┼─────────────────┤
│ 实现难度        │  ⭐        │  ⭐⭐       │  ⭐⭐⭐      │   ⭐⭐⭐          │
│ 处理速度        │  ⭐⭐⭐⭐     │  ⭐⭐⭐      │  ⭐⭐       │   ⭐⭐           │
│ 语义完整性      │  ⭐⭐       │  ⭐⭐⭐      │  ⭐⭐⭐⭐     │   ⭐⭐⭐⭐         │
│ 检索精度        │  ⭐⭐       │  ⭐⭐⭐      │  ⭐⭐⭐⭐     │   ⭐⭐⭐⭐⭐        │
│ 存储成本        │  ⭐⭐⭐⭐     │  ⭐⭐⭐      │  ⭐⭐⭐⭐     │   ⭐⭐           │
│ 适用文档        │ 任意       │ 任意      │ 结构化    │ 长文档          │
│ 成本            │  无        │  无        │  API 费用   │   API 费用        │
└─────────────────┴──────────┴──────────┴──────────┴─────────────────┘

场景推荐:
  - 快速原型 → 固定规则（实现最快）
  - 生产环境 → 滑动窗口（通用首选）
  - 高质量 RAG → AI 辅助 + 概要
  - 成本敏感 → 滑动窗口（无需 API）
  - 实时处理 → 固定规则/滑动窗口
""")


def chunking_selection_guide():
    """打印切片方法选择指南"""
    print(f"\n-- 示例 6: 切片方法选择指南")

    print("""
如何选择切片方法？

Q1: 有 Token 预算吗？
├─ 没有 → 滑动窗口（推荐默认）
└─ 有 → Q2

Q2: 文档平均长度？
├─ <500 字 → AI 辅助切片（按语义分）
└─ >500 字 → Q3

Q3: 检索精度要求？
├─ 高 → 概要生成切片（摘 + 原文混合检索）
└─ 中 → AI 辅助切片

推荐配置:
1. 通用配置：滑动窗口，window=500, step=250
2. 高质量配置：AI 辅助 + 概要生成
3. 快速原型配置：固定字符，chunk_size=300
4. 长文档配置：概要生成 + 混合检索
""")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  切片方法对比（Chunking Comparison）")
    print("=" * 70 + "\n")

    demo_all_methods()
    demo_quality_evaluation()
    demo_search_comparison()
    demo_file_comparison()
    comprehensive_comparison_table()
    chunking_selection_guide()

    print("\n" + "=" * 70)
    print("  四种切片方法各有优劣，根据场景选择最合适的")
    print("  下一步：03_retrieval_methods/（检索方法）")
    print("=" * 70 + "\n")
