# =============================================================================
# 01_fixed_chunking — 固定规则切片
# =============================================================================
# 用途：学习最简单的文档切片方法 — 按固定字符数/段落/句子切割文档
# 难度：⭐（1 星）
# =============================================================================
# 核心概念：
#   - 按固定字符数/单词数切片，不考虑语义边界
#   - 简单快速，但可能切断语义
# 常见规则类型：
#   1. 按字符数切片：每 N 字符一切
#   2. 按段落切片：每个段落一切
#   3. 按句子切片：每个句子一切
# =============================================================================

import re


# =============================================================================
# 核心切片函数
# =============================================================================

def fixed_char_chunking(text, chunk_size=100):
    """
    按固定字符数切片

    参数:
        text: 输入文本
        chunk_size: 每个切片的字符数
    返回:
        切片列表
    """
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks


def fixed_paragraph_chunking(text):
    """
    按段落切片

    以空行作为段落分隔符

    参数:
        text: 输入文本
    返回:
        切片列表
    """
    paragraphs = text.split('\n\n')
    chunks = [p.strip() for p in paragraphs if p.strip()]
    return chunks


def fixed_sentence_chunking(text):
    """
    按句子切片

    使用标点符号作为句子分隔符

    参数:
        text: 输入文本
    返回:
        切片列表
    """
    sentences = re.split(r'[.!?。！？]', text)
    chunks = []
    for sent in sentences:
        sent = sent.strip()
        if sent:
            chunks.append(sent)
    return chunks


def fixed_chunking_with_overlap(text, chunk_size=100, overlap=20):
    """
    带重叠的固定切片

    相邻切片之间保留一定重叠，减少语义切断的影响

    参数:
        text: 输入文本
        chunk_size: 每个切片的字符数
        overlap: 重叠字符数
    返回:
        切片列表
    """
    chunks = []
    step = chunk_size - overlap

    if step <= 0:
        raise ValueError("overlap 必须小于 chunk_size")

    i = 0
    while i < len(text):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
        i += step

        if i >= len(text) - chunk_size:
            if i < len(text):
                last_chunk = text[-chunk_size:]
                if last_chunk and last_chunk != chunks[-1]:
                    chunks.append(last_chunk)
            break

    return chunks


def chunk_file(file_path, chunk_size=200):
    """
    读取文件并按固定字符数切片

    参数:
        file_path: 文件路径
        chunk_size: 每个切片的字符数
    返回:
        切片列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    chunks = fixed_char_chunking(text, chunk_size=chunk_size)
    return chunks


# =============================================================================
# 演示函数
# =============================================================================

def demo_fixed_char_chunking():
    """演示按固定字符数切片"""
    print(f"\n-- 示例 1: 按固定字符数切片")

    text = """人工智能（Artificial Intelligence，简称 AI）是计算机科学的一个分支，
它试图理解智能的本质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
人工智能从 1956 年诞生至今，已经经历了多次发展浪潮。
随着大数据、计算能力的提升和深度学习算法的突破，
人工智能在近年来取得了突飞猛进的发展，
并在医疗、金融、交通、教育等各个领域得到广泛应用。"""

    print(f"原始文本长度：{len(text)} 字符")
    print(f"切片大小：100 字符\n")

    chunks = fixed_char_chunking(text, chunk_size=100)

    print(f"切片数量：{len(chunks)}\n")
    for i, chunk in enumerate(chunks):
        print(f"【切片 {i+1}】({len(chunk)} 字符)")
        print(f"  {repr(chunk)}")
        print()


def demo_fixed_paragraph_chunking():
    """演示按段落切片"""
    print(f"\n-- 示例 2: 按段落切片")

    text = """人工智能（AI）是模拟人类智能的计算机科学领域。
它包括机器学习、深度学习、自然语言处理等多个分支。

机器学习是 AI 的核心技术之一。
它通过训练数据让计算机自动学习规律，无需显式编程。
常见的机器学习算法包括决策树、神经网络、支持向量机等。

深度学习是机器学习的子集，使用多层神经网络。
它在图像识别、语音识别等领域取得了突破性进展。
卷积神经网络（CNN）和循环神经网络（RNN）是两种经典架构。

自然语言处理（NLP）让计算机理解和生成人类语言。
应用包括机器翻译、情感分析、智能客服等。
近年来，大语言模型（LLM）在 NLP 领域引发了革命。"""

    print(f"原始文本：{len(text)} 字符")
    print(f"切片方式：按段落分割\n")

    chunks = fixed_paragraph_chunking(text)

    print(f"切片数量：{len(chunks)}\n")
    for i, chunk in enumerate(chunks):
        preview = chunk[:50] + "..." if len(chunk) > 50 else chunk
        print(f"【段落 {i+1}】({len(chunk)} 字符)")
        print(f"  {preview}")
        print()


def demo_fixed_sentence_chunking():
    """演示按句子切片"""
    print(f"\n-- 示例 3: 按句子切片")

    text = """人工智能是当今科技领域的热门话题。它正在改变我们的生活方式！
机器学习是人工智能的核心技术。深度学习又是机器学习的重要分支。
自然语言处理让计算机能够理解人类语言。这项技术有很多应用场景。
你觉得人工智能会取代人类吗？这是一个值得深思的问题。"""

    print(f"原始文本：{len(text)} 字符\n")

    chunks = fixed_sentence_chunking(text)

    print(f"切片数量：{len(chunks)}\n")
    for i, chunk in enumerate(chunks):
        print(f"【句子 {i+1}】({len(chunk)} 字符)")
        print(f"  {chunk}")
        print()


def demo_file_chunking():
    """演示读取文件并切片"""
    print(f"\n-- 示例 4: 读取文件并切片")

    file_path = "../data/txt/milvus_intro.txt"
    print(f"文件路径：{file_path}")
    print(f"切片大小：200 字符\n")

    try:
        chunks = chunk_file(file_path, chunk_size=200)
        print(f"文件切片结果：{len(chunks)} 个切片\n")

        for i, chunk in enumerate(chunks):
            preview = chunk[:80] + "..." if len(chunk) > 80 else chunk
            print(f"【切片 {i+1}】({len(chunk)} 字符)")
            print(f"  {preview}")
            print()

    except FileNotFoundError:
        print(f"文件不存在：{file_path}")
        print("  请确保 data/txt/ 目录下有测试文件")


def demo_overlap_chunking():
    """演示带重叠的固定切片"""
    print(f"\n-- 示例 5: 带重叠的固定切片")

    text = "这是一段测试文本用于演示带重叠的切片方法。这是一段测试文本用于演示带重叠的切片方法。" * 10

    print(f"原始文本长度：{len(text)} 字符")
    print(f"切片大小：50 字符")
    print(f"重叠大小：10 字符\n")

    chunks_no_overlap = fixed_char_chunking(text, chunk_size=50)
    print(f"无重叠切片数：{len(chunks_no_overlap)}")

    chunks_with_overlap = fixed_chunking_with_overlap(text, chunk_size=50, overlap=10)
    print(f"有重叠切片数：{len(chunks_with_overlap)}")

    print("\n对比最后两个切片的内容：")
    if len(chunks_no_overlap) >= 2:
        print(f"  无重叠 - 最后一片：{chunks_no_overlap[-1][:30]}...")
    if len(chunks_with_overlap) >= 2:
        print(f"  有重叠 - 最后一片：{chunks_with_overlap[-1][:30]}...")


def fixed_chunking_summary():
    """固定规则切片的优缺点总结"""
    print(f"\n-- 示例 6: 固定规则切片总结")

    print("""
固定规则切片方法对比

    方法         优点                   缺点
  ─────────────────────────────────────────────────
  固定字符数    简单、快速             无视语义边界
               容易实现               可能切断句子/段落

  固定段落数    保持段落完整           段落长度可能不均
               符合人类阅读           依赖文档格式

  固定句子数    语义相对完整           句子长度差异大
               粒度较细               可能丢失上下文

使用建议:

1. 快速原型开发
   -> 用固定字符数切片（实现最简单）

2. 文档结构清晰
   -> 用段落切片（保持结构完整）

3. 需要细粒度检索
   -> 用句子切片（粒度最小）

4. 减少语义切断
   -> 添加重叠（overlap=10-50 字符）

参数建议:

  - 字符数切片：chunk_size = 200-500
  - 重叠大小：overlap = 20-50（约 10-25 个中文字）
  - 段落切片：无需参数，依赖原文档结构
""")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  固定规则切片（Fixed Chunking）")
    print("=" * 60)

    demo_fixed_char_chunking()
    demo_fixed_paragraph_chunking()
    demo_fixed_sentence_chunking()
    demo_file_chunking()
    demo_overlap_chunking()
    fixed_chunking_summary()
