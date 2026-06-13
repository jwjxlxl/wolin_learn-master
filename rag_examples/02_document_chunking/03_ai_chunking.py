# =============================================================================
# 03_ai_chunking — AI 辅助切片
# =============================================================================
# 用途：学习使用 AI（大语言模型）按语义边界智能切片
# 难度：⭐⭐⭐（3 星）
# =============================================================================
# 核心概念：
#   - 利用 AI 识别语义边界，按主题/内容自然分段
#   - 切片质量最高，但需要 API 调用
# 工作原理：
#   1. 将文档发送给 LLM
#   2. LLM 分析内容，识别主题/语义边界
#   3. LLM 返回建议的切分点
#   4. 按切分点分割文档
# =============================================================================

import os
import re
import json
from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# 核心切片函数
# =============================================================================

def mock_ai_chunking(text, max_chunk_size=200):
    """
    模拟 AI 切片（不使用真实 API）

    用启发式规则模拟"语义边界"识别
    实际使用时请替换为真实的 LLM 调用

    参数:
        text: 输入文本
        max_chunk_size: 最大切片大小
    返回:
        切片列表
    """
    paragraphs = re.split(r'\n\s*\n', text)

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para
        else:
            current_chunk += "\n\n" + para if current_chunk else para

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def ai_chunking_with_llm(text, max_chunks=5, api_key=None):
    """
    使用阿里云百炼 DashScope API 进行语义切片

    参数:
        text: 输入文本
        max_chunks: 期望的最大切片数
        api_key: API Key，默认从环境变量读取
    返回:
        切片列表
    """
    from openai import OpenAI

    if api_key is None:
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIYUN_API_KEY")
        if not api_key:
            raise ValueError("未找到 API Key，请设置环境变量 DASHSCOPE_API_KEY 或 ALIYUN_API_KEY")

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    prompt = f"""请分析以下文本，找出{max_chunks}个语义边界（最适合切分的位置）。

文本：
{text}

请按以下 JSON 格式返回切分点（每个切分点是文本中的起始索引位置）：
{{{{
    "split_points": [位置 1, 位置 2, ...],
    "reasons": ["为什么在这里切分的原因 1", "原因 2", ...]
}}}}

注意：
1. 切分点应该是段落或主题的边界
2. 保持每个切片的语义完整性
3. 返回纯 JSON，不要其他解释
4. 第一个切分点应该是 0
"""

    print(f"\n调用模型：qwen-plus")
    print(f"文本长度：{len(text)} 字符")
    print("-" * 40)

    try:
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        )

        response_text = completion.choices[0].message.content
        print(f"API 响应原始内容:\n{response_text[:300]}...")

        # 清理可能的 markdown 标记
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        response = json.loads(response_text)
        split_points = response.get("split_points", [])
        reasons = response.get("reasons", [])

        print(f"\nLLM 返回的切分点: {split_points}")
        print(f"切分原因: {reasons[:3]}...")

        # 按切分点分割文本
        chunks = []
        if split_points:
            if split_points[0] != 0:
                split_points.insert(0, 0)

            for i, start in enumerate(split_points):
                end = split_points[i + 1] if i + 1 < len(split_points) else len(text)
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)
        else:
            print("  警告：API 未返回有效切分点，退化为按段落分割")
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            chunks = paragraphs

        print(f"生成的切片数量：{len(chunks)}")
        return chunks

    except Exception as e:
        print(f"API 调用失败：{e}")
        print("退化为按段落分割...")
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        return paragraphs


def sentence_clustering_chunking(text, sentences_per_chunk=3):
    """
    基于句子聚类的切片

    简化版"智能"切片：
    1. 先分割成句子
    2. 将相近的句子聚在一起
    3. 形成语义连贯的切片

    参数:
        text: 输入文本
        sentences_per_chunk: 每个切片的句子数
    返回:
        切片列表
    """
    sentences = re.split(r'[.!?!.!?。！？]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        group = sentences[i:i + sentences_per_chunk]
        chunk = "。".join(group) + "。"
        chunks.append(chunk)

    return chunks


def fixed_char_chunking(text, chunk_size=150):
    """固定字符切片（用于对比）"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def sliding_window_chunking(text, window_size=150, step_size=75):
    """滑动窗口切片（用于对比）"""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+window_size])
        start += step_size
        if start >= len(text):
            break
    return chunks


# =============================================================================
# 演示函数
# =============================================================================

def demo_mock_ai_chunking():
    """演示模拟 AI 切片"""
    print(f"\n-- 示例 1: 模拟 AI 切片（无 API 时的演示）")

    text = """人工智能（AI）是模拟人类智能的计算机科学领域。它包括机器学习、深度学习、自然语言处理等多个分支。

机器学习是 AI 的核心技术之一。它通过训练数据让计算机自动学习规律，无需显式编程。常见的机器学习算法包括决策树、神经网络、支持向量机等。

深度学习是机器学习的子集，使用多层神经网络模拟人脑。它在图像识别、语音识别等领域取得了突破性进展。卷积神经网络（CNN）和循环神经网络（RNN）是两种经典架构。

自然语言处理（NLP）让计算机理解和生成人类语言。应用包括机器翻译、情感分析、智能客服等。近年来，大语言模型（LLM）在 NLP 领域引发了革命。

计算机视觉（CV）让计算机能够"看懂"图像和视频。应用包括人脸识别、物体检测、医学图像分析等。深度学习大幅提升了 CV 的性能。"""

    print(f"原始文本：{len(text)} 字符")
    print(f"最大切片大小：200 字符\n")

    chunks = mock_ai_chunking(text, max_chunk_size=200)

    print(f"切片数量：{len(chunks)}\n")
    for i, chunk in enumerate(chunks):
        print(f"【语义切片 {i+1}】({len(chunk)} 字符)")
        print(f"  主题：{chunk[:30]}...")
        print(f"  内容预览：{chunk[:80]}...")
        print()


def demo_real_ai_chunking():
    """演示真实 AI 切片（使用阿里云百炼 API）"""
    print(f"\n-- 示例 2: 真实 AI 切片（阿里云百炼 API）")

    text = """人工智能（AI）是模拟人类智能的计算机科学领域。它试图理解智能的本质，并生产出能以人类智能相似的方式做出反应的智能机器。该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。

机器学习是 AI 的核心技术之一，通过训练数据让计算机自动学习规律，无需显式编程。监督学习、无监督学习、强化学习是三种主要的学习范式。常见的算法包括决策树、随机森林、支持向量机、神经网络等。

深度学习是机器学习的子集，使用多层神经网络模拟人脑。卷积神经网络（CNN）在图像识别领域取得突破，循环神经网络（RNN）和 Transformer 架构在自然语言处理领域表现优异。深度学习的成功得益于大数据、强算力和算法创新。

自然语言处理（NLP）让计算机理解和生成人类语言。从早期的规则方法到统计方法，再到现在的深度学习方法，NLP 取得了长足进步。机器翻译、情感分析、文本生成、对话系统等应用已经深入人们的生活。

计算机视觉（CV）让计算机能够"看懂"图像和视频。目标检测、图像分割、人脸识别、姿态估计等任务都有了突破性进展。自动驾驶、医学影像分析、工业质检等应用正在改变各个行业。"""

    print(f"测试文本长度：{len(text)} 字符")
    print(f"期望切片数：3\n")

    chunks = ai_chunking_with_llm(text, max_chunks=3)

    print("\n切片结果预览：")
    for i, chunk in enumerate(chunks):
        print(f"\n【切片 {i+1}】({len(chunk)} 字符)")
        print(f"  内容：{chunk[:80]}...")


def demo_sentence_clustering():
    """演示基于句子聚类的切片"""
    print(f"\n-- 示例 3: 基于句子聚类的切片")

    text = """人工智能是计算机科学的一个分支。它试图理解智能的本质。机器学习是 AI 的核心技术。
深度学习使用多层神经网络。它在图像识别领域取得突破。自然语言处理让计算机理解语言。
计算机视觉让机器看懂图像。推荐系统根据偏好推荐内容。知识图谱用图结构表示知识。
大语言模型基于海量文本训练。它能生成高质量的文本。AI 正在改变我们的生活。"""

    print(f"每切片句子数：3\n")

    chunks = sentence_clustering_chunking(text, sentences_per_chunk=3)

    print(f"切片数量：{len(chunks)}\n")
    for i, chunk in enumerate(chunks):
        sentence_count = len([s for s in chunk.split('。') if s.strip()])
        print(f"【聚类切片 {i+1}】({sentence_count} 句子)")
        print(f"  {chunk}")
        print()


def demo_langchain_splitter():
    """演示 LangChain 文本分割器"""
    print(f"\n-- 示例 4: LangChain 文本分割器")

    try:
        from langchain_text_splitters import (
            CharacterTextSplitter,
            RecursiveCharacterTextSplitter,
        )

        text = """人工智能（AI）是模拟人类智能的计算机科学领域。
它包括机器学习、深度学习、自然语言处理等多个分支。
机器学习通过训练数据让计算机自动学习规律。
深度学习使用多层神经网络模拟人脑。
自然语言处理让计算机理解和生成人类语言。
计算机视觉让计算机能够"看懂"图像和视频。"""

        print("1. CharacterTextSplitter（字符分割器）")
        print("-" * 40)

        char_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=100,
            chunk_overlap=20,
            length_function=len
        )

        char_chunks = char_splitter.split_text(text)
        print(f"切片数量：{len(char_chunks)}")
        for i, chunk in enumerate(char_chunks[:3]):
            print(f"  [{i+1}] {chunk[:50]}...")

        print()
        print("2. RecursiveCharacterTextSplitter（递归字符分割器）")
        print("-" * 40)

        recursive_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", ".", " "],
            chunk_size=100,
            chunk_overlap=20,
            length_function=len
        )

        recursive_chunks = recursive_splitter.split_text(text)
        print(f"切片数量：{len(recursive_chunks)}")
        for i, chunk in enumerate(recursive_chunks[:3]):
            print(f"  [{i+1}] {chunk[:50]}...")

    except ImportError as e:
        print(f"需要安装：pip install langchain-text-splitters")
        print(f"错误：{e}")


def compare_chunking_methods(text):
    """
    对比 AI 切片和固定切片的效果

    参数:
        text: 输入文本
    """
    print(f"\n-- 示例 5: AI 切片 vs 固定切片对比")

    fixed_chunks = fixed_char_chunking(text, chunk_size=150)
    sliding_chunks = sliding_window_chunking(text, window_size=150, step_size=75)
    ai_chunks = mock_ai_chunking(text, max_chunk_size=150)

    print(f"文本长度：{len(text)} 字符\n")

    avg_fixed = sum(len(c) for c in fixed_chunks) / len(fixed_chunks)
    avg_sliding = sum(len(c) for c in sliding_chunks) / len(sliding_chunks)
    avg_ai = sum(len(c) for c in ai_chunks) / len(ai_chunks)

    print(f"  固定字符切片  | 切片数: {len(fixed_chunks):2d} | 平均长度: {avg_fixed:5.1f} | 语义完整度: 一般")
    print(f"  滑动窗口切片  | 切片数: {len(sliding_chunks):2d} | 平均长度: {avg_sliding:5.1f} | 语义完整度: 良好")
    print(f"  AI 辅助切片   | 切片数: {len(ai_chunks):2d} | 平均长度: {avg_ai:5.1f} | 语义完整度: 优秀")

    print("\n说明：语义完整度是主观评估，AI 切片通常能保持更好的语义连贯性")


def ai_chunking_best_practices():
    """AI 切片的最佳实践建议"""
    print(f"\n-- 示例 6: AI 切片最佳实践")

    print("""
AI 辅助切片最佳实践

1. 选择合适的触发时机
   - 文档预处理时一次性切片（推荐）
   - 按需动态切片（灵活但慢）

2. 设计有效的 Prompt
   - 明确告知 AI 切片的用途（检索/问答）
   - 指定期望的切片数量和大小
   - 要求返回结构化格式（JSON）

3. 降低成本的方法
   - 先用规则粗切，再用 AI 精切
   - 对长文档分段处理
   - 使用本地模型（Ollama）

4. 质量保证
   - 检查切片是否为空
   - 验证切片大小是否合理
   - 人工抽检切片质量

推荐流程:

  原始文档
      |
  [预处理] 清理格式、统一编码
      |
  [粗切] 按段落/章节分割（规则方法）
      |
  [精切] 对每个粗切片段用 AI 识别语义边界
      |
  [后处理] 合并过小切片，过滤空切片
      |
  最终切片列表

成本对比:

| 方法      | 速度 | 成本 | 质量    | 推荐场景      |
|-----------|------|------|---------|---------------|
| 固定切片  | 快   | 无   | 一般    | 原型/测试     |
| 滑动窗口  | 中   | 无   | 良好    | 生产环境      |
| AI 切片   | 慢   | 有   | 优秀    | 高质量 RAG    |
| 混合方法  | 中   | 低   | 优秀    | 推荐          |
""")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  AI 辅助切片（AI-Assisted Chunking）")
    print("=" * 60)

    demo_mock_ai_chunking()

    # 示例 2 需要 API Key，如果不需要可以注释掉
    # demo_real_ai_chunking()

    demo_sentence_clustering()
    demo_langchain_splitter()

    test_text = """人工智能（AI）是模拟人类智能的计算机科学领域。它包括机器学习、深度学习、自然语言处理等多个分支。AI 从 1956 年诞生至今，已经经历了多次发展浪潮。

机器学习是 AI 的核心技术之一。它通过训练数据让计算机自动学习规律，无需显式编程。监督学习、无监督学习、强化学习是三种主要的学习范式。

深度学习是机器学习的子集，使用多层神经网络模拟人脑。它在图像识别、语音识别等领域取得了突破性进展。卷积神经网络（CNN）和循环神经网络（RNN）是两种经典架构。

自然语言处理（NLP）让计算机理解和生成人类语言。应用包括机器翻译、情感分析、智能客服等。近年来，大语言模型（LLM）在 NLP 领域引发了革命。

计算机视觉（CV）让计算机能够"看懂"图像和视频。应用包括人脸识别、物体检测、医学图像分析等。深度学习大幅提升了 CV 的性能。"""

    compare_chunking_methods(test_text)
    ai_chunking_best_practices()
