# =============================================================================
# 04_summary_chunking — 概要生成切片
# =============================================================================
# 用途：学习为每个切片生成摘要，用摘要进行检索，原文用于回答
# 难度：⭐⭐⭐（3 星）
# =============================================================================
# 核心概念：
#   - 为文档切片生成摘要
#   - 用摘要进行检索，原文用于回答
#   - 适合长文档检索场景
# 工作流程：
#   1. 将文档切片
#   2. 为每个切片生成一句话摘要
#   3. 检索时：用问题匹配摘要
#   4. 回答时：将原文 + 问题交给 LLM
# =============================================================================

import os
import re
from dotenv import load_dotenv
load_dotenv()


def generate_summary_with_llm(chunk, summary_length=50, api_key=None):
    """
    使用阿里云百炼 DashScope API 生成摘要

    参数:
        chunk: 输入切片
        summary_length: 摘要最大长度
        api_key: API Key
    返回:
        摘要字符串
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

    prompt = f"""请为以下文本生成一个简短的摘要（{summary_length}字以内）：

{chunk}

摘要要求：
1. 概括核心内容
2. 保持语义完整
3. 适合用于检索匹配
4. 直接返回摘要内容，不要其他解释

摘要："""

    print(f"调用模型：qwen-plus")
    print(f"原文长度：{len(chunk)} 字符")
    print(f"摘要长度限制：{summary_length} 字")
    print("-" * 40)

    try:
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        )

        summary = completion.choices[0].message.content.strip()
        if summary.startswith('"') and summary.endswith('"'):
            summary = summary[1:-1]

        print(f"生成的摘要: {summary}")
        print("-" * 40)

        return summary

    except Exception as e:
        print(f"API 调用失败：{e}")
        print("退化为模拟摘要生成...")

def search_by_summary(query, chunks, summaries, top_k=2):
    """
    基于摘要的检索

    用问题匹配摘要，返回最相关的原文切片

    参数:
        query: 查询问题
        chunks: 原文切片列表
        summaries: 摘要列表
        top_k: 返回最相关的 k 个结果
    返回:
        (relevant_chunks, relevant_summaries, scores) 元组
    """
    query_words = set(query.lower())

    scores = []
    for i, summary in enumerate(summaries):
        summary_words = set(summary.lower())
        overlap = len(query_words & summary_words)
        scores.append((i, overlap))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, score in scores[:top_k]]

    relevant_chunks = [chunks[i] for i in top_indices]
    relevant_summaries = [summaries[i] for i in top_indices]
    top_scores = [score for _, score in scores[:top_k]]

    return relevant_chunks, relevant_summaries, top_scores


def rag_with_summary_retrieval(query, chunks, summaries, use_llm=False, api_key=None):
    """
    使用摘要检索的 RAG 流程

    1. 用问题匹配摘要
    2. 获取相关原文
    3. 生成回答（可调用真实 LLM）

    参数:
        query: 用户问题
        chunks: 原文切片列表
        summaries: 摘要列表
        use_llm: 是否使用真实 LLM 生成回答
        api_key: API Key
    返回:
        (answer, relevant_chunks, scores) 元组
    """
    if api_key is None:
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIYUN_API_KEY")

    # 1. 用问题匹配摘要
    relevant_chunks, relevant_summaries, scores = search_by_summary(
        query, chunks, summaries, top_k=2
    )

    context = "\n".join(relevant_chunks)

    if use_llm and api_key:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        prompt = f"""请根据以下相关信息回答问题。

相关信息：
{context}

用户问题：{query}

请基于上述信息给出准确、简洁的回答："""

        try:
            completion = client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]
            )
            answer = completion.choices[0].message.content.strip()
        except Exception as e:
            answer = f"[LLM 调用失败：{e}]\n\n根据以下相关信息：\n\n{context}\n\n我的回答：这个问题需要基于上述信息进行分析...（模拟回答）"
    else:
        answer = f"""根据以下相关信息：

{context}

我的回答：这个问题需要基于上述信息进行分析...（模拟回答）"""

    return answer, relevant_chunks, scores


def hybrid_search_with_summary(query, chunks, summaries):
    """
    混合检索：同时使用摘要和原文

    结合两者的优势，提高召回质量

    参数:
        query: 查询问题
        chunks: 原文切片列表
        summaries: 摘要列表
    返回:
        带分数的排序结果列表 [(索引, 分数), ...]
    """
    query_words = set(query.lower())

    # 摘要检索分数
    summary_scores = []
    for i, summary in enumerate(summaries):
        overlap = len(query_words & set(summary.lower()))
        summary_scores.append(overlap)

    # 原文检索分数
    chunk_scores = []
    for i, chunk in enumerate(chunks):
        overlap = len(query_words & set(chunk.lower()))
        chunk_scores.append(overlap)

    # 加权融合（摘要权重 0.6，原文权重 0.4）
    combined_scores = [
        0.6 * s + 0.4 * c
        for s, c in zip(summary_scores, chunk_scores)
    ]

    indexed_scores = [(i, score) for i, score in enumerate(combined_scores)]
    indexed_scores.sort(key=lambda x: x[1], reverse=True)

    return indexed_scores



def demo_llm_summary():
    """演示 LLM 生成摘要（需要 API Key）"""
    print(f"\n-- 示例 2: LLM 生成摘要（阿里云百炼 API）")

    chunk = """自然语言处理（NLP）是人工智能的重要分支，它研究如何让计算机理解、解释和生成人类语言。
    NLP 的应用包括机器翻译、情感分析、文本摘要、对话系统等。
    近年来，基于 Transformer 架构的大语言模型在 NLP 领域取得了突破性进展。"""

    print(f"原文:\n{chunk}\n")

    summary = generate_summary_with_llm(chunk, summary_length=40)
    print(f"\n最终摘要：{summary}")



def compare_summary_vs_direct_chunking():
    """摘要切片 vs 直接切片对比"""
    print(f"\n-- 示例 6: 摘要切片 vs 直接切片对比")

    print("""
摘要切片 vs 直接切片对比

直接切片检索：
  - 检索内容：切片原文
  - 优点：简单直接，无需额外处理
  - 缺点：长文本有噪声，匹配精度可能低
  - 适用：短文档（<500 字符）

摘要切片检索：
  - 检索内容：切片摘要
  - 优点：简洁精准，减少噪声干扰
  - 缺点：需要生成摘要，增加成本
  - 适用：长文档（>500 字符）

使用建议:

1. 文档较短（<500 字符）
   -> 直接用原文检索（简单高效）

2. 文档中等（500-2000 字符）
   -> 摘要 + 原文混合检索（平衡效果和成本）

3. 文档很长（>2000 字符）
   -> 用摘要检索（减少噪声）
""")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  概要生成切片（Summary Chunking）")
    print("=" * 60)


    # 示例 2 需要 API Key，如果不需要可以注释掉
    # demo_llm_summary()

    compare_summary_vs_direct_chunking()
