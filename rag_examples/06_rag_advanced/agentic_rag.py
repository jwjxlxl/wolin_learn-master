# =============================================================================
# agentic_rag — Agentic RAG（反思型循环检索）
# =============================================================================
# 用途：学习比「一次检索 + 直接生成」更进阶的 RAG 范式——Agentic RAG。
#       核心是把检索变成 Agent 手里可以反复调用的「工具」，形成一个循环：
#       检索 → 反思（资料够不够回答？）→ 决定下一步（改写问题 / 拆子问题）
#       → 再检索……直到满意，或达到轮数上限。
# 难度：⭐⭐⭐⭐⭐（5 星）
#
# 核心概念（与普通 RAG 的区别）：
#   普通 RAG：query → 检索一次 → 生成（一条直线，检索效果不好也"认命"了）
#   Agentic RAG：query ⇄ 检索 ⇄ 反思 ⇄ 改写/拆分（一个循环，会自我纠错）
#
# 复用：
#   - 检索：utils/hybrid_retriever.py 的 HybridRetriever（混合检索，含 BM25）
#   - 向量：embedding_examples/aliyun_embedding.py 的 AliyunEmbedding
#   - 模型：utils/model_utils.py 的 get_model()（默认 Ollama 本地，无需 API Key）
#
# 依赖：
#   - 反思 / 改写 / 拆分 / 生成：Ollama 本地 qwen3.5:2b（运行 `ollama serve`）
#   - 检索：Milvus 服务运行中 + ALIYUN_API_KEY（Embedding）
# =============================================================================

import json
import re
import sys

from dotenv import load_dotenv
load_dotenv()

from pymilvus import MilvusClient
from rag_examples.milvus_config import MILVUS_URI, DEFAULT_DIMENSION
from rag_examples.utils.hybrid_retriever import HybridRetriever
from rag_examples.embedding_examples.aliyun_embedding import AliyunEmbedding
from utils.model_utils import get_model

# Windows 控制台编码
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


# =============================================================================
# 第一部分：Prompt 模板（反思 / 改写 / 拆分 / 生成）
# =============================================================================

JUDGE_PROMPT = """你是 RAG 系统的「反思质检员」。请判断当前检索到的资料是否足以回答用户问题，并决定下一步动作。

规则：
1. 只输出一个 JSON 对象，不要输出任何其他文字。
2. JSON 必须包含三个键：
   - sufficient：布尔值，资料是否足以回答问题（true 或 false）
   - action：字符串，下一步动作，只能是 answer / rewrite / decompose 之一
       · answer：资料已足够，可以直接回答
       · rewrite：资料不足，需要改写问题后重新检索
       · decompose：问题太复杂，需要拆成多个子问题分别检索
   - missing_info：字符串，sufficient 为 false 时说明还缺什么信息；否则填"无"

用户问题：{query}

当前检索到的资料：
{context_block}

请只输出 JSON："""

REWRITE_PROMPT = """你是一个「查询改写」助手。用户的问题第一次检索效果不佳，请你根据「缺失信息」把问题改写成一句更精准、更适合检索的提问。

规则：
1. 只输出改写后的问题，不要解释、不要回答。
2. 保留原意，补全缺失信息中提到的内容，去掉口语和模糊表述。

原始问题：{query}
缺失信息：{missing_info}

改写后的问题："""

DECOMPOSE_PROMPT = """你是一个「查询拆分」助手。请把用户这个复杂问题拆分成多个独立的子问题，便于分别检索。

规则：
1. 每个子问题单独占一行，用「数字. 」开头。
2. 只输出子问题列表，不要回答、不要解释。

用户问题：{query}

子问题："""

ANSWER_PROMPT = """你是一个严谨的知识问答助手。请仅根据下面的【参考资料】回答用户问题，不要编造资料之外的内容。

参考资料：
{context_block}

用户问题：{query}

回答："""


# =============================================================================
# 第二部分：工具函数（解析 / 调用 / 上下文组装）
# =============================================================================

def _call_llm(llm, prompt):
    """调用模型，失败时给出友好提示。"""
    try:
        return llm.invoke(prompt).content.strip()
    except Exception as e:
        raise RuntimeError(
            f"模型调用失败（{e}）。请确认已本地运行 `ollama serve` 并拉取 qwen3.5:2b，"
            "或改用 get_model('qwen') 走云端。"
        )


def _extract_json(text):
    """从模型输出中提取 JSON 对象（容忍模型多说的废话/代码块围栏）。"""
    text = re.sub(r"```(?:json)?", "", text).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("未找到 JSON 对象")
    return json.loads(m.group(0))


def build_context_block(contexts):
    """把上下文列表编号后拼成一段文本（复用 HybridRetriever.retrieve 的输出格式）。"""
    return "\n".join(f"[{c['index']}] {c['text']}" for c in contexts)


def _parse_lines(text):
    """把模型输出的多行文本解析成干净的子问题列表（去掉序号/项目符号）。"""
    queries = []
    for line in text.strip().splitlines():
        line = re.sub(r"^\s*(?:\d+[.、)）]\s*|[-*·]\s*)+", "", line).strip()
        if line:
            queries.append(line)
    return queries


# =============================================================================
# 第三部分：Agentic RAG 的四个动作（反思 / 改写 / 拆分 / 生成）
# =============================================================================

def llm_judge(llm, query, contexts):
    """② 反思：判断当前资料是否足以回答原问题，并给出下一步动作。

    返回:
        dict: {"sufficient": bool, "action": "answer|rewrite|decompose", "missing_info": str}
    """
    prompt = JUDGE_PROMPT.format(query=query, context_block=build_context_block(contexts))
    raw = _call_llm(llm, prompt)
    try:
        data = _extract_json(raw)
    except Exception:
        # 反思失败就保守判定「不足 → 改写」，保证循环还能继续走
        return {"sufficient": False, "action": "rewrite", "missing_info": "资料不够充分"}
    return {
        "sufficient": bool(data.get("sufficient", False)),
        "action": data.get("action", "rewrite"),
        "missing_info": data.get("missing_info", ""),
    }


def rewrite_query(llm, query, missing_info):
    """④a 改写问题：根据缺失信息，把模糊/口语化的问题改写成精准检索式。"""
    prompt = REWRITE_PROMPT.format(query=query, missing_info=missing_info)
    return _call_llm(llm, prompt).strip('"“”‘’')


def decompose_query(llm, query):
    """④b 拆分问题：把复杂问题拆成多个独立子问题。"""
    prompt = DECOMPOSE_PROMPT.format(query=query)
    return _parse_lines(_call_llm(llm, prompt))


def llm_answer(llm, query, contexts):
    """③ 生成：基于最终上下文回答问题。"""
    prompt = ANSWER_PROMPT.format(query=query, context_block=build_context_block(contexts))
    return _call_llm(llm, prompt)


def multi_query_retrieve(retriever, sub_queries, top_k=3):
    """对每个子问题分别检索，再按文本去重、保留最高分、重新编号。"""
    best = {}
    for q in sub_queries:
        for item in retriever.retrieve(q, top_k=top_k):
            text = item["text"]
            if text not in best or item["score"] > best[text]["score"]:
                best[text] = item
    ordered = sorted(best.values(), key=lambda x: x["score"], reverse=True)
    return [
        {"index": i, "text": r["text"], "score": round(r["score"], 4)}
        for i, r in enumerate(ordered, 1)
    ]


# =============================================================================
# 第四部分：主循环（Agentic RAG 的核心）
# =============================================================================

def agentic_rag(query, retriever, llm, max_rounds=3, top_k=5):
    """Agentic RAG 反思循环：检索 → 反思 → 决策 → 再检索，直到满意或达到轮数上限。

    参数:
        query: 用户原始问题
        retriever: HybridRetriever 实例（检索工具）
        llm: 聊天模型（反思 / 改写 / 拆分 / 生成共用）
        max_rounds: 最大轮数上限
        top_k: 每轮检索条数
    返回:
        (answer, trace): 最终回答 + 每一轮的反思过程记录（用于打印循环过程）
    """
    trace = []
    current_query = query

    for round_no in range(1, max_rounds + 1):
        # ① 检索：检索只是 Agent 手里的一个工具，用当前问题调它
        contexts = retriever.retrieve(current_query, top_k=top_k)

        # ② 反思：让 LLM 当质检员，判断资料够不够回答
        # 判断模式使用不同的
        judge_llm = get_model("deepseek")
        verdict = llm_judge(judge_llm, query, contexts)
        trace.append({
            "round": round_no,
            "query": current_query,
            "contexts": contexts,
            "verdict": verdict,
        })

        # ③ 决策：够了就收工
        if verdict["sufficient"] or verdict["action"] == "answer":
            return llm_answer(llm, query, contexts), trace

        # 不够 → 决定下一步
        if verdict["action"] == "rewrite":
            # 改写问题，下一轮用新问题再查
            current_query = rewrite_query(llm, query, verdict["missing_info"])
        elif verdict["action"] == "decompose":
            # 拆成子问题，分别检索再合并，直接作为最终上下文回答
            sub_queries = decompose_query(llm, query)
            merged = multi_query_retrieve(retriever, sub_queries, top_k=top_k)
            trace[-1]["decomposed"] = sub_queries
            trace[-1]["merged_contexts"] = merged
            return llm_answer(llm, query, merged), trace
        else:
            # 动作不明确时，退化为改写
            current_query = rewrite_query(llm, query, verdict["missing_info"])

    # 兜底：轮数用尽，用原始问题再查一次直接回答
    contexts = retriever.retrieve(query, top_k=top_k)
    return llm_answer(llm, query, contexts), trace


# =============================================================================
# 第五部分：演示
# =============================================================================

def explain_agentic_rag():
    """讲解 Agentic RAG 的核心思想（纯文字，无需任何依赖）。"""
    print(f"\n-- 概念: 普通 RAG vs Agentic RAG")

    print("""
普通 RAG（一条直线）：
  用户问题 → 检索一次 → 把结果塞给 LLM → 生成答案
  问题：检索只做一次，效果不好就"认命"了，不会自我纠错。

Agentic RAG（一个循环）：
  用户问题 ──→ ① 检索（工具）
      ↑            ↓
   ④ 改写/拆子  ② 反思（LLM 当质检员：资料够不够回答？）
      ↑            ↓
      └────── ③ 决策：不够就再来一轮，够了就生成答案

  「检索」从"流程里的一步"变成了"Agent 手里可反复调用的工具"，
  由 LLM 自己判断、自己纠错，直到满意或达到轮数上限（max_rounds）。
""")


def prepare_documents():
    """准备知识库文档（覆盖检索相关概念，便于演示反思/改写/拆分）。"""
    return [
        "RRF（倒数排名融合）把多路检索结果的排名取倒数再相加，无需调参就能融合异构分数。",
        "向量检索把文本编码成稠密向量，通过计算余弦相似度查找语义相近的内容，擅长理解同义表达。",
        "BM25 是基于关键词匹配的稀疏检索算法，擅长精确匹配专有名词和术语，但对同义词不敏感。",
        "混合检索同时跑稠密向量检索和 BM25 稀疏检索，把两路结果融合，兼顾语义和关键词，召回率更高。",
        "RRF（倒数排名融合）把多路检索结果的排名取倒数再相加，无需调参就能融合异构分数。",
        "RAG（检索增强生成）先从知识库检索相关片段，再让大模型基于这些片段生成答案，减少幻觉。",
        "查询改写把用户口语化、模糊的问题改写成标准检索式，从而提升召回质量。",
        "查询拆分把一个复杂问题拆成多个独立子问题分别检索，再合并去重，覆盖更全面。",
    ]


def setup_retriever(collection_name="agentic_rag_demo"):
    """建集合 + 插数据（复用 HybridRetriever.create_collection / insert）。"""
    client = MilvusClient(uri=MILVUS_URI)
    emb = AliyunEmbedding()
    retriever = HybridRetriever(client, collection_name, emb)

    documents = prepare_documents()
    dense_vectors = [emb.embed(d) for d in documents]
    retriever.create_collection(dim=DEFAULT_DIMENSION)
    retriever.insert(documents, dense_vectors)
    print(f"  ✓ 知识库就绪（{len(documents)} 篇文档）")
    return retriever


def print_trace(trace):
    """打印循环的每一轮：检索问题、命中结果、反思结论、拆分动作。"""
    for step in trace:
        verdict = step["verdict"]
        print(f"\n  ── 第 {step['round']} 轮 ──")
        print(f"  检索问题：{step['query']}")
        for c in step["contexts"]:
            print(f"    [{c['index']}] ({c['score']:.4f}) {c['text'][:28]}")
        if step.get("decomposed"):
            print(f"  拆分为 {len(step['decomposed'])} 个子问题：")
            for i, q in enumerate(step["decomposed"], 1):
                print(f"    {i}. {q}")
            print(f"  合并检索到 {len(step['merged_contexts'])} 条上下文")
        print(f"  反思：充分={verdict['sufficient']}，动作={verdict['action']}，"
              f"缺失={verdict['missing_info'] or '无'}")


def run_demo():
    """完整演示：一个模糊问题触发「改写」循环 + 一个复合问题触发「拆子问题」。"""
    # 反思/改写/拆分/生成共用同一个模型（默认 Ollama 本地，无需 API Key）
    # 想走云端可改成 get_model("qwen")，但需 ALIYUN_API_KEY
    llm = get_model("qwen")

    # 检索工具：复用 HybridRetriever（需 Milvus + ALIYUN_API_KEY）
    retriever = setup_retriever()

    # ---- 示例 1：模糊问题 → 反思说「不足」→ 改写后重新检索 ----
    print(f"\n-- 示例 1: 模糊问题 → 触发「改写」循环")
    q1 = "两个结果整合后的方法是什么"
    print(f"\n  用户问题：{q1}")
    answer1, trace1 = agentic_rag(q1, retriever, llm, max_rounds=3, top_k=5)
    print_trace(trace1)
    print(f"\n  最终回答：\n  {answer1}")

    # ---- 示例 2：复合问题 → 反思说「太复杂」→ 拆成子问题分别检索 ----
    print(f"\n-- 示例 2: 复合问题 → 触发「拆子问题」")
    q2 = "混合检索和向量检索有什么区别？RRF 是怎么融合结果的？"
    print(f"\n  用户问题：{q2}")
    answer2, trace2 = agentic_rag(q2, retriever, llm, max_rounds=3, top_k=5)
    print_trace(trace2)
    print(f"\n  最终回答：\n  {answer2}")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  Agentic RAG — 反思型循环检索")
    print("=" * 70)

    # explain_agentic_rag()

    try:
        run_demo()
    except RuntimeError as e:
        print(f"\n✗ {e}")
    except Exception as e:
        print(f"\n✗ 运行失败：{e}")
        print("  提示：检索部分需要 Milvus 服务运行中 + ALIYUN_API_KEY")

    print("\n" + "=" * 70)
    print("  完成！检索不再是「一步到位」，而是会反思、会纠错的循环。")
    print("=" * 70)
