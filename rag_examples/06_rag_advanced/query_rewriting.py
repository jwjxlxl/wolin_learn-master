# =============================================================================
# query_rewriting — 查询改写（Query Rewriting）
# =============================================================================
# 用途：学习 RAG 检索优化技术「查询改写」。
#       用户的问题往往不够清晰，直接拿去检索效果差，改写后能显著提升召回。
# 难度：⭐⭐⭐⭐（4 星）
#
# 核心概念（三种改写场景）：
#   1. 太口语 → 扩写为标准表述："退货的规矩是啥"  → "退货政策和规则是什么？"
#   2. 太笼统 → 补全为具体问题："那个钱的事你给说说" → "退款到账规则是什么？"
#   3. 太复杂 → 拆分为多个子查询，分别检索后再合并去重
#
# 复用：检索部分全部复用 utils/hybrid_retriever.py 的 HybridRetriever，
#       向量生成复用 embedding_examples/aliyun_embedding.py 的 AliyunEmbedding，
#       改写模型复用 utils/model_utils.py 的 get_model()（默认 Ollama 本地）。
#       只有「改写/拆分的 Prompt + 解析 + 合并」是本节新增内容。
#
# 依赖：
#   - 查询改写：Ollama 本地模型 qwen3.5:2b（运行 `ollama serve`，无需 API Key）
#   - 检索：Milvus 服务运行中 + ALIYUN_API_KEY（Embedding）
# =============================================================================

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
# 第一部分：改写 / 拆分的 Prompt 模板（本节新增）
# =============================================================================

REWRITE_PROMPT = """你是一个「查询改写」助手。用户向知识库提问时，常常说得太口语或太笼统，导致检索效果差。请把用户的问题改写成一句清晰、标准、适合检索的正式提问。

规则：
1. 只输出改写后的问题本身，不要解释、不要回答该问题。
2. 保留原意，把口语词换成规范术语，补全省略的信息。

示例：
输入：退货的规矩是啥
输出：退货政策和规则是什么？

输入：那个钱的事你给说说
输出：退款到账时间和规则是什么？

现在请改写：
输入：{query}
输出："""

DECOMPOSE_PROMPT = """你是一个「查询拆分」助手。用户有时会一口气问多个相关但独立的问题。请把复杂问题拆分成多个独立的子查询，便于分别检索。

规则：
1. 每个子查询单独占一行，用「数字. 」开头。
2. 只输出子查询列表，不要回答、不要解释。
3. 子查询之间尽量独立，各自都能单独检索。

示例：
输入：退货和换货有什么区别？发票怎么开？
输出：
1. 退货和换货有什么区别？
2. 如何申请开具发票？

现在请拆分：
输入：{query}
输出："""


# =============================================================================
# 第二部分：改写 / 拆分（本节新增的核心）
# =============================================================================

def _call_llm(llm, prompt):
    """调用改写模型，失败时给出友好提示。"""
    try:
        return llm.invoke(prompt).content.strip()
    except Exception as e:
        raise RuntimeError(
            f"改写模型调用失败（{e}）。"
            "请确认已本地运行 `ollama serve` 并拉取 qwen3.5:2b，"
            "或改用 `get_model('qwen')` 走云端。"
        )


def rewrite_query(llm, query):
    """把口语化/笼统的问题改写为一句标准、便于检索的提问。

    参数:
        llm: LangChain 聊天模型（get_model() 返回）
        query: 用户原始问题
    返回:
        str: 改写后的标准问题
    """
    prompt = REWRITE_PROMPT.format(query=query)
    return _call_llm(llm, prompt).strip('"“”‘’')


def decompose_query(llm, query):
    """把复杂问题拆分成多个独立子查询。

    参数:
        llm: LangChain 聊天模型
        query: 用户原始（复杂）问题
    返回:
        list[str]: 子查询列表
    """
    prompt = DECOMPOSE_PROMPT.format(query=query)
    raw = _call_llm(llm, prompt)
    return _parse_lines(raw)


def _parse_lines(text):
    """把模型输出的多行文本解析成干净的子查询列表。

    去掉每行行首的序号/项目符号前缀（如 "1." / "1、" / "-" / "*"），
    并丢弃空行。
    """
    queries = []
    for line in text.strip().splitlines():
        line = re.sub(r"^\s*(?:\d+[.、)）]\s*|[-*·]\s*)+", "", line).strip()
        if line:
            queries.append(line)
    return queries


# =============================================================================
# 第三部分：检索（全部复用 HybridRetriever + AliyunEmbedding）
# =============================================================================

def prepare_documents():
    """准备知识库文档（电商客服 FAQ，最能体现查询改写的价值）"""
    return [
        "退货政策：商品签收后 7 天内支持无理由退货，需保持商品全新、吊牌完好、不影响二次销售。",
        "退款规则：退货审核通过后，退款将在 3-5 个工作日内原路退回付款账户。",
        "换货流程：商品存在质量问题可申请换货，需在签收后 15 天内提交申请并上传凭证。",
        "物流查询：下单后可在订单详情页查看物流轨迹，现货商品发货时效为 48 小时内。",
        "发票开具：订单完成后 30 天内可申请开具电子发票，抬头支持个人和企业。",
        "保修服务：电子产品享有一年保修，保修期内非人为损坏可免费维修。",
    ]


def setup_retriever(collection_name="query_rewrite_demo"):
    """建集合 + 插数据（复用 HybridRetriever.create_collection / insert）。"""
    client = MilvusClient(uri=MILVUS_URI)
    emb = AliyunEmbedding()
    retriever = HybridRetriever(client, collection_name, emb)

    documents = prepare_documents()
    dense_vectors = [emb.embed(d) for d in documents]
    retriever.create_collection(dim=DEFAULT_DIMENSION)
    retriever.insert(documents, dense_vectors)
    print(f"  ✓ 知识库就绪（{len(documents)} 篇客服 FAQ）")
    return retriever


def merge_results(result_lists):
    """合并多组检索结果：按文本去重，保留最高分，重新编号。

    多个子查询可能命中同一篇文档，直接拼接会重复，故需去重。
    """
    best = {}
    for results in result_lists:
        for item in results:
            text = item["text"]
            score = item["score"]
            if text not in best or score > best[text]["score"]:
                best[text] = {"text": text, "score": score}

    ordered = sorted(best.values(), key=lambda x: x["score"], reverse=True)
    return [
        {"index": i, "text": r["text"], "score": round(r["score"], 4)}
        for i, r in enumerate(ordered, 1)
    ]


def multi_query_retrieve(retriever, sub_queries, top_k=2):
    """对每个子查询分别检索，再合并去重（复用 HybridRetriever.retrieve）。"""
    result_lists = [retriever.retrieve(q, top_k=top_k) for q in sub_queries]
    return merge_results(result_lists)


def print_hits(title, contexts):
    """打印 retrieve() 返回的命中结果。"""
    print(f"  {title}")
    if not contexts:
        print("    （无结果）")
        return
    for c in contexts:
        print(f"    [{c['index']}] ({c['score']:.4f}) {c['text'][:36]}")


# =============================================================================
# 第四部分：演示
# =============================================================================

def run_demo():
    """完整演示：三种改写场景 + 改写前后检索对比 + 多子查询合并检索"""
    # ---- 改写模型：默认 Ollama 本地（无需 API Key）----
    # 想走云端可改成 get_model("qwen")，但需 ALIYUN_API_KEY
    llm = get_model()

    # ---- 场景 1：太口语 → 标准表述 ----
    print(f"\n-- 示例 1: 口语化改写（太口语 → 标准表述）")
    q1 = "退货的规矩是啥"
    r1 = rewrite_query(llm, q1)
    print(f"  原始：{q1}")
    print(f"  改写：{r1}")

    # ---- 场景 2：太笼统 → 补全具体 ----
    print(f"\n-- 示例 2: 模糊补全（太笼统 → 补全信息）")
    q2 = "那个钱的事你给说说"
    r2 = rewrite_query(llm, q2)
    print(f"  原始：{q2}")
    print(f"  改写：{r2}")

    # ---- 场景 3：太复杂 → 拆分多子查询 ----
    print(f"\n-- 示例 3: 复杂拆分（太复杂 → 多个子查询）")
    q3 = "退货和换货有什么区别？发票怎么开？"
    sub_queries = decompose_query(llm, q3)
    print(f"  原始：{q3}")
    print(f"  拆分为 {len(sub_queries)} 个子查询：")
    for i, q in enumerate(sub_queries, 1):
        print(f"    {i}. {q}")

    # ---- 检索对比：复用 HybridRetriever（需 Milvus + ALIYUN_API_KEY）----
    print(f"\n-- 检索对比（复用 HybridRetriever）")
    retriever = setup_retriever()

    print(f"\n  ① 口语化问题，改写前 vs 改写后：")
    print_hits("原始检索：", retriever.retrieve(q1, top_k=2))
    print_hits("改写检索：", retriever.retrieve(r1, top_k=2))

    print(f"\n  ② 笼统问题，改写前 vs 改写后：")
    print_hits("原始检索：", retriever.retrieve(q2, top_k=2))
    print_hits("改写检索：", retriever.retrieve(r2, top_k=2))

    print(f"\n  ③ 复杂问题，多子查询分别检索 + 合并：")
    merged = multi_query_retrieve(retriever, sub_queries, top_k=2)
    print_hits("合并结果：", merged)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  RAG 高级技术 - 查询改写（Query Rewriting）")
    print("=" * 70)

    try:
        run_demo()
    except RuntimeError as e:
        print(f"\n✗ {e}")
    except Exception as e:
        print(f"\n✗ 运行失败：{e}")
        print("  提示：检索部分需要 Milvus 服务运行中 + ALIYUN_API_KEY")

    print("\n" + "=" * 70)
    print("  完成！查询改写让检索更稳定、召回更全面。")
    print("=" * 70)
