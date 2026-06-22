"""诊断 BM25 混合检索返回空的原因 — 分步测试"""
import os, sys, random
from dotenv import load_dotenv
load_dotenv()

from rag_examples.milvus_config import MILVUS_URI, DEFAULT_DIMENSION
from pymilvus import MilvusClient, AnnSearchRequest, Function, FunctionType

print("=" * 70)
print("  BM25 混合检索诊断")
print("=" * 70)

# ── 连接 ──
client = MilvusClient(uri=MILVUS_URI)
try:
    ver = client.get_server_version()
    print(f"✓ Milvus 版本: {ver}")
    # 检查是否 >= 2.4（BM25 Function 需要）
    parts = ver.lstrip('v').split('.')
    major, minor = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    if major < 2 or (major == 2 and minor < 4):
        print(f"\n❌ BM25 Function 需要 Milvus 2.4+，当前版本 {ver}")
        sys.exit(1)
except Exception as e:
    print(f"✗ 无法获取服务器版本: {e}")
    sys.exit(1)

COLL = "advanced_documents"

# ── 检查集合 ──
if not client.has_collection(COLL):
    print(f"\n❌ 集合 '{COLL}' 不存在，请先运行主脚本创建")
    sys.exit(1)

desc = client.describe_collection(COLL)
print(f"\n集合 '{COLL}' schema:")
for f in desc["fields"]:
    extra = ""
    if f.get("enable_analyzer"):
        extra = " [analyzer=True]"
    if f.get("dim"):
        extra = f" [dim={f['dim']}]"
    print(f"  - {f['name']}: {f['type']}{extra}")
print(f"函数: {desc.get('functions', '无')}")

# ── 检查数据 ──
records = client.query(COLL, filter="id > 0", limit=100, output_fields=["id", "text"])
print(f"\n总记录数: {len(records)}")
for r in records:
    print(f"  id={r['id']}: {r['text'][:60]}...")

if not records:
    print("\n❌ 集合中没有数据！")
    sys.exit(1)

query = "Milvus 支持什么检索方式？"

# ── 测试 1: 纯稠密向量 ──
print(f"\n{'─'*60}")
print(f"测试 1: 纯稠密向量检索")
random.seed(hash(query) % 10000)
qv = [random.uniform(-1, 1) for _ in range(DEFAULT_DIMENSION)]
results = client.search(
    collection_name=COLL, data=[qv], anns_field="dense_vector",
    param={"nprobe": 10}, limit=3, output_fields=["text"],
)
print(f"  返回 {len(results[0])} 条结果")
for hit in results[0]:
    print(f"    [{hit['distance']:.4f}] {hit['entity']['text'][:50]}...")

# ── 测试 2: 纯 BM25 ──
print(f"\n{'─'*60}")
print(f"测试 2: 纯 BM25 检索 (client.search)")
results = client.search(
    collection_name=COLL, data=[query], anns_field="sparse_vector",
    param={"metric_type": "BM25"}, limit=5, output_fields=["text"],
)
print(f"  返回 {len(results[0])} 条结果")
for hit in results[0]:
    print(f"    [{hit['distance']:.4f}] {hit['entity']['text'][:50]}...")

# ── 测试 3: 单关键词 BM25 ──
print(f"\n{'─'*60}")
print(f"测试 3: 单关键词 BM25 检索")
for q in ["Milvus", "向量", "BM25", "RAG", "TF-IDF"]:
    r = client.search(
        collection_name=COLL, data=[q], anns_field="sparse_vector",
        param={"metric_type": "BM25"}, limit=3, output_fields=["text"],
    )
    print(f"  query='{q}': {len(r[0])} 条结果")
    for hit in r[0]:
        print(f"    [{hit['distance']:.4f}] {hit['entity']['text'][:40]}...")

# ── 测试 4: hybrid_search 单路 BM25 ──
print(f"\n{'─'*60}")
print(f"测试 4: hybrid_search 单路（只有 BM25）")
req_sparse = AnnSearchRequest(
    data=[query], anns_field="sparse_vector",
    param={"metric_type": "BM25"}, limit=3,
)
ranker = Function(
    name="rrf", input_field_names=[],
    function_type=FunctionType.RERANK,
    params={"reranker": "rrf", "k": 100},
)
results = client.hybrid_search(
    collection_name=COLL, reqs=[req_sparse], ranker=ranker,
    limit=3, output_fields=["text"],
)
print(f"  返回 {len(results)} 组")
for i, hits in enumerate(results):
    print(f"    组 {i}: {len(hits)} 条结果")
    for hit in hits:
        print(f"      [{hit['distance']:.4f}] {hit['entity']['text'][:50]}...")

# ── 测试 5: hybrid_search 双路 ──
print(f"\n{'─'*60}")
print(f"测试 5: hybrid_search 双路（稠密 + BM25）")
req_dense = AnnSearchRequest(
    data=[qv], anns_field="dense_vector",
    param={"nprobe": 10}, limit=3,
)
req_sparse = AnnSearchRequest(
    data=[query], anns_field="sparse_vector",
    param={"metric_type": "BM25"}, limit=3,
)
ranker = Function(
    name="rrf", input_field_names=[],
    function_type=FunctionType.RERANK,
    params={"reranker": "rrf", "k": 100},
)
results = client.hybrid_search(
    collection_name=COLL, reqs=[req_dense, req_sparse], ranker=ranker,
    limit=3, output_fields=["text"],
)
print(f"  返回 {len(results)} 组")
for i, hits in enumerate(results):
    print(f"    组 {i}: {len(hits)} 条结果")
    for hit in hits:
        print(f"      [{hit['distance']:.4f}] {hit['entity']['text'][:50]}...")

# ── QA 集合 ──
QA_COLL = "advanced_qa_pairs"
print(f"\n{'─'*60}")
print(f"测试 QA 集合 '{QA_COLL}'")
if client.has_collection(QA_COLL):
    qa_records = client.query(QA_COLL, filter="qa_id > 0", limit=100, output_fields=["qa_id", "question"])
    print(f"  总记录数: {len(qa_records)}")
    for r in qa_records[:3]:
        print(f"    qa_id={r['qa_id']}: {r['question']}")

    results = client.search(
        collection_name=QA_COLL, data=[query], anns_field="sparse_vector",
        param={"metric_type": "BM25"}, limit=5, output_fields=["question"],
    )
    print(f"  BM25 检索返回 {len(results[0])} 条结果")
    for hit in results[0]:
        print(f"    [{hit['distance']:.4f}] {hit['entity']['question'][:50]}...")
else:
    print(f"  集合 '{QA_COLL}' 不存在")

print(f"\n{'='*60}")
print("诊断完成")
print(f"{'='*60}")
