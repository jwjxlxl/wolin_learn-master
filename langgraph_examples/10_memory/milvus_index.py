# =============================================================================
# 10_memory — Milvus 瘦索引（派生副本，可随时从 MySQL 全量重建，需求 D6 / §6.2）
# =============================================================================
# 集合 mem_vectors 只存 pk/user_id/mem_key/content/vector：权威字段一律回表取，
# 重排所需字段永远从 MySQL 读最新值；content 仅为 embed 输入快照（供比对去重与调试展示）。
# 去重规则（FR-6）：比对 mem_entry.content（截断 8192）与 Milvus 行 content，不一致才调 embedding。
# 降级（FR-9）：未配置 ALIYUN_API_KEY 或 USE_REAL_EMBEDDING=false → hash 种子伪向量
# （同文本同向量、可复现；能跑通 schema/过滤，但相似度无语义，打印警示）。
# =============================================================================

import hashlib
import random
import socket
import time
from urllib.parse import urlparse

from db import load_config

COLLECTION = "mem_vectors"
DIMENSION = 1024                       # 与全项目维度约定一致（text-embedding-v4）
EMBED_MODEL = "text-embedding-v4"
EMBED_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MAX_CONTENT_CHARS = 8192
OP_TIMEOUT = 5                         # Milvus 操作超时秒数：不可达时快速失败 → 调用方降级（FR-9）


def pseudo_vector(text: str, dim: int = DIMENSION) -> list:
    """hash 种子伪向量：同文本同向量、可复现；相似度无语义（FR-9 降级）。"""
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    vec = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
    norm = sum(x * x for x in vec) ** 0.5 or 1.0
    return [round(x / norm, 6) for x in vec]


class MilvusIndex:
    """mem_vectors 瘦索引封装：连接/embed/增删查/同步/重建。"""

    def __init__(self, config: dict | None = None, collection: str | None = None):
        self._cfg = config or load_config()
        # 集合名可覆盖：测试用独立集合，避免 rebuild_from_entries 的 drop 清空演示数据
        self._collection = collection or COLLECTION
        self._client = None
        self._openai = None
        self._down_until = 0.0   # TCP 预检失败后的负缓存截止时间（秒级时间戳）

    # ---- 连接 ----

    def _tcp_reachable(self) -> bool:
        """TCP 层预检（2 秒）：gRPC 对 SYN 丢包端口会长时间重试，必须在构造客户端前拦截。

        失败后进入 30 秒负缓存，期间直接判定不可达，避免每个请求都支付预检成本。
        """
        if time.time() < self._down_until:
            return False
        parsed = urlparse(self._cfg["milvus"]["uri"])
        sock = socket.socket()
        sock.settimeout(2)
        try:
            sock.connect((parsed.hostname, parsed.port or 19530))
            return True
        except Exception:
            self._down_until = time.time() + 30
            return False
        finally:
            sock.close()

    def _get_client(self):
        if self._client is None:
            if not self._tcp_reachable():
                raise ConnectionError(f"Milvus 不可达：{self._cfg['milvus']['uri']}")
            from pymilvus import MilvusClient
            kwargs = {"uri": self._cfg["milvus"]["uri"],
                      "db_name": self._cfg["milvus"]["db_name"] or "default"}
            if self._cfg["milvus"]["user"]:
                kwargs["user"] = self._cfg["milvus"]["user"]
                kwargs["password"] = self._cfg["milvus"]["password"]
            self._client = MilvusClient(**kwargs)
        return self._client

    def check_connection(self, quiet: bool = False) -> bool:
        try:
            self._get_client().get_server_version(timeout=OP_TIMEOUT)
            return True
        except Exception as e:
            if not quiet:
                print(f"[降级] Milvus 连接失败，长期召回将退化关键词打分（对话不受影响）：{e}")
            return False

    def ensure_collection(self):
        """建集合（幂等）：瘦索引 schema 见需求 §6.2；pk 长度 = 64+1+128 = 193。"""
        client = self._get_client()
        if client.has_collection(self._collection, timeout=OP_TIMEOUT):
            return
        from pymilvus import CollectionSchema, DataType, FieldSchema
        fields = [
            FieldSchema("pk", DataType.VARCHAR, max_length=193, is_primary=True),
            FieldSchema("user_id", DataType.VARCHAR, max_length=64),
            FieldSchema("mem_key", DataType.VARCHAR, max_length=128),
            FieldSchema("content", DataType.VARCHAR, max_length=MAX_CONTENT_CHARS),
            FieldSchema("vector", DataType.FLOAT_VECTOR, dim=DIMENSION),
        ]
        schema = CollectionSchema(fields, description="长期记忆瘦索引（派生，可重建）")
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        client.create_collection(self._collection, schema=schema, index_params=index_params,
                                 timeout=OP_TIMEOUT)
        try:
            client.load_collection(self._collection, timeout=OP_TIMEOUT)
        except Exception:
            pass

    # ---- embedding（真实 / 伪向量降级）----

    def _use_real(self) -> bool:
        return bool(self._cfg.get("aliyun_api_key")) and bool(self._cfg.get("use_real_embedding"))

    def embed(self, text: str, warn: bool = True) -> list:
        text = (text or "")[:MAX_CONTENT_CHARS]
        if not self._use_real():
            if warn:
                print("[降级] 未配置 ALIYUN_API_KEY 或 USE_REAL_EMBEDDING=false："
                      "使用伪向量（相似度无语义，仅跑通 schema/过滤）")
            return pseudo_vector(text)
        if self._openai is None:
            from openai import OpenAI
            self._openai = OpenAI(api_key=self._cfg["aliyun_api_key"], base_url=EMBED_BASE_URL)
        resp = self._openai.embeddings.create(model=EMBED_MODEL, input=text, dimensions=DIMENSION)
        return resp.data[0].embedding

    # ---- CRUD（pk = "{user_id}:{mem_key}"，天然 upsert）----

    @staticmethod
    def _pk(user_id: str, mem_key: str) -> str:
        return f"{user_id}:{mem_key}"

    @staticmethod
    def _esc(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def _fetch_content(self, user_id: str, mem_key: str):
        # 与写入同用 Strong：紧接写入的去重比对必须读到刚写的快照，否则重复 embed
        rows = self._get_client().query(
            self._collection,
            filter=f'pk == "{self._esc(self._pk(user_id, mem_key))}"',
            output_fields=["content"],
            consistency_level="Strong",
            timeout=OP_TIMEOUT,
        )
        return rows[0]["content"] if rows else None

    def upsert_entry(self, user_id: str, mem_key: str, content: str, warn: bool = True) -> bool:
        """内容快照比对去重（FR-6）：content 未变则不重复 embed。返回是否实际写入。"""
        self.ensure_collection()
        content = (content or "")[:MAX_CONTENT_CHARS]
        if self._fetch_content(user_id, mem_key) == content:
            return False
        vector = self.embed(content, warn=warn)
        self._get_client().upsert(self._collection, data=[{
            "pk": self._pk(user_id, mem_key),
            "user_id": user_id,
            "mem_key": mem_key,
            "content": content,
            "vector": vector,
        }], timeout=OP_TIMEOUT)
        return True

    def delete_entry(self, user_id: str, mem_key: str) -> bool:
        self.ensure_collection()
        self._get_client().delete(
            self._collection, filter=f'pk == "{self._esc(self._pk(user_id, mem_key))}"',
            consistency_level="Strong",
            timeout=OP_TIMEOUT)
        return True

    def search_candidates(self, user_id: str, query: str, limit: int) -> list:
        """向量粗召回：返回 [{mem_key, sim}]。连接/集合任何异常向上抛（调用方降级 FR-9）。

        显式 Strong 一致性：默认 Bounded 为「写后约 3 秒内可见」，上一轮刚同步的条目
        在本轮召回时可能查不到（query 已能读到、search 返回空），跨轮次认人会间歇失效。
        """
        self.ensure_collection()
        vector = self.embed(query, warn=False)
        res = self._get_client().search(
            self._collection, data=[vector],
            filter=f'user_id == "{self._esc(user_id)}"',
            limit=limit, output_fields=["pk"],
            consistency_level="Strong",
            timeout=OP_TIMEOUT,
        )
        out = []
        for hit in (res[0] if res else []):
            # pymilvus 2.6 返回主键在 "pk" 键（旧版文档示例写的是 "id"），两者都取
            pk = hit.get("pk") or hit.get("id")
            if isinstance(pk, str) and ":" in pk:
                out.append({"mem_key": pk.split(":", 1)[1],
                            "sim": float(hit.get("distance", 0.0))})
        return out

    # ---- 同步与重建（FR-6）----

    def sync_entry(self, entry: dict) -> bool:
        """按 MySQL 写路径事件同步单条长期条目（调用时机：响应流结束后，M3 前请求内同步）。

        mid 条目不同步（D4）；touch 不进入本函数（权威字段回表制保证，FR-6）。
        """
        if entry.get("memory_type") != "long":
            return False
        return self.upsert_entry(entry["user_id"], entry["mem_key"], entry.get("content") or "")

    def rebuild_from_entries(self, entries: list) -> int:
        """从 MySQL 全量扫表重建索引（派生可重建的直接证明；降级恢复演示用，验收 §10 第 7 条）。"""
        client = self._get_client()
        if client.has_collection(self._collection, timeout=OP_TIMEOUT):
            client.drop_collection(self._collection, timeout=OP_TIMEOUT)
        count = 0
        if entries:
            print(f"[rebuild] 使用伪/真实 embedding 重建 {len(entries)} 条长期记忆索引...")
            for entry in entries:
                if self.upsert_entry(entry["user_id"], entry["mem_key"],
                                     entry.get("content") or "", warn=False):
                    count += 1
        print(f"[rebuild] 完成，写入 {count} 条")
        return count
