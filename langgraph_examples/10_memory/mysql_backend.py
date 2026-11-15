# =============================================================================
# 10_memory — 存储后端：MySQLBackend + HybridBackend（MemoryBackend 四方法）
# =============================================================================
# - MySQLBackend：put 内以事务 + SELECT ... FOR UPDATE 包裹 get→judge→put
#   （需求 §4 读-改-写并发控制，不引入分布式锁）
# - HybridBackend：检索两段式（D6）——Milvus 瘦索引出候选 → MySQL 回表出权威；
#   Milvus 不可达退化关键词打分（FR-7/FR-9），对话功能不受影响
# - 检索命中条目 touch 写回（hit_count / last_accessed_at），touch 不触发向量同步
# =============================================================================

import json

import memory_core as mc
from db import borrow


_ENTRY_COLS = ("user_id, mem_key, memory_type, status, content, source, confidence, "
               "halflife_days, hit_count, evidence, supersedes, revision, "
               "created_at, updated_at, last_accessed_at")

_INSERT_SQL = (
    f"INSERT INTO mem_entry ({_ENTRY_COLS}) "
    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
)
_UPSERT_SQL = _INSERT_SQL + (
    " ON DUPLICATE KEY UPDATE memory_type=VALUES(memory_type), status=VALUES(status),"
    " content=VALUES(content), source=VALUES(source), confidence=VALUES(confidence),"
    " halflife_days=VALUES(halflife_days), hit_count=VALUES(hit_count),"
    " evidence=VALUES(evidence), supersedes=VALUES(supersedes), revision=VALUES(revision),"
    " updated_at=VALUES(updated_at), last_accessed_at=VALUES(last_accessed_at)"
)


def _row_to_entry(row) -> dict:
    """MySQL 行 → 内存条目（§6.4 条目 schema）；JSON 字段反序列化，DECIMAL 转 float。"""
    entry = dict(row)
    entry["confidence"] = float(entry["confidence"])
    for field in ("evidence", "supersedes"):
        value = entry.get(field)
        entry[field] = json.loads(value) if isinstance(value, str) else (value or [])
    return entry


def _entry_values(entry) -> tuple:
    return (
        entry["user_id"], entry["mem_key"], entry["memory_type"], entry["status"],
        entry["content"], entry["source"], entry["confidence"], entry["halflife_days"],
        entry.get("hit_count", 0),
        json.dumps(entry.get("evidence") or [], ensure_ascii=False),
        json.dumps(entry.get("supersedes") or [], ensure_ascii=False),
        entry["revision"], entry["created_at"], entry["updated_at"], entry["last_accessed_at"],
    )


class MemoryBackend:
    """记忆读写接口（四方法 put/get/search/delete + touch/周键/列表辅助）。

    算法层与存储层解耦的接缝（D8）：图节点与记忆工具只依赖本接口。
    """

    def put(self, user_id, mem_key, content, source="user_stated", explicitness=None,
            memory_type="long", thread_id=None, now=None) -> dict:
        raise NotImplementedError

    def get(self, user_id, mem_key, now=None):
        raise NotImplementedError

    def search(self, user_id, query, top_k=mc.FINAL_TOPK, now=None) -> list:
        raise NotImplementedError

    def delete(self, user_id, mem_key) -> bool:
        raise NotImplementedError


class MySQLBackend(MemoryBackend):
    """MySQL 权威存储实现（mem_entry 表；中期与长期合表，memory_type 区分）。"""

    def __init__(self, llm=None):
        self.llm = llm
        # 待向量同步清单（FR-6：M3 前由服务层在响应流结束后消费）
        self.pending_sync: list = []

    # ---- put：事务 + 行锁包裹 get→judge→put（§4）----

    def put(self, user_id, mem_key, content, source="user_stated", explicitness=None,
            memory_type="long", thread_id=None, now=None) -> dict:
        """写入/更新长期条目。返回 {action, relation, entry, archived} 供 memory_log 使用。

        action: created | confirmed | supplemented | contradicted | rejected（unrelated 拒写）。
        """
        with borrow() as conn:
            conn.begin()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT {_ENTRY_COLS} FROM mem_entry "
                        "WHERE user_id=%s AND mem_key=%s FOR UPDATE",
                        (user_id, mem_key),
                    )
                    row = cur.fetchone()
                old = _row_to_entry(row) if row else None
                if old is None:
                    if explicitness is None:
                        explicitness = mc.judge_explicitness(content, llm=self.llm)
                    explicitness = mc.normalize_explicitness(source, explicitness)
                    entry = mc.new_entry(user_id, mem_key, content, source, explicitness,
                                         memory_type=memory_type, now=now,
                                         evidence=[mc.evidence_item(thread_id, content, explicitness, now)])
                    result = {"action": "created", "relation": None, "entry": entry, "archived": None}
                    with conn.cursor() as cur:
                        cur.execute(_INSERT_SQL, _entry_values(entry))
                else:
                    relation, judged_exp = mc.judge_relation_and_explicitness(
                        old["content"], content, llm=self.llm, mem_key=mem_key)
                    if explicitness is None:
                        explicitness = judged_exp
                    result = mc.resolve_upsert(old, content, source, explicitness,
                                               relation, thread_id=thread_id, now=now)
                    if result["action"] == "rejected":
                        conn.rollback()  # 拒写：不产生任何变更（防记忆污染）
                        return result
                    with conn.cursor() as cur:
                        if result["archived"] is not None:
                            cur.execute(_INSERT_SQL, _entry_values(result["archived"]))
                        cur.execute(_UPSERT_SQL, _entry_values(result["entry"]))
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        self._mark_sync(result["entry"], action="upsert")
        return result

    # ---- 周摘要写回（FR-5；mid 条目不进 Milvus，D4）----

    def put_weekly(self, user_id, week_key, summary, now=None) -> dict:
        """读出旧摘要 → merge_weekly 按行去重合并 → 写回同一周键。返回 {action, entry}。"""
        with borrow() as conn:
            conn.begin()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT {_ENTRY_COLS} FROM mem_entry "
                        "WHERE user_id=%s AND mem_key=%s FOR UPDATE",
                        (user_id, week_key),
                    )
                    row = cur.fetchone()
                old = _row_to_entry(row) if row else None
                result = mc.resolve_weekly(user_id, week_key, summary, old=old, now=now)
                with conn.cursor() as cur:
                    if result["action"] == "created":
                        cur.execute(_INSERT_SQL, _entry_values(result["entry"]))
                    else:
                        cur.execute(_UPSERT_SQL, _entry_values(result["entry"]))
                conn.commit()
                return result
            except Exception:
                conn.rollback()
                raise

    # ---- 读取 ----

    def get(self, user_id, mem_key, now=None):
        with borrow() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT {_ENTRY_COLS} FROM mem_entry WHERE user_id=%s AND mem_key=%s",
                (user_id, mem_key),
            )
            row = cur.fetchone()
        return _row_to_entry(row) if row else None

    def search(self, user_id, query, top_k=mc.FINAL_TOPK, now=None) -> list:
        """关键词召回（MySQL 全量扫，Milvus 降级路径）：重合度打分 → memory_core 管线 → touch。"""
        with borrow() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT {_ENTRY_COLS} FROM mem_entry "
                "WHERE user_id=%s AND memory_type='long' AND status='active'",
                (user_id,),
            )
            rows = cur.fetchall()
        candidates = [{**_row_to_entry(r), "sim": mc.keyword_score(query, r["content"])}
                      for r in rows]
        hits = mc.recall_pipeline(candidates, query, now=now, final_top_k=top_k)
        for hit in hits:
            hit["recall"] = "keyword"
            self.touch(user_id, hit["mem_key"], now=now)
        return hits

    def get_weekly(self, user_id, week_keys) -> list:
        """按周键主键直达读取中期摘要（D4：中期无向量检索需求），按传入顺序返回。"""
        keys = [k for k in (week_keys or []) if k]
        if not keys:
            return []
        placeholders = ",".join(["%s"] * len(keys))
        with borrow() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT {_ENTRY_COLS} FROM mem_entry "
                f"WHERE user_id=%s AND memory_type='mid' AND status='active' AND mem_key IN ({placeholders})",
                (user_id, *keys),
            )
            rows = cur.fetchall()
        by_key = {_row_to_entry(r)["mem_key"]: _row_to_entry(r) for r in rows}
        return [by_key[k] for k in keys if k in by_key]

    def list_memories(self, user_id, include_archived=False, now=None) -> list:
        """FR-8 记忆检查数据源：全部字段 + 读时派生 eff/tier；归档行标注原 key。"""
        sql = f"SELECT {_ENTRY_COLS} FROM mem_entry WHERE user_id=%s"
        if not include_archived:
            sql += " AND status='active'"
        sql += " ORDER BY memory_type DESC, mem_key ASC"
        with borrow() as conn, conn.cursor() as cur:
            cur.execute(sql, (user_id,))
            rows = cur.fetchall()
        out = []
        for r in rows:
            entry = _row_to_entry(r)
            entry["eff"] = mc.effective_confidence(entry, now=now)
            entry["tier"] = mc.classify_tier(entry["eff"])
            if "__archived_" in entry["mem_key"]:
                entry["original_key"] = entry["mem_key"].split("__archived_", 1)[0]
            out.append(entry)
        return out

    def list_all_long_entries(self) -> list:
        """全量长期条目（限长期 + active），仅供 rebuild 从 MySQL 重建 Milvus 索引使用。"""
        with borrow() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT {_ENTRY_COLS} FROM mem_entry "
                "WHERE memory_type='long' AND status='active' ORDER BY user_id, mem_key")
            rows = cur.fetchall()
        return [_row_to_entry(r) for r in rows]

    # ---- touch 写回（用进废退）与删除 ----

    def touch(self, user_id, mem_key, now=None):
        """命中刷新 hit_count / last_accessed_at，衰减重新起算（§6.4）。不同步向量（FR-6）。"""
        with borrow() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE mem_entry SET hit_count=hit_count+1, last_accessed_at=%s "
                "WHERE user_id=%s AND mem_key=%s",
                (mc.now_or(now), user_id, mem_key),
            )
            conn.commit()

    def delete(self, user_id, mem_key) -> bool:
        """物理删除（forget_memory）。Milvus 派生行由服务层按 pending_sync 清理。"""
        with borrow() as conn, conn.cursor() as cur:
            affected = cur.execute(
                "DELETE FROM mem_entry WHERE user_id=%s AND mem_key=%s",
                (user_id, mem_key),
            )
            conn.commit()
        if affected:
            self.pending_sync.append({"user_id": user_id, "mem_key": mem_key, "action": "delete"})
        return bool(affected)

    # ---- 内部 ----

    def _mark_sync(self, entry, action: str):
        if entry.get("memory_type") == "long":
            self.pending_sync.append(
                {"user_id": entry["user_id"], "mem_key": entry["mem_key"], "action": action})


class HybridBackend(MySQLBackend):
    """检索两段式（D6）：Milvus 瘦索引出候选（topK 候选 key）→ MySQL 主键批量回表出权威。

    Milvus 不可达时退化为 MySQL 关键词打分（FR-9），对话功能不受影响。
    """

    def __init__(self, llm=None, milvus=None):
        super().__init__(llm=llm)
        from milvus_index import MilvusIndex  # 延迟导入：无 pymilvus 环境仍可用 MySQL 路径
        self.milvus = milvus if milvus is not None else MilvusIndex()

    def search(self, user_id, query, top_k=mc.FINAL_TOPK, now=None) -> list:
        try:
            candidates = self.milvus.search_candidates(user_id, query, mc.CANDIDATE_TOPK)
        except Exception:
            return super().search(user_id, query, top_k=top_k, now=now)  # 降级 FR-9
        entries = self._batch_get(user_id, [c["mem_key"] for c in candidates])
        by_key = {e["mem_key"]: e for e in entries}
        merged = []
        for cand in candidates:
            entry = by_key.get(cand["mem_key"])
            if entry is None:
                continue  # 索引行存在但权威行已删除：派生副本陈旧，跳过（rebuild 可修复）
            merged.append({**entry, "sim": cand["sim"]})
        hits = mc.recall_pipeline(merged, query, now=now, final_top_k=top_k)
        for hit in hits:
            hit["recall"] = "vector"
            self.touch(user_id, hit["mem_key"], now=now)
        return hits

    def _batch_get(self, user_id, mem_keys) -> list:
        if not mem_keys:
            return []
        placeholders = ",".join(["%s"] * len(mem_keys))
        with borrow() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT {_ENTRY_COLS} FROM mem_entry "
                f"WHERE user_id=%s AND mem_key IN ({placeholders})",
                (user_id, *mem_keys),
            )
            rows = cur.fetchall()
        return [_row_to_entry(r) for r in rows]
