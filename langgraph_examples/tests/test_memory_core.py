# =============================================================================
# memory_core 算法层确定性测试（需求 §8）
# =============================================================================
# 覆盖：来源赋分 / 明确度折算（FR-10）/ 衰减曲线 / 四档分界 / 四类关系 /
#       upsert 全分支 / 周摘要合并 / recall 管线 / 渲染 / 伪向量确定性。
# 全部走规则降级（llm=None）与时间注入（now=固定值），不触发任何网络与数据库。
# =============================================================================

from datetime import datetime, timedelta

import memory_core as mc

NOW = datetime(2026, 3, 1)


# ---- 来源 → 初始置信度 / 半衰期（§6.4 来源表）----

def test_initial_confidence_by_source():
    assert mc.initial_confidence("user_correction") == 0.95
    assert mc.initial_confidence("user_stated") == 0.90
    assert mc.initial_confidence("tool_result") == 0.75
    assert mc.initial_confidence("session_summary") == 0.60
    assert mc.initial_confidence("agent_inferred") == 0.50
    assert mc.initial_confidence("imported") == 0.40
    assert mc.initial_confidence("unknown_source") == 0.40  # 未知来源按最低档


def test_initial_confidence_with_explicitness():
    assert mc.initial_confidence("user_stated", "explicit") == 0.90  # ×1.0
    assert mc.initial_confidence("user_stated", "implicit") == 0.72  # ×0.8
    assert mc.initial_confidence("user_stated", "hedged") == 0.54    # ×0.6（验收 9）
    assert mc.initial_confidence("agent_inferred", "hedged") == 0.30  # 落入 pending_verify


def test_halflife_lookup():
    assert mc.halflife_for("user_stated", "long") == 90
    assert mc.halflife_for("session_summary", "mid") == 7  # 中期固定 7 天，覆盖来源表
    assert mc.halflife_for("anything_unknown", "long") == 30


# ---- 有效置信度衰减 / 四档分界 ----

def _entry(**overrides):
    entry = mc.new_entry("u1", "name", "我叫李明", "user_stated", "explicit", now=NOW)
    entry.update(overrides)
    return entry


def test_effective_confidence_decay_curve():
    entry = _entry(last_accessed_at=NOW)
    assert mc.effective_confidence(entry, now=NOW) == 0.90  # 刚触达：无衰减
    after_90d = mc.effective_confidence(entry, now=NOW + timedelta(days=90))
    assert abs(after_90d - 0.475) < 1e-6  # 半衰期后：(0.90-0.05)/2 + 0.05
    after_century = mc.effective_confidence(entry, now=NOW + timedelta(days=36500))
    assert 0.05 <= after_century <= 0.06  # 遗忘可逆：不衰减到 0


def test_effective_confidence_touch_resets_decay():
    entry = _entry(last_accessed_at=NOW)
    aged = mc.effective_confidence(entry, now=NOW + timedelta(days=90))
    entry["last_accessed_at"] = NOW + timedelta(days=90)  # touch 刷新
    refreshed = mc.effective_confidence(entry, now=NOW + timedelta(days=90))
    assert refreshed == 0.90 > aged  # 用进废退：衰减重新起算


def test_classify_tier_boundaries():
    assert mc.classify_tier(0.75) == "trusted"
    assert mc.classify_tier(0.7499) == "active"
    assert mc.classify_tier(0.45) == "active"
    assert mc.classify_tier(0.4499) == "pending_verify"
    assert mc.classify_tier(0.25) == "pending_verify"
    assert mc.classify_tier(0.2499) == "archived"


# ---- 关系判断与明确度（规则降级，llm=None）----

def test_judge_relation_rules():
    assert mc.judge_relation_rules("我叫李明", "我叫李明") == "confirm"
    assert mc.judge_relation_rules("我的目标是考研", "对，我的目标是考研") == "confirm"
    assert mc.judge_relation_rules("我叫李明", "我叫李明，今年18岁") == "supplement"
    assert mc.judge_relation_rules("我的目标是考研", "我的目标是考公务员") == "supplement"
    assert mc.judge_relation_rules("我叫李明", "不对，我叫李强") == "contradict"
    assert mc.judge_relation_rules("我叫李明", "今天天气怎么样") == "unrelated"


def test_judge_explicitness_rules():
    assert mc.judge_explicitness_rules("我叫李明") == "explicit"
    assert mc.judge_explicitness_rules("我可能叫李明吧") == "hedged"
    assert mc.judge_explicitness_rules("平时喜欢打球") == "implicit"  # 默认档


def test_judge_falls_back_without_llm():
    relation, explicitness = mc.judge_relation_and_explicitness(
        "我叫李明", "不对，我叫李强", llm=None)
    assert (relation, explicitness) == ("contradict", "explicit")
    # 新建场景：relation 为 None，仅判明确度
    relation2, explicitness2 = mc.judge_relation_and_explicitness(None, "我叫李明", llm=None)
    assert relation2 is None and explicitness2 == "explicit"


# ---- upsert 全分支（§6.4 关系表）----

def test_resolve_upsert_rejects_unrelated():
    old = mc.new_entry("u1", "name", "我叫李明", "user_stated", now=NOW)
    result = mc.resolve_upsert(old, "今天天气怎么样", "user_stated", "implicit",
                               "unrelated", now=NOW)
    assert result["action"] == "rejected"
    assert result["entry"] is old and result["archived"] is None


def test_resolve_upsert_confirm_and_supplement():
    old = mc.new_entry("u1", "goal", "我的目标是考研", "user_stated", now=NOW,
                       evidence=[mc.evidence_item("t0", "我的目标是考研", "explicit", NOW)])
    confirmed = mc.resolve_upsert(old, "对，我的目标是考研", "user_stated", "explicit",
                                  "confirm", now=NOW)
    assert confirmed["action"] == "confirmed"
    assert confirmed["entry"]["confidence"] == 0.95  # 0.90 + 0.05
    assert confirmed["entry"]["revision"] == 2
    assert len(confirmed["entry"]["evidence"]) == 2

    supplemented = mc.resolve_upsert(old, "我的目标是考研，目标院校是清华", "user_stated",
                                     "explicit", "supplement", now=NOW)
    assert supplemented["action"] == "supplemented"
    assert "清华" in supplemented["entry"]["content"] and "考研" in supplemented["entry"]["content"]
    assert supplemented["entry"]["revision"] == 2


def test_resolve_upsert_contradict_archives():
    old = mc.new_entry("u1", "goal", "我的目标是考研", "user_stated", now=NOW)  # 0.90 / rev 1
    result = mc.resolve_upsert(old, "不对，我的目标是考公", "user_correction", "explicit",
                               "contradict", thread_id="t1", now=NOW)
    assert result["action"] == "contradicted"
    archived = result["archived"]
    assert archived["mem_key"] == "goal__archived_1" and archived["status"] == "archived"
    assert archived["confidence"] == 0.60  # 旧条目 −0.30
    new = result["entry"]
    assert new["confidence"] == 0.95  # user_correction ×1.0（验收 4）
    assert new["revision"] == 2 and new["supersedes"] == ["goal__archived_1"]


def test_archived_key_monotonic():
    assert mc.archived_key("goal", 1) == "goal__archived_1"
    assert mc.archived_key("goal", 2) != mc.archived_key("goal", 1)  # 同键归档后缀永不冲突


def test_single_value_key_override_to_contradict():
    # 单值键取值变化（LLM/rules 误判 supplement 时兜底）：旧值未被保留 → 取代归档
    old = mc.new_entry("u1", "goal", "考研", "user_stated", now=NOW)
    result = mc.resolve_upsert(old, "考公务员", "user_correction", "implicit",
                               "supplement", now=NOW)
    assert result["action"] == "contradicted"
    assert result["archived"]["mem_key"] == "goal__archived_1"
    assert result["entry"]["confidence"] == 0.95  # user_correction 强制 explicit（验收 4）

    # 命名空间键（含 ":"，可多条）保持 supplement 合并
    old_ns = mc.new_entry("u1", "weak_point:python", "闭包", "user_stated", now=NOW)
    merged = mc.resolve_upsert(old_ns, "闭包和装饰器", "user_stated", "implicit",
                               "supplement", now=NOW)
    assert merged["action"] == "supplemented"

    # 单值键 affirm 但取值变化：confirm 兜底改判 contradict
    old_name = mc.new_entry("u1", "name", "李明", "user_stated", now=NOW)
    affirmed = mc.resolve_upsert(old_name, "对，我叫李强", "user_stated", "explicit",
                                 "confirm", now=NOW)
    assert affirmed["action"] == "contradicted"


def test_normalize_explicitness():
    assert mc.normalize_explicitness("user_correction", "implicit") == "explicit"
    assert mc.normalize_explicitness("user_stated", "hedged") == "hedged"  # 仅纠正强制
    assert mc.normalize_explicitness("user_stated", None) == "implicit"


# ---- 中期周摘要（FR-5 / 验收 5）----

def test_merge_weekly_dedup_lines():
    merged = mc.merge_weekly("第1行：函数\n第2行：循环", "第2行：循环\n第3行：推导式")
    assert merged.splitlines() == ["第1行：函数", "第2行：循环", "第3行：推导式"]


def test_resolve_weekly_created_and_merged():
    created = mc.resolve_weekly("u1", "2026-W45", "第1行：学习函数", now=NOW)
    assert created["action"] == "created"
    assert created["entry"]["confidence"] == 0.60
    assert created["entry"]["revision"] == 1
    assert created["entry"]["halflife_days"] == 7
    assert created["entry"]["memory_type"] == "mid"

    merged = mc.resolve_weekly("u1", "2026-W45", "第2行：完成练习",
                               old=created["entry"], now=NOW)
    assert merged["action"] == "merged"
    assert merged["entry"]["confidence"] == 0.65  # 0.60 + 0.05（验收 5：三次 → 0.70 / rev 3）
    assert merged["entry"]["revision"] == 2


def test_week_key_format():
    d = datetime(2026, 11, 12)
    iso = d.isocalendar()
    assert mc.week_key(d) == f"{iso.year}-W{iso.week:02d}"


# ---- recall 管线（§6.4 / FR-7）----

def test_recall_pipeline_filter_and_rank():
    def cand(key, content, conf, sim, days_since_access):
        entry = mc.new_entry("u1", key, content, "user_stated", now=NOW)
        entry["confidence"] = conf
        entry["last_accessed_at"] = NOW - timedelta(days=days_since_access)
        return {**entry, "sim": sim}

    candidates = [
        cand("a", "条件一", 0.90, 0.9, 0),    # 高 sim + 高 eff → 第一
        cand("b", "条件二", 0.90, 0.5, 0),
        cand("c", "条件三", 0.10, 0.9, 0),    # eff < 注入下限 0.25 → 过滤
        cand("d", "条件四", 0.90, 0.0, 100),  # 无相似度且久未触达 → 靠后
    ]
    hits = mc.recall_pipeline(candidates, "任意查询", now=NOW, final_top_k=2)
    assert [h["mem_key"] for h in hits] == ["a", "b"]
    assert all(h["eff"] >= mc.INJECT_FLOOR for h in hits)
    assert hits[0]["final"] >= hits[1]["final"]


def test_keyword_score():
    assert mc.keyword_score("我叫什么名字", "我叫李明") > 0
    assert mc.keyword_score("量子力学", "我叫李明") == 0


# ---- 摘要与渲染 ----

def test_summarize_week_rule_fallback():
    messages = [
        {"role": "user", "content": "我在复习函数"},
        {"role": "assistant", "content": "好的，我们来看函数的定义"},
        {"role": "user", "content": "闭包还不熟"},
    ]
    summary = mc.summarize_week(messages, llm=None)
    assert "函数" in summary and "闭包" in summary and len(summary) <= 200


def test_render_memory_block():
    long_entry = {**mc.new_entry("u1", "name", "李明", "user_stated", now=NOW),
                  "eff": 0.9, "tier": "trusted"}
    block = mc.render_memory_block(
        [long_entry], [{"mem_key": "2026-W45", "content": "学习了函数"}], now=NOW)
    assert "李明" in block and "0.9" in block and "可直接使用" in block
    assert "2026-W45" in block and "update_memory" in block


# ---- 伪向量确定性（FR-9 降级）----

def test_pseudo_vector_deterministic():
    from milvus_index import pseudo_vector
    v1 = pseudo_vector("我叫李明")
    v2 = pseudo_vector("我叫李明")
    v3 = pseudo_vector("今天天气不错")
    assert v1 == v2
    assert len(v1) == 1024
    assert v1 != v3
