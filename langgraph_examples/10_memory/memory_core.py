# =============================================================================
# 10_memory — 记忆算法层（memory_core）：纯函数，零存储依赖
# =============================================================================
# 规格来源：mysql_memory_dialogue_requirements.md §6.4 / FR-10。
# 确定性约束（§6.4 硬要求）：
#   - 所有时间函数接受 now=None 注入；
#   - 所有 LLM 函数接受 llm=None 并带规则降级；
#   - 本模块不做任何 print（教学日志由图节点负责）。
# =============================================================================

import json
import re
from datetime import datetime

import jieba

# ---- 置信度体系常量（§6.4）----
FLOOR = 0.05          # 遗忘下限：不衰减到 0，归档记忆被再次提到可爬升
CAP = 0.95            # 上限：永不给 1.0
INJECT_FLOOR = 0.25   # 注入下限（低于则不注入不返回）

# 来源 → (初始基准分, 半衰期天数)。原则：亲口说的 > 客观数据 > 二手概括 > 机器推断 > 外部导入
SOURCE_TABLE = {
    "user_correction": (0.95, 180),
    "user_stated":     (0.90, 90),
    "tool_result":     (0.75, 60),
    "session_summary": (0.60, 30),
    "agent_inferred":  (0.50, 14),
    "imported":        (0.40, 30),
}
DEFAULT_SOURCE = (0.40, 30)   # 未知来源按最低档
MID_HALFLIFE_DAYS = 7         # 中期（mid）半衰期固定 7 天，覆盖来源表半衰期列

# 明确度系数（FR-10）：初始分 = 来源基准分 × 明确度系数
EXPLICITNESS_FACTOR = {"explicit": 1.0, "implicit": 0.8, "hedged": 0.6}
DEFAULT_EXPLICITNESS = "implicit"

# 四档状态阈值（由有效置信度 eff 分档，决定注入措辞）
TIER_TRUSTED, TIER_ACTIVE, TIER_PENDING = 0.75, 0.45, 0.25

# 召回管线常量（§6.4 / FR-7）
CANDIDATE_TOPK = 50
FINAL_TOPK = 3
W_SIM, W_EFF, W_RECENCY = 0.6, 0.3, 0.1      # final = 0.6×sim + 0.3×eff + 0.1×recency
RECENCY_HALFLIFE_DAYS = 30.0                  # recency 因子半衰期（模块自定义口径）

# 关系判断调分（§6.4 关系表）
CONFIRM_DELTA = 0.05        # confirm 加分区间 +0.05~0.10 的确定取值
SUPPLEMENT_DELTA = 0.05
CONTRADICT_PENALTY = 0.30
WEEKLY_MERGE_DELTA = 0.05   # 周摘要每次合并 +0.05（FR-5）

# 规则降级词表 / 句式
_HEDGED_WORDS = ("可能", "大概", "好像", "似乎", "也许", "估计", "算是吧", "差不多", "不一定")
_NEGATION_WORDS = ("不是", "不对", "错了", "纠正", "其实", "已经不", "别记", "搞错了")
_AFFIRM_WORDS = ("对", "是的", "没错", "正确", "嗯")
_EXPLICIT_PATTERNS = ("我叫", "我是", "我的名字", "我的目标", "我想考", "我要考",
                      "我在准备", "我今年", "我住", "我的学校", "我的专业", "我的薄弱")

RELATIONS = ("confirm", "contradict", "supplement", "unrelated")


# -----------------------------------------------------------------------------
# 时间与基础构造
# -----------------------------------------------------------------------------

def now_or(now=None) -> datetime:
    """确定性注入点：now=None 取当前时间（§6.4 确定性约束）。"""
    return now or datetime.now()


def week_key(now=None) -> str:
    """ISO 周键（如 2026-W45），中期摘要主键（FR-5）。"""
    iso = now_or(now).isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def halflife_for(source: str, memory_type: str = "long") -> int:
    """半衰期查表：mid 固定 7 天（覆盖来源表）；未知来源按最低档。"""
    if memory_type == "mid":
        return MID_HALFLIFE_DAYS
    return SOURCE_TABLE.get(source, DEFAULT_SOURCE)[1]


def initial_confidence(source: str, explicitness: str | None = None) -> float:
    """初始分 = 来源基准分 × 明确度系数（FR-10），收敛于 CAP。

    explicitness=None（未判定 / 机器生成来源，如 session_summary）时不打折取完整基准分。
    """
    base = SOURCE_TABLE.get(source, DEFAULT_SOURCE)[0]
    factor = EXPLICITNESS_FACTOR.get(explicitness, 1.0)
    return round(min(base * factor, CAP), 2)


def evidence_item(thread_id, quote, kind=None, now=None) -> dict:
    """一条可溯源证据 {thread_id, quote, at, kind}（§6.4 条目 schema；kind 记录明确度取值）。"""
    return {
        "thread_id": thread_id,
        "quote": (quote or "")[:200],
        "at": now_or(now).strftime("%Y-%m-%d %H:%M:%S"),
        "kind": kind or DEFAULT_EXPLICITNESS,
    }


def new_entry(user_id, mem_key, content, source, explicitness=None,
              memory_type="long", evidence=None, supersedes=None, now=None) -> dict:
    """构造新条目（内存形态，§6.4 条目 schema；身份 = (user_id, mem_key)，无独立 id）。"""
    ts = now_or(now)
    return {
        "user_id": user_id,
        "mem_key": mem_key,
        "memory_type": memory_type,
        "status": "active",
        "content": content,
        "source": source,
        "confidence": initial_confidence(source, explicitness),
        "halflife_days": halflife_for(source, memory_type),
        "hit_count": 0,
        "evidence": list(evidence or []),
        "supersedes": list(supersedes or []),
        "revision": 1,
        "created_at": ts,
        "updated_at": ts,
        "last_accessed_at": ts,
    }


def archived_key(mem_key: str, revision) -> str:
    """归档行主键 = {原key}__archived_{被归档时的revision}，随 revision 单调递增永不冲突。"""
    return f"{mem_key}__archived_{revision}"


# -----------------------------------------------------------------------------
# 有效置信度与四档状态（读时派生，不落盘）
# -----------------------------------------------------------------------------

def effective_confidence(entry: dict, now=None) -> float:
    """eff = FLOOR + (confidence − FLOOR) × 0.5^(age/halflife)。

    age 基于 last_accessed_at：检索命中 touch 刷新后衰减重新起算（用进废退）。
    """
    last = entry.get("last_accessed_at") or now_or(now)
    age_days = max((now_or(now) - last).total_seconds() / 86400.0, 0.0)
    half = max(int(entry.get("halflife_days") or 1), 1)
    conf = float(entry.get("confidence", FLOOR))
    eff = FLOOR + (conf - FLOOR) * (0.5 ** (age_days / half))
    return round(max(eff, FLOOR), 4)


def classify_tier(eff: float) -> str:
    """四档状态：≥0.75 trusted / ≥0.45 active / ≥0.25 pending_verify / <0.25 archived。"""
    if eff >= TIER_TRUSTED:
        return "trusted"
    if eff >= TIER_ACTIVE:
        return "active"
    if eff >= TIER_PENDING:
        return "pending_verify"
    return "archived"


# -----------------------------------------------------------------------------
# 新旧关系判断 + 明确度评分（LLM + 规则降级双路径，同一次 LLM 调用输出两者）
# -----------------------------------------------------------------------------

def _char_bigrams(text: str) -> set:
    t = re.sub(r"\s+", "", text or "")
    if len(t) < 2:
        return {t} if t else set()
    return {t[i:i + 2] for i in range(len(t) - 1)}


def content_overlap(old_content: str, new_content: str) -> tuple:
    """返回 (shared, novelty)：
    shared = 新旧 2-gram 交集 / min(|old|, |new|)；novelty = 新增 2-gram / |new|。
    短内容（如"考研"/"李明"）2-gram 过脆：2-gram 不重合时以单字重合率兜底。"""
    bo, bn = _char_bigrams(old_content), _char_bigrams(new_content)
    if not bo or not bn:
        return 0.0, 1.0
    shared = len(bo & bn) / min(len(bo), len(bn))
    novelty = len(bn - bo) / len(bn)
    if shared < 0.3:
        chars_old, chars_new = set(old_content), set(new_content)
        unigram = len(chars_old & chars_new) / min(len(chars_old), len(chars_new))
        if unigram > shared:
            shared = unigram
            novelty = len(chars_new - chars_old) / len(chars_new)
    return shared, novelty


def _starts_with_affirm(text: str) -> bool:
    t = (text or "").strip()
    return any(t.startswith(w) for w in _AFFIRM_WORDS)


def judge_explicitness_rules(new_content: str) -> str:
    """规则降级：hedged 词表命中 → hedged；明确自述句式/肯定应答 → explicit；否则 implicit（默认档）。"""
    if any(w in new_content for w in _HEDGED_WORDS):
        return "hedged"
    if any(p in new_content for p in _EXPLICIT_PATTERNS):
        return "explicit"
    if _starts_with_affirm(new_content):
        return "explicit"
    return DEFAULT_EXPLICITNESS


def judge_relation_rules(old_content: str, new_content: str) -> str:
    """规则降级判定（§6.4）：否定词+话题重合 → contradict；高度一致/肯定应答 → confirm；
    关键词重合 → supplement；否则 unrelated（拒写，防记忆污染）。"""
    shared, novelty = content_overlap(old_content, new_content)
    if any(w in new_content for w in _NEGATION_WORDS) and shared >= 0.2:
        return "contradict"
    if (shared >= 0.8 and novelty <= 0.2) or (_starts_with_affirm(new_content) and shared >= 0.5):
        return "confirm"
    if shared >= 0.3:
        return "supplement"
    return "unrelated"


_RELATION_PROMPT = (
    '判断新旧两条学生记忆的关系与表述明确度，只输出 JSON：'
    '{"relation": "confirm|contradict|supplement|unrelated", '
    '"explicitness": "explicit|implicit|hedged"}\n'
    'relation 定义：confirm=新观察支持旧记忆；contradict=新观察取代旧记忆——'
    '同一主题下互斥的新取值（如姓名/目标等单值事实的值发生变化）或明确的否定；'
    'supplement=在旧记忆基础上补充不互斥的细节；unrelated=与该记忆的主题完全无关（应拒写）。\n'
    'explicitness 定义：explicit=明确陈述/直接确认；implicit=上下文间接可得；'
    'hedged=含糊/猜测（如"可能/大概/好像"）。\n'
    '旧记忆（键名 {key}，键名代表记忆主题）：{old}\n新观察：{new}'
)

_EXPLICIT_PROMPT = (
    '判断这条学生表述的明确度，只输出 JSON：{"explicitness": "explicit|implicit|hedged"}\n'
    'explicit=明确陈述/直接确认；implicit=上下文间接可得；hedged=含糊/猜测。\n'
    '学生表述：{new}'
)


def _llm_text(resp) -> str:
    content = getattr(resp, "content", resp)
    return content if isinstance(content, str) else str(content)


def _parse_json_field(text: str, key: str, allowed) -> str | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except Exception:
        return None
    value = data.get(key)
    return value if value in allowed else None


def judge_explicitness(new_content: str, llm=None) -> str:
    """明确度判定：LLM 失败/未提供时走规则降级（FR-10）。"""
    if llm is not None:
        try:
            text = _llm_text(llm.invoke(_EXPLICIT_PROMPT.format(new=new_content)))
            value = _parse_json_field(text, "explicitness", EXPLICITNESS_FACTOR)
            if value:
                return value
        except Exception:
            pass
    return judge_explicitness_rules(new_content)


def judge_relation_and_explicitness(old_content, new_content, llm=None, mem_key="") -> tuple:
    """新旧关系 + 明确度，同一次 LLM 调用输出（FR-10，不增加额外 LLM 往返）。

    mem_key 传入判定提示词：裸内容（如"考研"）没有主题，键名是判断 related 的关键上下文。
    old_content 为 None（新建场景）时返回 (None, explicitness)。
    任一环节失败即回落规则降级；LLM 只给出关系时，明确度由规则补齐（反之亦然）。
    """
    if old_content is None:
        return None, judge_explicitness(new_content, llm=llm)
    relation = explicitness = None
    if llm is not None:
        try:
            prompt = _RELATION_PROMPT.format(key=mem_key or "（无）", old=old_content, new=new_content)
            text = _llm_text(llm.invoke(prompt))
            relation = _parse_json_field(text, "relation", RELATIONS)
            explicitness = _parse_json_field(text, "explicitness", EXPLICITNESS_FACTOR)
        except Exception:
            relation = explicitness = None
    return (
        relation or judge_relation_rules(old_content, new_content),
        explicitness or judge_explicitness_rules(new_content),
    )


# -----------------------------------------------------------------------------
# upsert 分支的纯逻辑（MySQLBackend 在事务 + FOR UPDATE 内调用，§4 / §6.4 关系表）
# -----------------------------------------------------------------------------

def _merge_content(old_content: str, new_content: str) -> str:
    """supplement 合并内容：新内容已含于旧内容则不变，否则追加。"""
    if new_content in old_content:
        return old_content
    return f"{old_content}；{new_content}"


def normalize_explicitness(source: str, explicitness) -> str:
    """明确度归一：user_correction 天然是明确表述，强制 explicit（验收 4：新条目 0.95）。"""
    if source == "user_correction" and explicitness != "explicit":
        return "explicit"
    return explicitness or DEFAULT_EXPLICITNESS


def resolve_upsert(old, content, source, explicitness, relation, thread_id=None, now=None) -> dict:
    """对同 key 新旧内容判定后的分支处理（§6.4 关系表）。

    返回 dict：
      action:   created | confirmed | supplemented | contradicted | rejected
      relation: 最终关系（created 为 None）
      entry:    落库后的最新条目（键不变；rejected 时为旧条目）
      archived: contradict 产生的归档行（否则 None）
    """
    ts = now_or(now)
    explicitness = normalize_explicitness(source, explicitness)
    ev = evidence_item(thread_id, content, kind=explicitness, now=ts)
    if relation == "unrelated":
        return {"action": "rejected", "relation": relation, "entry": old, "archived": None}

    # 单值键（无 ":" 命名空间，如 name/goal）取值变化兜底：旧值未被新内容保留，
    # 说明新说法取代旧说法——LLM 判 supplement/confirm 时强制改判 contradict（可审计归档）
    if relation in ("confirm", "supplement") and ":" not in old["mem_key"] \
            and old["content"].strip() not in content:
        relation = "contradict"

    if relation == "confirm":
        entry = dict(old)
        entry["confidence"] = round(min(float(old["confidence"]) + CONFIRM_DELTA, CAP), 2)
        entry["revision"] = int(old["revision"]) + 1
        entry["evidence"] = list(old.get("evidence") or []) + [ev]
        entry["updated_at"] = ts
        return {"action": "confirmed", "relation": relation, "entry": entry, "archived": None}

    if relation == "supplement":
        base = max(float(old["confidence"]), initial_confidence(source, explicitness))
        entry = dict(old)
        entry["content"] = _merge_content(old["content"], content)
        entry["confidence"] = round(min(base + SUPPLEMENT_DELTA, CAP), 2)
        entry["revision"] = int(old["revision"]) + 1
        entry["evidence"] = list(old.get("evidence") or []) + [ev]
        entry["updated_at"] = ts
        return {"action": "supplemented", "relation": relation, "entry": entry, "archived": None}

    # contradict：旧条目降分归档（可审计、可复活），新条目按来源起分并记 supersedes
    archived = dict(old)
    archived["mem_key"] = archived_key(old["mem_key"], old["revision"])
    archived["status"] = "archived"
    archived["confidence"] = round(max(float(old["confidence"]) - CONTRADICT_PENALTY, FLOOR), 2)
    archived["updated_at"] = ts
    entry = new_entry(
        user_id=old["user_id"], mem_key=old["mem_key"], content=content,
        source=source, explicitness=explicitness,
        memory_type=old.get("memory_type", "long"),
        evidence=[ev], supersedes=[archived["mem_key"]], now=ts,
    )
    entry["revision"] = int(old["revision"]) + 1
    return {"action": "contradicted", "relation": relation, "entry": entry, "archived": archived}


# -----------------------------------------------------------------------------
# 中期周摘要（FR-5）
# -----------------------------------------------------------------------------

def merge_weekly(old_content: str, new_summary: str) -> str:
    """周摘要按行去重合并，旧行在前；old_content 为空时直接取新摘要。"""
    new_summary = (new_summary or "").strip()
    if not old_content:
        return new_summary
    lines = [l.strip() for l in old_content.splitlines() if l.strip()]
    for line in (l.strip() for l in new_summary.splitlines()):
        if line and line not in lines:
            lines.append(line)
    return "\n".join(lines)


def resolve_weekly(user_id, key, summary, old=None, now=None) -> dict:
    """周键写入纯逻辑：无旧行 → 新建（session_summary 0.60 / revision 1 / 半衰期 7 天）；
    有旧行 → 按行去重合并、confidence+0.05、revision+1（FR-5）。"""
    ts = now_or(now)
    if old is None:
        return {"action": "created", "entry": new_entry(
            user_id, key, (summary or "").strip(), source="session_summary",
            memory_type="mid", now=ts,
        )}
    entry = dict(old)
    entry["content"] = merge_weekly(old["content"], summary)
    entry["confidence"] = round(min(float(old["confidence"]) + WEEKLY_MERGE_DELTA, CAP), 2)
    entry["revision"] = int(old["revision"]) + 1
    entry["updated_at"] = ts
    return {"action": "merged", "entry": entry}


# -----------------------------------------------------------------------------
# 召回管线（§6.4 / FR-7）
# -----------------------------------------------------------------------------

def keyword_score(query: str, text: str) -> float:
    """关键词重合度打分（jieba 为硬依赖，FR-7 降级路径）：查询词在文本词中的命中率。"""
    q_words = [w for w in jieba.lcut(query or "") if w.strip()]
    if not q_words:
        return 0.0
    t_words = {w for w in jieba.lcut(text or "") if w.strip()}
    return sum(1 for w in q_words if w in t_words) / len(q_words)


def recall_pipeline(candidates, query, now=None, final_top_k=FINAL_TOPK) -> list:
    """召回管线后段（§6.4）：有效置信度过滤（≥0.25）→ 加权重排 → 终选。

    candidates 每项 = {**entry, "sim": float}，sim 由上游给定（向量 score 或关键词分）。
    本函数不负责 touch 写回（由存储层执行）。
    """
    ts = now_or(now)
    scored = []
    for cand in candidates:
        eff = effective_confidence(cand, now=ts)
        if eff < INJECT_FLOOR:
            continue
        last = cand.get("last_accessed_at") or ts
        age_days = max((ts - last).total_seconds() / 86400.0, 0.0)
        recency = 0.5 ** (age_days / RECENCY_HALFLIFE_DAYS)
        final = round(W_SIM * float(cand.get("sim", 0.0)) + W_EFF * eff + W_RECENCY * recency, 4)
        scored.append({**cand, "eff": eff, "tier": classify_tier(eff), "final": final})
    scored.sort(key=lambda x: x["final"], reverse=True)
    return scored[:final_top_k]


# -----------------------------------------------------------------------------
# 摘要与注入渲染（FR-5 / §6.5）
# -----------------------------------------------------------------------------

_SUMMARY_PROMPT = (
    "把以下学生与助教的对话压缩成不超过 200 字的学习进度摘要，"
    "按行输出要点（每行一条），不要寒暄：\n{convo}"
)


def summarize_week(messages, llm=None) -> str:
    """会话摘要（FR-5）：LLM ≤200 字 / 无 Key 规则抽取降级。messages = [{role, content}]。"""
    if llm is not None:
        try:
            convo = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            out = _llm_text(llm.invoke(_SUMMARY_PROMPT.format(convo=convo[:4000]))).strip()
            if out:
                return out[:200]
        except Exception:
            pass
    user_lines = [m["content"].strip() for m in messages if m.get("role") == "user" and m.get("content")]
    return "；".join(user_lines)[:200] or "（无有效对话内容）"


_TIER_HINT = {
    "trusted": "（可信，可直接使用）",
    "active": "（较可信，用不确定措辞）",
    "pending_verify": "（待确认，适时向学生核实）",
}


def render_memory_block(recall_results, weekly_entries, now=None) -> str:
    """记忆块渲染（§6.5）：进 SystemMessage，不污染对话历史。

    长期画像每条带有效置信度数值 + 分档措辞提示；中期按周键标注；末尾附使用规则。
    """
    lines = ["【你对该学生的长期记忆】"]
    if recall_results:
        for r in recall_results:
            hint = _TIER_HINT.get(r.get("tier"), "")
            lines.append(f"- [{r['mem_key']}] {r['content']}（有效置信度 {r.get('eff')}）{hint}")
    else:
        lines.append("-（暂无）")
    if weekly_entries:
        lines.append("【近两周学习进度摘要】")
        for w in weekly_entries:
            lines.append(f"- [{w['mem_key']}] {w['content']}")
    lines.append("【记忆使用规则】")
    lines.append("- 有效置信度 ≥0.75 的记忆可直接使用；0.45~0.75 用「我记得/可能」等不确定措辞；<0.45 先向学生确认再更新。")
    lines.append("- 学生纠正你的说法时，必须调用 update_memory 更新记忆（source 用 user_correction）。")
    lines.append("- 学生自我介绍类新信息请调用 save_memory 保存（key 用 name/goal/weak_point 等语义键）。")
    return "\n".join(lines)
