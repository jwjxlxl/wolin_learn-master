# =============================================================================
# 10_memory — 5 节点记忆图：recall → prepare → agent ⇄ tools → consolidate
# =============================================================================
# - 每请求独立构建、不挂 Checkpointer（需求 D2）：短期窗口由服务层从 MySQL 加载
# - 记忆工具经闭包绑定请求级 backend / user_id / thread_id（FR-4）
# - consolidate 在图内完成巩固（pending_facts / summary_due 为演示显式注入；
#   生产由会话结束触发的 LLM 事实抽取填充，FR-4 生产触发说明）
# - 库内不 print：教学打印由本文件节点负责（§6.4 确定性约束）
# =============================================================================

import sys
import os
from datetime import datetime, timedelta
from typing import Annotated, TypedDict

# 导入项目根 utils/（统一路径设置，见 CLAUDE.md）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

import memory_core as mc
import db as store


SYSTEM_PROMPT = (
    "你是一名学生问答学习助手。请用中文回答学生的提问，耐心、准确。"
    "系统消息中给出了你对该学生的记忆（长期画像与近期摘要），按记忆使用规则措辞。"
    "学生透露任何值得长期记住的事实（姓名、目标、学校、薄弱点等）时，"
    "必须在回答前先调用 save_memory 写入记忆，不要只在回复里复述。"
)

TOOL_RESULT_TMPL = "已处理：{action}（mem_key={mem_key}）"


class MemoryGraphState(TypedDict, total=False):
    user_id: str
    thread_id: str
    message: str
    history_window: list          # [(role, content)] 服务层从 MySQL 加载的短期窗口
    recall_preview: list          # meta 事件用：[{mem_key, content, eff, tier}]
    weekly_entries: list          # 近两周中期摘要条目
    meta_ready: bool              # recall 完成标记（SSE meta 发送时机）
    messages: Annotated[list, add_messages]   # agent 循环消息（含注入的记忆块）
    memory_log: list              # 记忆读写日志（教学透明化，禁止落库，§4）
    tool_trace: list              # 工具调用轨迹
    pending_facts: list           # 演示注入：待巩固事实 [{mem_key, content, source}]
    summary_due: bool             # 演示注入：是否执行周摘要巩固
    answer: str
    weekly_written: bool


def build_memory_tools(backend, user_id, thread_id, memory_log, tool_trace) -> list:
    """记忆工具工厂：闭包绑定请求级 backend / user_id / thread_id（FR-4）。"""

    @tool
    def save_memory(mem_key: str, content: str, source: str = "user_stated") -> str:
        """保存一条关于学生的长期记忆。mem_key 用语义键：单值键（name/goal/school 等，不含冒号）
        或命名空间键（weak_point:闭包 这类可多条的键）。content 写完整的事实陈述，不要只写碎片。"""
        result = backend.put(user_id, mem_key.strip(), content.strip(),
                             source=source, thread_id=thread_id)
        memory_log.append({"tool": "save_memory", "mem_key": mem_key.strip(),
                           "action": result["action"], "relation": result.get("relation")})
        tool_trace.append(f"save_memory({mem_key.strip()}) -> {result['action']}")
        return TOOL_RESULT_TMPL.format(action=result["action"], mem_key=mem_key.strip())

    @tool
    def search_memory(query: str) -> str:
        """按语义检索该学生的长期记忆，返回命中条目与置信度分档。"""
        hits = backend.search(user_id, query)
        tool_trace.append(f"search_memory({query}) -> {len(hits)} 条")
        if not hits:
            return "（无命中记忆）"
        return "\n".join(
            f"[{h['mem_key']}] {h['content']}（eff={h['eff']}，{h['tier']}）" for h in hits)

    @tool
    def update_memory(mem_key: str, content: str, source: str = "user_stated") -> str:
        """更新既有记忆：新旧关系自动判定（confirm/supplement/contradict）。学生纠正说法时必须调用本工具
        （source 用 user_correction，content 写学生纠正后的完整新事实）。"""
        mem_key = mem_key.strip()
        if backend.get(user_id, mem_key) is None:
            return f"记忆 {mem_key} 不存在，请改用 save_memory"
        result = backend.put(user_id, mem_key, content.strip(),
                             source=source, thread_id=thread_id)
        memory_log.append({"tool": "update_memory", "mem_key": mem_key,
                           "action": result["action"], "relation": result.get("relation")})
        tool_trace.append(f"update_memory({mem_key}) -> {result['action']}")
        return TOOL_RESULT_TMPL.format(action=result["action"], mem_key=mem_key)

    @tool
    def forget_memory(mem_key: str) -> str:
        """删除一条该学生的记忆（物理删除）。"""
        mem_key = mem_key.strip()
        ok = backend.delete(user_id, mem_key)
        memory_log.append({"tool": "forget_memory", "mem_key": mem_key,
                           "action": "deleted" if ok else "not_found"})
        tool_trace.append(f"forget_memory({mem_key}) -> {'删除成功' if ok else '未找到'}")
        return f"已删除 {mem_key}" if ok else f"未找到记忆 {mem_key}"

    return [save_memory, search_memory, update_memory, forget_memory]


def initial_state(user_id, thread_id, message, pending_facts=None, summary_due=False,
                  history_window=None, keep_last_n=8) -> dict:
    return {
        "user_id": user_id,
        "thread_id": thread_id,
        "message": message,
        "history_window": history_window or [],
        "recall_preview": [],
        "weekly_entries": [],
        "meta_ready": False,
        "messages": [],
        "memory_log": [],
        "tool_trace": [],
        "pending_facts": pending_facts or [],
        "summary_due": bool(summary_due),
        "answer": "",
        "weekly_written": False,
    }


def build_memory_graph(backend, llm, user_id, thread_id, now=None):
    """构建 5 节点记忆图（每请求一次，compile() 不挂 checkpointer，D2）。

    返回 (graph, memory_log, tool_trace)：后两个为闭包共享列表，供服务层取用。
    """
    now = mc.now_or(now)
    memory_log, tool_trace = [], []
    tools = build_memory_tools(backend, user_id, thread_id, memory_log, tool_trace)
    llm_with_tools = llm.bind_tools(tools) if llm is not None else None
    tools_by_name = {t.name: t for t in tools}

    # ---- ① recall：读中长期（长期语义召回 + 中期周键直达）----

    def recall_node(state):
        hits = backend.search(user_id, state["message"])
        weekly = backend.get_weekly(
            user_id, [mc.week_key(now), mc.week_key(now - timedelta(days=7))])
        preview = [{"mem_key": h["mem_key"], "content": h["content"],
                    "eff": h["eff"], "tier": h["tier"], "recall": h.get("recall")}
                   for h in hits]
        print(f"[recall] 长期命中 {len(preview)} 条（{hits[0]['recall'] if hits else '-'}），"
              f"中期摘要 {len(weekly)} 条")
        return {"recall_preview": preview, "weekly_entries": weekly, "meta_ready": True}

    # ---- ② prepare：短期窗口注入 + 记忆块进 SystemMessage（§6.5）----

    def prepare_node(state):
        block = mc.render_memory_block(
            state.get("recall_preview") or [], state.get("weekly_entries") or [], now=now)
        msgs = [SystemMessage(content=f"{SYSTEM_PROMPT}\n\n{block}")]
        for m in state.get("history_window") or []:
            if m["role"] == "user":
                msgs.append(HumanMessage(content=m["content"]))
            else:
                msgs.append(AIMessage(content=m["content"]))
        msgs.append(HumanMessage(content=state["message"]))
        return {"messages": msgs}

    # ---- ③ agent：ReAct 循环（无 Key 时规则降级应答，FR-9）----

    def agent_node(state):
        if llm_with_tools is None:
            known = "；".join(f"{p['mem_key']}: {p['content']}"
                              for p in state.get("recall_preview") or []) or "暂无长期记忆"
            reply = (f"（降级模式：未配置 ALIYUN_API_KEY）已收到你的消息：{state['message']}。"
                     f"我当前记得：{known}")
            return {"messages": [AIMessage(content=reply)]}
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def route_after_agent(state):
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "consolidate"

    # ---- ④ tools：执行记忆工具，失败兜底为 ToolMessage ----

    def tools_node(state):
        last = state["messages"][-1]
        outputs = []
        for call in last.tool_calls:
            name, tid = call["name"], call["id"]
            func = tools_by_name.get(name)
            if func is None:
                outputs.append(ToolMessage(content=f"未知工具 {name}", tool_call_id=tid))
                continue
            try:
                outputs.append(ToolMessage(content=str(func.invoke(call["args"])),
                                           tool_call_id=tid))
            except Exception as e:
                outputs.append(ToolMessage(content=f"工具执行失败：{type(e).__name__}: {e}",
                                           tool_call_id=tid))
        return {"messages": outputs}

    # ---- ⑤ consolidate：巩固（演示注入触发；生产触发方式见 FR-4）----

    def consolidate_node(state):
        answer = ""
        for m in reversed(state["messages"]):
            if isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None):
                answer = m.content if isinstance(m.content, str) else str(m.content)
                break
        for fact in state.get("pending_facts") or []:
            result = backend.put(user_id, fact["mem_key"].strip(), fact["content"].strip(),
                                 source=fact.get("source", "agent_inferred"),
                                 thread_id=thread_id, now=now)
            memory_log.append({"tool": "consolidate", "mem_key": fact["mem_key"],
                               "action": result["action"], "relation": result.get("relation")})
        weekly_written = False
        if state.get("summary_due"):
            convo = [{"role": m["role"], "content": m["content"]}
                     for m in store.load_full_messages(thread_id)]
            convo.append({"role": "user", "content": state["message"]})
            convo.append({"role": "assistant", "content": answer})
            summary = mc.summarize_week(convo, llm=llm)
            key = mc.week_key(now)
            result = backend.put_weekly(user_id, key, summary, now=now)
            weekly_written = True
            memory_log.append({"tool": "consolidate", "mem_key": key,
                               "action": result["action"], "relation": None})
            print(f"[consolidate] 周摘要写回 {key}：{result['action']}（revision={result['entry']['revision']}）")
        return {"answer": answer, "weekly_written": weekly_written,
                "memory_log": list(memory_log), "tool_trace": list(tool_trace)}

    graph = StateGraph(MemoryGraphState)
    graph.add_node("recall", recall_node)
    graph.add_node("prepare", prepare_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("consolidate", consolidate_node)
    graph.add_edge(START, "recall")
    graph.add_edge("recall", "prepare")
    graph.add_edge("prepare", "agent")
    graph.add_conditional_edges("agent", route_after_agent,
                                {"tools": "tools", "consolidate": "consolidate"})
    graph.add_edge("tools", "agent")
    graph.add_edge("consolidate", END)
    return graph.compile(), memory_log, tool_trace
