# =============================================================================
# 10_memory — FastAPI 对话服务：SSE 流式对话 / 会话记录 / 记忆检查 / health
# =============================================================================
# 无状态服务（D2）：短期窗口从 MySQL 加载，图按请求独立构建，可多副本部署。
# SSE 事件（FR-1）：meta（召回预览）→ delta*（增量 token）→ final（回答+memory_log+
# 工具轨迹）→ error（异常兜底）。实现模式遵循微服务指南 §7.6"流式输出"：课程环境直连
# FastAPI；生产过网关需关闭缓冲并透传。
# 时序（outline_dev_doc.md §4.1）：巩固在图内完成（final 前，D3）→ 落库 → final → 向量同步。
# 运行：在 10_memory 目录下 `uvicorn service:app --port 8000`
# =============================================================================

import json
import os
import sys
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import db as store
from graph import build_memory_graph, initial_state
from milvus_index import MilvusIndex
from mysql_backend import HybridBackend, MySQLBackend

CONFIG: dict | None = None
MILVUS: MilvusIndex | None = None


def build_llm():
    """LLM 入口（测试可替换）。未配置 ALIYUN_API_KEY 时返回 None → 全链路规则降级（FR-9）。"""
    try:
        from utils.model_utils import get_qwen_client
        return get_qwen_client()
    except ImportError as e:
        print(f"[降级] LLM 客户端不可用（{e}），走规则降级")
        return None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global CONFIG, MILVUS
    CONFIG = store.load_config()
    try:
        store.init_db()
    except Exception as e:
        # MySQL 是本模块主题，属硬依赖，不降级（FR-9）：错误信息给出启动指引
        raise RuntimeError(
            "MySQL 不可达——本模块硬依赖 MySQL，不做降级（FR-9）。"
            "请检查 .env 的 MYSQL_DB_* 配置，并确认 MySQL 已启动"
            "（Docker 示例：docker run -d -p 3306:3306 -e MYSQL_ROOT_PASSWORD=xxx mysql:8）。"
            f"原始错误：{e}"
        ) from e
    MILVUS = MilvusIndex(CONFIG)
    if not MILVUS.check_connection():
        print("[降级] Milvus 不可达：长期召回退化为关键词打分，对话功能不受影响（FR-9）；"
              "恢复后可调用 rebuild 重建索引")
    yield
    store.close_pool()


app = FastAPI(title="10_memory 三层记忆对话服务", version="1.0", lifespan=lifespan)

# 演示页面（memory_demo.html）可能从 file:// 或其它端口打开，跨域调用本接口需放开 CORS。
# 课程环境允许全放行；生产按网关白名单收紧。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1)
    thread_id: str | None = None
    # 演示注入（FR-4 生产触发说明：生产由会话结束触发的 LLM 事实抽取填充，保证确定性演示）
    pending_facts: list[dict] | None = None
    summary_due: bool = False


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


@app.post("/api/chat")
def chat(req: ChatRequest):
    try:
        thread_id = store.get_or_create_thread(req.user_id, req.thread_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    llm = build_llm()
    backend = HybridBackend(llm=llm)
    window = store.load_window(thread_id, (CONFIG or store.load_config())["keep_last_n"])
    graph, _memory_log, _tool_trace = build_memory_graph(
        backend, llm=llm, user_id=req.user_id, thread_id=thread_id)
    state = initial_state(req.user_id, thread_id, req.message,
                          pending_facts=req.pending_facts, summary_due=req.summary_due,
                          history_window=window)

    # 用户消息在开流前落库：客户端随时断流都不会丢（FR-2）。
    # 不能放在生成器末尾——Starlette 断流时直接 raise ClientDisconnect，生成器里的
    # finally 不保证执行完，正常路径的 append 会被整个跳过。
    store.append_messages(thread_id, req.user_id, [("user", req.message, None)])
    model_name = getattr(llm, "model_name", None) or ("rule-fallback" if llm is None else None)

    def sse_stream():
        meta_sent = False
        final_state = dict(state)
        streamed: list[str] = []   # 断流兜底：已推送的增量文本
        assistant_written = False
        try:
            for mode, payload in graph.stream(state, stream_mode=["messages", "values"],
                                              config={"recursion_limit": 30}):
                if mode == "values":
                    final_state = payload
                    if payload.get("meta_ready") and not meta_sent:
                        meta_sent = True
                        yield _sse("meta", {
                            "thread_id": thread_id,
                            "recall": payload.get("recall_preview") or [],
                            "weekly_count": len(payload.get("weekly_entries") or []),
                        })
                else:  # messages：仅透传 agent 节点的回答 token（FR-1 增量流式）
                    chunk, meta_info = payload
                    if (meta_info or {}).get("langgraph_node") == "agent":
                        text = getattr(chunk, "content", "")
                        if isinstance(text, str) and text:
                            streamed.append(text)
                            yield _sse("delta", {"content": text})
            # 巩固已在图内完成（final 前，D3）；现在追加本轮 assistant（FR-2）
            answer = final_state.get("answer", "")
            store.append_messages(thread_id, req.user_id, [("assistant", answer, model_name)])
            assistant_written = True
            yield _sse("final", {
                "thread_id": thread_id,
                "answer": answer,
                "memory_log": final_state.get("memory_log") or [],
                "tool_trace": final_state.get("tool_trace") or [],
                "weekly_written": final_state.get("weekly_written", False),
                "recall": final_state.get("recall_preview") or [],
            })
            # 响应流结束后执行向量同步（FR-6：M3 前请求内同步；失败不阻断，可 rebuild 修复）
            for item in backend.pending_sync:
                try:
                    if item["action"] == "delete":
                        MILVUS.delete_entry(item["user_id"], item["mem_key"])
                    else:
                        entry = backend.get(item["user_id"], item["mem_key"])
                        if entry is not None:
                            MILVUS.sync_entry(entry)
                except Exception as e:
                    print(f"[降级] 向量同步失败（可 rebuild 修复）：{e}")
        except Exception as e:
            yield _sse("error", {"message": f"{type(e).__name__}: {e}"})
        finally:
            # 断流（Ctrl+C / 提前关闭连接）时补写已推送的部分回答，让会话记录与用户所见一致。
            # 断连走 ClientDisconnect（BaseException 子类），故这里捕获 BaseException。
            if not assistant_written:
                try:
                    store.append_messages(thread_id, req.user_id, [
                        ("assistant", "".join(streamed), model_name)])
                except BaseException as e:
                    print(f"[降级] 断流补写助手消息失败：{type(e).__name__}: {e}")

    return StreamingResponse(sse_stream(), media_type="text/event-stream")


@app.get("/api/threads/{thread_id}/messages")
def list_thread_messages(thread_id: str):
    """完整对话记录，seq 升序（FR-2）。"""
    try:
        store.get_or_create_thread("", thread_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return {"thread_id": thread_id, "messages": store.load_full_messages(thread_id)}


@app.get("/api/users/{user_id}/memories")
def list_user_memories(user_id: str, include_archived: bool = False):
    """记忆检查接口（FR-8）：全部字段 + eff/tier；归档行按 __archived 后缀标注原 key。
    同时是验收演示的观察窗口。"""
    entries = MySQLBackend().list_memories(user_id, include_archived=include_archived)
    return {
        "profile": [e for e in entries if e["memory_type"] == "long"],
        "weekly": [e for e in entries if e["memory_type"] == "mid"],
    }


@app.post("/api/admin/rebuild-index")
def rebuild_index():
    """从 MySQL 全量重建 mem_vectors（验收 §10 第 7 条：Milvus 恢复/索引清空后的修复入口）。

    派生索引可直接重建的证明：drop 集合 → 扫 mem_entry 长期 active 条目 → 重新 embed 写入。
    """
    entries = MySQLBackend().list_all_long_entries()
    try:
        written = MILVUS.rebuild_from_entries(entries)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Milvus 重建失败：{type(e).__name__}: {e}") from e
    return {"long_entries": len(entries), "indexed": written,
            "collection": "mem_vectors"}


@app.get("/health")
def health():
    """存活 + 依赖连接状态（FR-9 观察辅助）。MySQL 不可达不影响本接口返回。"""
    mysql_ok = False
    try:
        with store.borrow() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
            mysql_ok = True
    except Exception:
        pass
    milvus_ok = bool(MILVUS and MILVUS.check_connection(quiet=True))
    return {"status": "ok" if mysql_ok else "degraded",
            "mysql": mysql_ok, "milvus": milvus_ok}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
