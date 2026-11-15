# =============================================================================
# 接口层测试：FastAPI TestClient 覆盖 FR-1 / FR-2 / FR-8（需求 §8）
# =============================================================================
# LLM 以 FakeLLM 注入（service.build_llm 可替换），保证确定性、不消耗真实 API；
# MySQL 不可达时优雅 skip（FR-9 硬依赖）。SSE 事件断言：meta / delta / final。
# =============================================================================

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import service as svc
from mysql_backend import MySQLBackend


def _mysql_reachable() -> bool:
    import pymysql
    import db as store
    try:
        conn = pymysql.connect(connect_timeout=3, **store.load_config()["mysql"])
        conn.close()
        return True
    except Exception:
        return False


MYSQL_REACHABLE = _mysql_reachable()

pytestmark = pytest.mark.skipif(
    not MYSQL_REACHABLE, reason="MySQL 不可达：本模块硬依赖（FR-9），接口测试跳过")


class FakeLLM:
    """确定性假 LLM：bind_tools 返回自身，invoke 返回固定回答（不产生工具调用）。"""

    model_name = "fake-model"

    def bind_tools(self, tools):
        return self

    def invoke(self, messages, **kwargs):
        return AIMessage(content="好的，我记住了，李明！")


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(svc, "build_llm", lambda: FakeLLM())
    with TestClient(svc.app) as test_client:  # 进入上下文触发 lifespan（init_db + Milvus 检查）
        yield test_client


def _final_event(body: str) -> dict:
    for line in body.splitlines():
        if line.startswith("data: ") and '"answer"' in line:
            return json.loads(line[len("data: "):])
    raise AssertionError("final 事件缺失")


# ---- FR-1：SSE 对话（meta / delta / final 事件）----

def test_chat_sse_flow(client, clean_user):
    response = client.post("/api/chat", json={"user_id": clean_user, "message": "我叫李明"})
    assert response.status_code == 200
    body = response.text
    assert "event: meta" in body
    assert "event: delta" in body
    assert "event: final" in body

    final = _final_event(body)
    assert final["answer"] == "好的，我记住了，李明！"
    assert final["thread_id"]

    # FR-2：对话后 chat_message 出现 user + assistant 两行
    messages = client.get(f"/api/threads/{final['thread_id']}/messages").json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["model"] == "fake-model"


def test_chat_reuses_thread_with_window(client, clean_user):
    first = _final_event(client.post(
        "/api/chat", json={"user_id": clean_user, "message": "第一问"}).text)
    second = client.post("/api/chat",
                         json={"user_id": clean_user, "message": "第二问", "thread_id": first["thread_id"]})
    assert second.status_code == 200
    messages = client.get(f"/api/threads/{first['thread_id']}/messages").json()["messages"]
    assert [m["content"] for m in messages] == ["第一问", "好的，我记住了，李明！", "第二问", "好的，我记住了，李明！"]


def test_chat_thread_not_found(client):
    assert client.post("/api/chat",
                       json={"user_id": "u1", "message": "hi", "thread_id": "no-such-thread"}
                       ).status_code == 404


# ---- FR-4 巩固路径：pending_facts 演示注入 → 写入 → 下一轮召回 ----

def test_chat_pending_facts_then_recall(client, clean_user):
    first = _final_event(client.post("/api/chat", json={
        "user_id": clean_user,
        "message": "你好",
        "pending_facts": [{"mem_key": "name", "content": "学生叫李明", "source": "agent_inferred"}],
    }).text)
    assert any(entry["action"] == "created" and entry["mem_key"] == "name"
               for entry in first["memory_log"])

    memories = client.get(f"/api/users/{clean_user}/memories").json()
    assert memories["profile"][0]["mem_key"] == "name"
    assert memories["profile"][0]["source"] == "agent_inferred"

    # 第二轮：meta 事件带召回预览（跨 thread/轮次记住学生）
    second = client.post("/api/chat",
                         json={"user_id": clean_user, "message": "我叫什么名字？",
                               "thread_id": first["thread_id"]})
    assert '"mem_key": "name"' in second.text


# ---- FR-5 周摘要：summary_due 演示注入 ----

def test_chat_summary_due_writes_weekly(client, clean_user):
    response = client.post("/api/chat",
                           json={"user_id": clean_user, "message": "今天学了函数", "summary_due": True})
    assert '"weekly_written": true' in response.text
    memories = client.get(f"/api/users/{clean_user}/memories").json()
    assert memories["weekly"] and memories["weekly"][0]["memory_type"] == "mid"


# ---- FR-8 记忆检查接口：归档行标注原 key ----

def test_memories_endpoint_with_archived(client, clean_user):
    backend = MySQLBackend()
    backend.put(clean_user, "name", "我叫李明")
    backend.put(clean_user, "name", "不对，我叫李强")

    res = client.get(f"/api/users/{clean_user}/memories").json()
    assert res["profile"] and res["profile"][0]["mem_key"] == "name"
    assert all(e["status"] == "active" for e in res["profile"])

    full = client.get(f"/api/users/{clean_user}/memories?include_archived=true").json()
    archived = [e for e in full["profile"] if e["status"] == "archived"]
    assert archived and archived[0]["original_key"] == "name"


# ---- FR-2 / health ----

def test_messages_endpoint_404(client):
    assert client.get("/api/threads/nonexistent/messages").status_code == 404


def test_health(client):
    body = client.get("/health").json()
    assert body["mysql"] is True
