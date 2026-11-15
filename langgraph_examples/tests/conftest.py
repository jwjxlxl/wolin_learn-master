# =============================================================================
# langgraph_examples/tests 共享设置
# =============================================================================
# 10_memory 目录名以数字开头，无法作为常规包导入（import 语句层面不合法），
# 故在此将目录加入 sys.path 后按扁平模块导入——仅用于测试该目录自身代码。
# 纯逻辑测试（test_memory_core）不触发任何网络/数据库连接。
# =============================================================================

import os
import sys
import uuid

import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_MEM_DIR = os.path.abspath(os.path.join(_TESTS_DIR, "..", "10_memory"))
_ROOT = os.path.abspath(os.path.join(_TESTS_DIR, "..", ".."))
for _p in (_MEM_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import db as store  # noqa: E402


@pytest.fixture()
def clean_user():
    """隔离测试用户：先建库建表（幂等），teardown 清理该用户在全部表中的数据。"""
    store.init_db()
    user_id = f"ut_{uuid.uuid4().hex[:10]}"
    yield user_id
    try:
        with store.borrow() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM chat_message WHERE user_id=%s", (user_id,))
            cur.execute("DELETE FROM chat_thread WHERE user_id=%s", (user_id,))
            cur.execute("DELETE FROM mem_entry WHERE user_id=%s", (user_id,))
            conn.commit()
    except Exception:
        pass
