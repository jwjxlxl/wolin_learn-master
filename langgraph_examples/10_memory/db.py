# =============================================================================
# 10_memory — 存储层：环境配置 + MySQL 极简连接池 + 建表 DDL + 会话/消息读写
# =============================================================================
# 职责（outline_dev_doc.md §2.3）：
#   - load_config(): 读取项目根目录 .env（MYSQL_DB_* / MILVUS_* / REDIS_* 等）
#   - SimplePool:    固定大小、无健康检查的极简连接池（需求 D7）
#   - init_db():     建库 wolin_memory + 3 张表（需求 §6.1）
#   - 消息读写:      chat_thread / chat_message（短期记忆唯一权威存储，FR-2/FR-3）
# 单一属主原则：所有业务数据以 MySQL 为权威（需求 §4）；窗口只是读取时视图，不独立持久化。
# =============================================================================

import queue
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime

import pymysql
import pymysql.cursors
from dotenv import load_dotenv
import os

# 项目根目录（本文件位于 langgraph_examples/10_memory/ 下，根目录在上两级）
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

# 本模块独立库名（与 .env 中 MYSQL_DB_NAME 指向的业务库隔离，需求 §6.1）
DB_NAME = "wolin_memory"

# 连接池固定大小（D7：固定大小、无健康检查）
_POOL_MAX_SIZE = 8


def load_config() -> dict:
    """读取 .env 环境变量，返回本模块全部配置（变量清单见 outline_dev_doc.md §7）。"""
    return {
        "mysql": {
            "host": os.getenv("MYSQL_DB_HOST", "localhost"),
            "port": int(os.getenv("MYSQL_DB_PORT", "3306")),
            "user": os.getenv("MYSQL_DB_USER", "root"),
            "password": os.getenv("MYSQL_DB_PASSWORD", ""),
            "charset": os.getenv("MYSQL_DB_CHARSET", "utf8mb4"),
        },
        "milvus": {
            "uri": os.getenv("MILVUS_URI", "http://localhost:19530"),
            "db_name": os.getenv("MILVUS_DB_NAME", "default"),
            "user": os.getenv("MILVUS_USER", ""),
            "password": os.getenv("MILVUS_PASSWORD", ""),
        },
        "redis": {
            "host": os.getenv("REDIS_HOST", "localhost"),
            "port": int(os.getenv("REDIS_PORT", "6379")),
            "password": os.getenv("REDIS_PASSWORD", "") or None,
        },
        "aliyun_api_key": os.getenv("ALIYUN_API_KEY", ""),
        "use_real_embedding": os.getenv("USE_REAL_EMBEDDING", "true").strip().lower() == "true",
        "keep_last_n": int(os.getenv("KEEP_LAST_N", "8")),
    }


class SimplePool:
    """极简 MySQL 连接池（D7）：get() 取空闲连接或新建（达上限阻塞等待归还），put() 归还。

    无健康检查：连接失效由操作异常暴露，符合教学约束（不写泛化池）。
    """

    def __init__(self, connect_kwargs: dict, max_size: int = _POOL_MAX_SIZE):
        self._kwargs = connect_kwargs
        self._max = max_size
        self._idle: "queue.Queue" = queue.Queue(maxsize=max_size)
        self._lock = threading.Lock()
        self._created = 0

    def _new_conn(self):
        return pymysql.connect(
            autocommit=False,  # 事务由调用方显式 commit/rollback
            cursorclass=pymysql.cursors.DictCursor,
            **self._kwargs,
        )

    def get(self):
        try:
            return self._idle.get_nowait()
        except queue.Empty:
            with self._lock:
                if self._created < self._max:
                    self._created += 1
                    return self._new_conn()
            return self._idle.get()  # 达上限：阻塞等待归还

    def put(self, conn):
        try:
            self._idle.put_nowait(conn)
        except queue.Full:
            conn.close()

    def close_all(self):
        while True:
            try:
                self._idle.get_nowait().close()
            except queue.Empty:
                return


_pool: SimplePool | None = None


def get_pool() -> SimplePool:
    """获取全局连接池（懒加载；连接时要求 wolin_memory 已由 init_db() 创建）。"""
    global _pool
    if _pool is None:
        cfg = load_config()["mysql"]
        _pool = SimplePool({
            "host": cfg["host"], "port": cfg["port"],
            "user": cfg["user"], "password": cfg["password"],
            "charset": cfg["charset"], "database": DB_NAME,
        })
    return _pool


def close_pool():
    global _pool
    if _pool is not None:
        _pool.close_all()
        _pool = None


@contextmanager
def borrow():
    """借出/归还连接的上下文管理器。异常时回滚；归还前兜底回滚，防止未提交事务残留锁。"""
    pool = get_pool()
    conn = pool.get()
    try:
        yield conn
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        pool.put(conn)
        raise
    else:
        try:
            conn.rollback()
        except Exception:
            pass
        pool.put(conn)


# -----------------------------------------------------------------------------
# 建库建表（需求 §6.1 DDL）
# -----------------------------------------------------------------------------

_DDL_TABLES = [
    """
    CREATE TABLE IF NOT EXISTS chat_thread (
        thread_id      VARCHAR(64)  NOT NULL,
        user_id        VARCHAR(64)  NOT NULL,
        created_at     DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
        last_active_at DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
        PRIMARY KEY (thread_id),
        INDEX idx_user (user_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS chat_message (
        id         BIGINT       NOT NULL AUTO_INCREMENT,
        thread_id  VARCHAR(64)  NOT NULL,
        user_id    VARCHAR(64)  NOT NULL,
        seq        INT          NOT NULL,
        role       ENUM('user','assistant') NOT NULL,
        content    MEDIUMTEXT   NOT NULL,
        model      VARCHAR(64)  NULL,
        created_at DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
        PRIMARY KEY (id),
        UNIQUE KEY uk_thread_seq (thread_id, seq),
        CONSTRAINT fk_message_thread FOREIGN KEY (thread_id) REFERENCES chat_thread (thread_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS mem_entry (
        user_id          VARCHAR(64)  NOT NULL,
        mem_key          VARCHAR(128) NOT NULL,
        memory_type      ENUM('long','mid') NOT NULL,
        status           ENUM('active','archived') NOT NULL DEFAULT 'active',
        content          MEDIUMTEXT   NOT NULL,
        source           VARCHAR(32)  NOT NULL,
        confidence       DECIMAL(3,2) NOT NULL,
        halflife_days    INT          NOT NULL,
        hit_count        INT          NOT NULL DEFAULT 0,
        evidence         JSON         NULL,
        supersedes       JSON         NULL,
        revision         INT          NOT NULL DEFAULT 1,
        created_at       DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
        updated_at       DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
        last_accessed_at DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
        PRIMARY KEY (user_id, mem_key),
        INDEX idx_user_type (user_id, memory_type, status)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
]


def init_db():
    """建库建表（幂等）。MySQL 不可达时抛异常，由上层给出启动指引（FR-9 不降级）。"""
    cfg = load_config()["mysql"]
    conn = pymysql.connect(
        host=cfg["host"], port=cfg["port"], user=cfg["user"],
        password=cfg["password"], charset=cfg["charset"], connect_timeout=5,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` DEFAULT CHARACTER SET utf8mb4")
            cur.execute(f"USE `{DB_NAME}`")
            for ddl in _DDL_TABLES:
                cur.execute(ddl)
        conn.commit()
    finally:
        conn.close()


# -----------------------------------------------------------------------------
# 会话与消息读写（FR-2 / FR-3）
# -----------------------------------------------------------------------------

def create_thread(user_id: str) -> str:
    """创建会话。thread_id 只经 FR-1 缺省路径产生（无显式建会话接口）。"""
    thread_id = uuid.uuid4().hex[:32]
    with borrow() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO chat_thread (thread_id, user_id) VALUES (%s, %s)", (thread_id, user_id))
        conn.commit()
    return thread_id


def get_or_create_thread(user_id: str, thread_id: str | None) -> str:
    """thread_id 缺省 → 服务端创建；给定但不存在 → 视为非法请求。"""
    if not thread_id:
        return create_thread(user_id)
    with borrow() as conn, conn.cursor() as cur:
        cur.execute("SELECT thread_id FROM chat_thread WHERE thread_id=%s", (thread_id,))
        if cur.fetchone() is None:
            raise ValueError(f"thread 不存在: {thread_id}")
    return thread_id


def append_messages(thread_id: str, user_id: str, rows: list) -> None:
    """一轮对话结束后一次性追加（FR-2）。rows = [(role, content, model|None)]。

    seq 取 max(seq)+1；UNIQUE 约束兜底并发，冲突时重读 seq 重试（最多 3 次）。
    """
    for _attempt in range(3):
        try:
            with borrow() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT seq FROM chat_message WHERE thread_id=%s ORDER BY seq DESC LIMIT 1 FOR UPDATE",
                    (thread_id,),
                )
                top = cur.fetchone()
                base = top["seq"] if top else 0
                for offset, (role, content, model) in enumerate(rows, start=1):
                    cur.execute(
                        "INSERT INTO chat_message (thread_id, user_id, seq, role, content, model) "
                        "VALUES (%s, %s, %s, %s, %s, %s)",
                        (thread_id, user_id, base + offset, role, content, model),
                    )
                cur.execute(
                    "UPDATE chat_thread SET last_active_at=CURRENT_TIMESTAMP(3) WHERE thread_id=%s",
                    (thread_id,),
                )
                conn.commit()
                return
        except pymysql.err.IntegrityError:
            continue  # 并发写竞争：重读 max(seq) 再试
    raise RuntimeError(f"chat_message 并发写入冲突，重试后仍失败（thread={thread_id}）")


def load_window(thread_id: str, n: int) -> list:
    """短期窗口 = 最近 n 条消息（升序返回）。库内永远保存全量原文（FR-3）。"""
    with borrow() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT role, content FROM chat_message WHERE thread_id=%s ORDER BY seq DESC LIMIT %s",
            (thread_id, n),
        )
        rows = cur.fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def load_full_messages(thread_id: str) -> list:
    """全量对话记录，seq 升序（FR-2 查询接口 / FR-5 摘要输入）。"""
    with borrow() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT seq, role, content, model, created_at "
            "FROM chat_message WHERE thread_id=%s ORDER BY seq ASC",
            (thread_id,),
        )
        rows = cur.fetchall()
    return [
        {
            "seq": r["seq"],
            "role": r["role"],
            "content": r["content"],
            "model": r["model"],
            "created_at": r["created_at"].strftime("%Y-%m-%d %H:%M:%S") if r["created_at"] else None,
        }
        for r in rows
    ]
