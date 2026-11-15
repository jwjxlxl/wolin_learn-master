# 10_memory 概要开发文档

- **版本**：v1.1（环境配置对齐项目根目录 .env 实际变量名）
- **日期**：2026-11-12
- **需求依据**：[mysql_memory_dialogue_requirements.md](mysql_memory_dialogue_requirements.md) v0.5（唯一需求设计依据）
- **文档边界**：只写实现流程、模块划分与交付结果；算法公式、表结构 DDL、判定规则等细节一律引用需求文档章节号，不在本文展开。开发顺序 = 里程碑顺序（见 §6）。全局编码约定见仓库根 CLAUDE.md。

## 1. 技术栈选型

| 类别 | 选型 | 说明 |
|---|---|---|
| 语言 | Python 3.10+ | 课程统一环境，Windows 为主 |
| API 框架 | FastAPI + Uvicorn | SSE 用 StreamingResponse；生产过网关的差异按需求 FR-1 注解 |
| LLM 编排 | LangGraph `StateGraph` | 5 节点记忆图，每请求独立构建、不挂 Checkpointer（需求 D2） |
| LLM | `qwen-plus`（OpenAI 兼容） | 经 `utils/model_utils.get_qwen_client()`，需 `ALIYUN_API_KEY` |
| Embedding | 阿里云 `text-embedding-v4` | 1024 维，`check_embedding_ctx_length=False`（需求 §6.2） |
| RDB | MySQL 8.x + PyMySQL | 唯一权威存储；自写极简连接池（需求 D7：固定大小、无健康检查、≤50 行） |
| VDB | Milvus 2.4+（pymilvus） | 派生瘦索引，可随时从 MySQL 全量重建（需求 D6） |
| 中文分词 | jieba | **硬依赖**，Milvus 不可达时的关键词打分（需求 FR-7） |
| 配置 | 环境变量（项目根目录 `.env`，已配置，python-dotenv 读取） | 沿用全项目统一配置，模块不单独维护模板；禁止硬编码 |
| 测试 | pytest + FastAPI TestClient | 依赖不可达时优雅 skip（复用 rag_examples `check_connection` 模式） |
| 基础设施 | Docker（MySQL、Milvus） | 与 rag_examples/00_setup 指南一致 |
| 进阶选学（M4） | Redis 5+ | 仅作巩固任务队列 broker，单一职责（需求 D3）；连接复用根目录 `.env` 的 REDIS_* |

选型约束来自需求 ADR：MySQL 而非 PostgreSQL（D1）；无 Checkpointer、服务无状态（D2）；Redis 非 MVP 依赖（D3）；Milvus 只索引长期画像（D4）；中期与长期合表 mem_entry（D5）。

## 2. 代码结构设计

### 2.1 文件布局（7 文件约束，需求 §9）

```text
langgraph_examples/10_memory/
  README.md          # 课程说明 + 架构总览 + 演示指引
  db.py              # 环境变量、连接池、建表 DDL、会话/消息读写
  memory_core.py     # 纯算法层：置信度/衰减/分档/关系判断/明确度/重排/摘要/渲染
  mysql_backend.py   # MySQLBackend + HybridBackend（MemoryBackend 四方法，事务+行锁）
  milvus_index.py    # Milvus 瘦索引：embed（伪向量降级）/增删查/同步/重建
  graph.py           # 5 节点记忆图 + 记忆工具
  service.py         # FastAPI 入口：对话/查询/健康接口 + SSE + 流后同步钩子
langgraph_examples/tests/
  test_memory_core.py    # 算法层确定性测试（now/llm 注入）
  test_backends.py       # 后端层测试（不可达 skip；改写自失效的 test_memory.py）
  test_service.py        # 接口层测试（FR-1/FR-2/FR-8）
```

M4 的 `worker.py` 超出 7 文件约束，随 M4 文档以独立小节给出（需求 §9）。

### 2.2 依赖方向（单向，禁止环）

```text
service.py ──► graph.py ──► mysql_backend.py ──► db.py
    │             │               │
    │             │               └──► memory_core.py（事务内 get→judge→put 的判定）
    │             └──► memory_core.py（召回管线、注入渲染）
    └──► db.py（窗口/消息查询）、milvus_index.py（流结束后同步）
mysql_backend.py（HybridBackend.search）──► milvus_index.py
memory_core.py：纯算法层，零存储依赖，可独立单测
```

### 2.3 模块入口清单（概要签名，实现时命名可微调）

**db.py**
- `load_config()`：读根目录 `.env` 环境变量（MYSQL_DB_* / MILVUS_* / ALIYUN_API_KEY / USE_REAL_EMBEDDING / KEEP_LAST_N，清单见 §7）；MySQL 库名用模块常量 `wolin_memory`，与业务库隔离
- `get_pool()` / `close_pool()`：极简连接池
- `init_db()`：建库建表（chat_thread / chat_message / mem_entry，DDL 见需求 §6.1）
- `create_thread()` / `append_messages()` / `load_window()` / `load_full_messages()` / `list_messages()`

**memory_core.py**（全部纯函数，`now=None` / `llm=None` 注入，需求 §6.4 确定性约束）
- `judge_relation_and_explicitness(old, new_content, llm=None)`：新旧关系 + 明确度，**同一次 LLM 调用**输出（FR-10）
- `initial_confidence(source, explicitness)`：来源基准分 × 明确度系数
- `effective_confidence(entry, now=None)`：FLOOR + 半衰期衰减（用进废退）
- `classify_tier(eff)`：四档状态（trusted / active / pending_verify / archived）
- `rerank(candidates, query, now=None)`：0.6×sim + 0.3×eff + 0.1×recency，终选 FINAL_TOPK=3
- `summarize_week(messages, llm=None)`：周摘要（LLM ≤200 字 / 规则抽取降级）
- `merge_weekly(old, new)`：按行去重合并（confidence +0.05、revision +1）
- `render_memory_block(profile, weekly)`：注入渲染（需求 §6.5）

**mysql_backend.py**
- `MySQLBackend`（实现 `MemoryBackend` 四方法 put / get / search / delete）：put 内以事务 + `SELECT ... FOR UPDATE` 包裹 get→judge→put（需求 §4）；检索命中 touch 写回
- `HybridBackend(MySQLBackend)`：search = Milvus 粗召回 → MySQL 主键批量回表 → memory_core 管线；Milvus 不可达退化关键词打分（FR-7）

**milvus_index.py**
- `check_connection()` / `ensure_collection()`：集合 mem_vectors（瘦索引 schema 见需求 §6.2）
- `embed(text)`：真实 embedding / hash 种子伪向量降级（带警示打印）
- `upsert_entry()` / `delete_entry()` / `search_candidates(filter=user_id)`：候选取 CANDIDATE_TOPK=50
- `sync_entry(entry)`：content 快照比对去重（需求 FR-6）
- `rebuild_from_mysql()`：全量重建（派生可重建的直接证明）

**graph.py**
- `TOOL_SPECS`：save_memory / search_memory / update_memory / forget_memory（闭包绑定 backend）
- `build_memory_graph(backend, llm)`：5 节点拓扑 recall → prepare → agent ⇄ tools → consolidate
- 节点函数 `recall_node` / `prepare_node` / `consolidate_node`

**service.py**
- `POST /api/chat`（SSE）/ `GET /api/threads/{thread_id}/messages` / `GET /api/users/{user_id}/memories` / `GET /health`
- 流结束后同步钩子：收集本轮长期条目变更 → `milvus_index.sync_entry`

## 3. 功能模块与需求映射

| 功能模块 | 承载文件 | 需求 | 里程碑 |
|---|---|---|---|
| SSE 对话接口 | service.py | FR-1 | M1 |
| 会话/消息持久化 | db.py + service.py | FR-2 | M1 |
| 短期窗口管理 | db.py（查询）+ graph.py prepare | FR-3 | M1 |
| 长期记忆读写（置信度 + 明确度） | memory_core.py + mysql_backend.py + graph.py 记忆工具 | FR-4 / FR-10 | M2 |
| 语义召回与重排（含降级） | milvus_index.py + HybridBackend + memory_core.py | FR-7 / FR-9 | M2 |
| 向量同步与重建 | milvus_index.py + service.py 钩子 | FR-6 | M2 |
| 记忆检查接口 | service.py | FR-8 | M2 |
| 中期周摘要 | memory_core.py + graph.py consolidate | FR-5 | M3 |
| 异步巩固（进阶选学） | worker.py（独立小节）+ Redis | D3 | M4 |

## 4. 核心实现流程

### 4.1 对话主流程（一次 POST /api/chat）

1. 解析请求 `{user_id, message, thread_id?}`；thread_id 缺省则服务端创建（FR-1，无显式建会话接口）。
2. **recall 节点**：HybridBackend.search 召回长期画像（流程见 4.3）；按周键主键直达读取中期摘要（FR-5）。
3. **prepare 节点**：db.load_window 取该 thread 最近 KEEP_LAST_N=8 条消息；记忆块 render_memory_block 渲染进 SystemMessage（不污染对话历史，§6.5）。
4. **agent ⇄ tools**：ReAct 循环；记忆工具经闭包调用 MySQLBackend（写入流程见 4.2），流式推送 delta。
5. **consolidate 节点**：pending_facts / summary_due 注入时执行巩固（触发方式同 FR-4）；MVP 中巩固在 final 事件前同步完成（D3）。
6. 落库：user + assistant 两行一次性追加 chat_message（FR-2）。
7. SSE 收尾：发送 final（完整回答 + memory_log + 工具轨迹）→ 流结束后执行向量同步（4.5）→ 关闭连接。

SSE 事件时序：`meta`（recall/prepare 完成后：thread_id、召回预览 key/内容/有效置信度/状态、中期摘要条数）→ `delta*` → `final` → `error`（任意异常兜底）。

### 4.2 长期记忆写入流程（upsert_memory，事务 + 行锁）

1. 开启事务，`SELECT ... FOR UPDATE` 读同 (user_id, mem_key) 旧条目。
2. 同一次 LLM 调用产出：新旧关系（confirm / contradict / supplement / unrelated）+ 新证据明确度（explicit / implicit / hedged）；`llm=None` 走规则降级。
3. 分支处理：confirm 加分追加证据 / supplement 合并内容 / contradict 旧条目归档至 `{key}__archived_{revision}`、新条目按来源起分并记 supersedes（归档行 mem_key 数组）/ unrelated 拒写（防记忆污染）。
4. 初始分 = 来源基准 × 明确度系数（FR-10），写回并维护 revision，COMMIT。
5. 产生 memory_log（供 final 事件与教学打印）；长期条目变更登记待同步清单（交给 4.5）。

### 4.3 语义召回流程（含降级分支）

1. Milvus `search(query, filter=user_id)` 粗召回 CANDIDATE_TOPK=50 候选 pk。
2. MySQL 主键批量回表取权威条目（瘦索引不存业务字段，D6）。
3. memory_core 管线：有效置信度过滤（≥0.25）→ 重排 0.6×sim + 0.3×eff + 0.1×recency → 终选 FINAL_TOPK=3。
4. 命中条目 touch 写回（刷新 last_accessed_at / hit_count；touch 不触发向量同步，FR-6）。
5. **降级分支**：Milvus 不可达 → MySQL 全量扫 + jieba 关键词重合度打分，接同一管线后段（步骤 2-4 不变）；对话功能不受影响（FR-9）。

### 4.4 中期周摘要流程（consolidate，summary_due 注入）

1. 从 MySQL 读本会话**全量消息**（非仅窗口）。
2. `summarize_week`：LLM ≤200 字摘要，无 Key 走规则抽取降级。
3. 按本次会话所属 ISO 周键读旧摘要 → `merge_weekly` 按行去重合并 → 写回同一周键（confidence +0.05、revision +1）。
4. （进阶选学，M3 选做）同一事实连续 ≥2 周出现 → 升格长期条目（§6.4 摘要升格）。

### 4.5 向量同步与重建流程

1. **同步**（响应流结束后，M3 前请求内同步）：对每条变更条目，比对 mem_entry.content（截断至 8192）与 Milvus 行 content，不一致才调 embedding 并 upsert；delete 调 delete(pk)；touch 不同步。
2. **重建**：`rebuild_from_mysql` 全量扫长期条目重建 mem_vectors 集合（断 Milvus 恢复后的演示步骤，验收 §10 第 7 条）。

### 4.6 降级行为一览（细节见需求 FR-9）

| 依赖缺失 | 行为 | 落点 |
|---|---|---|
| Milvus 不可达 | 召回退化关键词打分，对话不受影响 | HybridBackend |
| ALIYUN_API_KEY 未配置或 USE_REAL_EMBEDDING=false | LLM 判定/摘要走规则降级；embedding 走伪向量（打印警示） | memory_core / milvus_index |
| MySQL 不可达 | 不降级，直接报错并给启动指引 | db.py |

## 5. 对外交付结果

### 5.1 接口一览

| 方法 | 路径 | 功能 | 需求 |
|---|---|---|---|
| POST | `/api/chat` | SSE 对话（meta/delta/final/error） | FR-1 |
| GET | `/api/threads/{thread_id}/messages` | 完整对话记录（seq 升序） | FR-2 |
| GET | `/api/users/{user_id}/memories?include_archived=true` | 记忆检查，含归档行标注原 key | FR-8 |
| GET | `/health` | 存活 + MySQL/Milvus 连接状态 | FR-9 观察辅助 |

### 5.2 数据落点

MySQL 3 张表（chat_thread / chat_message / mem_entry，需求 §6.1）+ Milvus 1 个集合（mem_vectors 瘦索引，需求 §6.2）。所有业务字段以 MySQL 为权威，Milvus 仅供检索候选。

### 5.3 演示可观察结果（对照验收 §10）

- `chat_message` 每轮新增 user + assistant 两行；SSE 流式输出、final 含工具轨迹（验收 1）
- 跨 thread 问答命中画像：meta 事件可见召回记录，回答包含记忆内容（验收 2/3）
- 纠正后 `mem_entry` 出现归档行 `{key}__archived_N`，新条目 supersedes 指向归档行（验收 4）
- 同周多会话合并至同一周键（revision 递增、confidence +0.05/次），下周会话可引用（验收 5）
- 重启 FastAPI 进程，对话与记忆完整保留（验收 6）
- 停 Milvus 对话仍通；重启后 rebuild 索引一致（验收 7）
- `pytest` 全绿（不可达环境自动 skip）；无 Key 环境全流程降级可跑通（验收 8/9）

## 6. 里程碑交付计划（实现顺序）

| 里程碑 | 交付文件 | 验证结果 |
|---|---|---|
| M1 存储与对话最小闭环 | db.py + graph.py（无记忆工具）+ service.py | 能对话、消息落库、**重启服务不丢** |
| M2 长期记忆 + Milvus | memory_core.py + mysql_backend.py + milvus_index.py + 记忆工具 + 降级 | 跨 thread 认人、纠正归档、断 Milvus 仍可对话、无 Key 走降级 |
| M3 中期周摘要 | consolidate 接入 summarize_week / merge_weekly | 同周合并同键、下周引用上周 |
| M4 进阶选学 | Redis 队列 + worker.py（随 M4 文档给出） | final 不再等待巩固、任务重启可恢复 |

每里程碑完成即运行对应测试文件（test_memory_core / test_backends / test_service 随 M1/M2 逐步充实）。

## 7. 环境与启动

**环境变量**：全部沿用项目根目录 `.env`（已配置），本模块不单独维护 `.env` 模板；`db.load_config()` 按下表读取（密钥与地址以根目录 `.env` 为准，此处只列变量名）：

| 服务 | 变量 | 说明 |
|---|---|---|
| MySQL（权威存储） | `MYSQL_DB_HOST` / `MYSQL_DB_PORT` / `MYSQL_DB_USER` / `MYSQL_DB_PASSWORD` / `MYSQL_DB_CHARSET` | 连接参数；库名用模块常量 `wolin_memory`（独立于 `MYSQL_DB_NAME` 指向的业务库），`init_db()` 自动建库 |
| Milvus（派生索引） | `MILVUS_URI` / `MILVUS_DB_NAME` / `MILVUS_USER` / `MILVUS_PASSWORD` | 连接与认证 |
| Redis（M4 选学） | `REDIS_HOST` / `REDIS_PORT` / `REDIS_PASSWORD` | MVP 不读取，M4 作队列 broker |
| LLM / Embedding | `ALIYUN_API_KEY` / `USE_REAL_EMBEDDING` | qwen-plus 与 text-embedding-v4；`USE_REAL_EMBEDDING=false` 等价强制伪向量降级（便于演示降级路径） |
| 可选（有默认值） | `KEEP_LAST_N=8` | 短期窗口条数 |

粗召回/终选常量（CANDIDATE_TOPK=50 / FINAL_TOPK=3）定义在 memory_core.py，不进环境变量。

**启动步骤**：

1. Docker 启动 MySQL 与 Milvus（参照 rag_examples/00_setup；`.env` 已指向现有实例）。
2. 确认根目录 `.env` 已配置 MySQL / Milvus / ALIYUN_API_KEY（Redis 仅 M4 需要）。
3. 首次运行自动建库建表（`init_db()`，服务启动时执行）。
4. 启动服务：`uvicorn service:app --port 8000`（10_memory 目录下）。
5. 演示对话：

```bash
curl -N -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "stu_001", "message": "我叫李明，目标是考研"}'
```

6. 测试：`pytest langgraph_examples/tests -k memory -v`。
