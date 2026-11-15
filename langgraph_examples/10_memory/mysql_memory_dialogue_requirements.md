# 企业标准三层记忆对话系统 — 需求文档（MySQL + Milvus）

- **状态**：v0.5（评审修订：FR-7 管线参数统一并引用 §6.4、supersedes 改存归档行 mem_key、删除死字段 category/tags/title、content 快照比对替代 content_hash、jieba 定为硬依赖、极简连接池边界、删 POST /api/threads；含 FR-10 写入端明确度评分；本文档为唯一设计依据）
- **日期**：2026-11-12
- **业务场景**：学生问答学习助手——学生通过对话接口与 Agent 交流，Agent 跨会话记住学生画像（姓名/基础/目标/薄弱点）、按周积累学习进度摘要、按置信度动态修正记忆

## 1. 背景与目标

本目录原有基于 InMemoryStore 的三层记忆教学模块（概念讲解 + 内存演示），已完成教学使命后整体移除。本模块作为其替代与升级，**直接以企业标准形态实现**：

- **MySQL 为唯一权威存储（system of record）**：完整对话记录、中期周摘要、长期画像事实全部落库，进程重启不丢、支持多副本部署。
- **Milvus 为派生的向量检索索引**：只加速长期画像的语义召回，数据可随时从 MySQL 全量重建。
- **提供真实 HTTP 对话接口**：SSE 流式对话 + 记忆检查接口，学生用 curl/页面即可观察"Agent 记住了什么"。

## 2. 范围

| 包含 | 不包含（说明去向） |
|---|---|
| 记忆算法层（置信度/衰减/关系判断/明确度评分，本模块自研，规格见 §6.4） | 用户认证体系（生产由 API 网关承担，课程假定 user_id 已认证，呼应微服务指南"认证权限服务"） |
| 三层记忆的 MySQL 持久化 | 多租户计费与限流（非功能需求中作生产注解，不实现） |
| SSE 流式对话接口 + 会话/记录查询接口 | LangGraph Checkpointer 持久化（见决策 D2；自定义 MySQLSaver 列为进阶选学） |
| Milvus 向量同步、语义召回、索引重建 | 语义缓存（超纲） |
| 无 Milvus / 无 Key 的降级阶梯 | Redis 队列与异步 Worker（M4 进阶选学，MVP 不引入） |

## 3. 总体架构（自上而下）

```mermaid
flowchart TD
    Client["学生 / 演示页面"] --> API["FastAPI 对话服务\n(SSE 流式 · 无状态 · 可多副本)"]

    API -->|"① 加载短期窗口\n② 追加对话消息"| MySQL[("MySQL — 唯一权威存储\nchat_thread / chat_message / mem_entry")]

    API --> G["记忆图（5 节点）\nrecall → prepare → agent ⇄ tools → consolidate"]

    G --> Algo["memory_core 算法层\n置信度/时间衰减/关系判断/加权重排\n（本模块自研，规格见 §6.4）"]

    G -->|"长期语义召回"| Milvus[("Milvus — 向量索引（派生）\nmem_vectors 瘦索引")]
    G -->|"中期/长期读写"| BE["MySQLBackend + HybridBackend\n(MemoryBackend 接口实现)"]
    Algo --> BE
    BE --> MySQL

    Milvus -->|"topK 候选 key"| BE
    MySQL -->|"主键批量回表取权威条目"| BE

    MySQL -->|"长期画像写入后\n派生同步（content 快照比对去重）"| Sync["向量同步"]
    Sync --> Milvus

    subgraph M4["M4 进阶选学：异步巩固"]
        Q["Redis（单一职责：任务 broker）"] --> W["巩固 Worker\n周摘要 + embedding + Milvus 同步"]
    end
    Sync -.->|"切换为投递任务"| Q
```

分层职责（自上而下）：

1. **接入层 — FastAPI 对话服务**：SSE 流式响应；每个请求无状态，短期窗口从 MySQL 加载，`user_id`/`thread_id` 由请求携带。
2. **编排层 — 记忆图**：5 节点拓扑——recall（读中长期）→ prepare（管短期窗口注入）→ agent⇄tools（ReAct）→ consolidate（巩固）。每请求独立构建，**不使用 Checkpointer**（见 D2）。
3. **算法层 — memory_core**：置信度赋分、时间衰减、四档状态、关系判断、加权重排，本模块自研（规格见 §6.4），与存储解耦（只依赖 `MemoryBackend` 四方法接口）。
4. **存储层 — MySQL（权威）+ Milvus（派生索引）**：职责见第 4 节矩阵。
5. **异步层（进阶）— Redis 队列 + Worker**：仅承担巩固任务投递，见 D3。

## 4. 数据职责矩阵（反冗余核心）

**单一属主规则：每一类数据有且只有一个权威存储；其余系统只能持有派生副本，且派生副本必须能随时从属主完整重建。**

| 数据 | 唯一属主 | 派生 / 消费方 | 明确禁止 |
|---|---|---|---|
| 完整对话记录 | MySQL `chat_message` | 窗口组装（读取时视图）、周摘要输入 | 禁止存入 Milvus / Redis |
| 短期窗口 | （无独立存储）从 `chat_message` 按 seq 派生 | prepare 节点注入 | 禁止把窗口独立持久化（它只是查询视图） |
| 中期周摘要 | MySQL `mem_entry`（memory_type='mid'） | 注入渲染 | 禁止进 Milvus（按周键主键直达，无需向量） |
| 长期画像事实 | MySQL `mem_entry`（memory_type='long'） | Milvus 索引行（瘦索引）、对话注入 | — |
| 语义向量 | Milvus（派生索引） | 长期召回的相似度候选 | 禁止在 Milvus 存权威业务字段（见 D6 瘦索引） |
| 记忆读写日志 | （无存储）请求内存态 | SSE final 事件、教学打印 | 禁止落库（审计需求由对话记录间接满足） |
| 巩固任务 | M3 前：请求进程内；M4 起：Redis 队列 | Worker 执行 | 禁止把会话状态放 Redis（防止第二属主） |

**读-改-写并发控制**：`upsert_memory` 的 get→judge→put 流程在 MySQLBackend 内用**事务 + `SELECT ... FOR UPDATE` 行锁**包裹，不引入分布式锁。

## 5. 功能需求

> 编写规则：每条 FR 只描述一个关注点；与其他 FR 的依赖用 ID 引用，禁止重述内容。

### FR-1 对话接口（SSE 流式）

`POST /api/chat`，请求体 `{user_id, message, thread_id?}`；thread_id 缺省时服务端创建。响应为 SSE 事件流：

- `meta`：thread_id、本轮召回的长期记忆预览（key/内容/有效置信度/状态）、中期摘要条数
- `delta`：增量 token（流式，模型生成即推送）
- `final`：完整回答 + memory_log + 工具调用轨迹（教学透明化）
- `error`：异常兜底

实现模式遵循微服务指南 §7.6"流式输出"：网关透传、关闭缓冲（课程环境直连 FastAPI，注解说明生产过网关的差异）。

### FR-2 会话与消息持久化

- `GET /api/threads/{thread_id}/messages` 返回完整对话记录（seq 升序）；会话创建只走 FR-1 路径（thread_id 缺省时服务端创建，meta 事件返回），不设显式建会话接口。
- 每轮对话结束后，用户消息与助手回答**一次性**追加到 `chat_message`（seq 取 `max(seq)+1`，UNIQUE 约束兜底并发）。

### FR-3 短期记忆管理（窗口 = 读取时视图）

- 注入模型的窗口 = 该 thread 最近 KEEP_LAST_N=8 条消息（可配），由接入层查询并放入图 state；**库内永远保存全量原文，被裁消息永不丢失**。
- 工具消息不落库（`role` 仅 user/assistant）：工具轨迹经 FR-1 final 事件透出供教学观察；持久化窗口只由 user/assistant 行组装，不存在"窗口起点切在工具消息上"的情形，无需孤儿防护。

### FR-4 长期画像与置信度（算法层规格见 §6.4）

- 读写全部经 `MySQLBackend`（实现 `MemoryBackend` 四方法：put/get/search/delete），置信度算法由 `memory_core` 提供（自研，规格见 §6.4）。
- 语义检索路径由 `HybridBackend.search` 提供：Milvus 粗召回 topK 候选 → MySQL 主键批量回表 → 组装为 Item(score=sim)（见 D6；依赖 FR-7）。
- 触发写入的主体：Agent 工具（save/search/update_memory，闭包绑定 backend）+ consolidate 的 pending_facts 路径。
- **生产触发说明（设计要求，必须写入代码注释与 README）**：pending_facts 与 summary_due 在图中无内部生产者，由演示显式注入以保证确定性；生产环境由"会话结束触发的 LLM 事实抽取"填充，避免学生误以为闭环完整。

### FR-5 中期周摘要

- 行为规格：会话结束时把本会话**全量消息**（从 MySQL 读全量，而非仅窗口）压缩成摘要（LLM ≤200 字 / 规则抽取降级）；同一 ISO 周的多次会话"读出旧摘要 → 按行去重合并 → 写回同一周键"，每次合并 confidence +0.05、revision +1。
- 写入时机：consolidate 在 summary_due=True 时执行（触发方式同 FR-4）。
- 中期条目 source/半衰期规格见 §6.4；摘要升格（mid→long）为进阶选学（§6.4）。

### FR-6 向量同步（派生规则）

| MySQL 事件 | Milvus 动作 | 去重规则 |
|---|---|---|
| mem_entry 新建 / 内容更新（confirm/supplement/contradict 新条目） | upsert(pk, mem_key, content, vector) | 比对 mem_entry.content（截断至 8192）与 Milvus 行 content，不一致才调 embedding |
| 归档行写入（`{key}__archived_N`） | upsert（content 通常变化） | 同上 |
| touch（命中刷新 hit_count/last_accessed_at） | **不同步** | 权威字段回表制保证（D6），content 未变 |
| delete（forget_memory） | delete(pk) | — |

- 同步执行时机：M3 前为请求内同步（响应流结束后执行）；M4 起改投递 Redis 队列。
- **索引重建**：提供从 MySQL 全量扫表重建 Milvus 索引的函数（派生可重建的直接证明；降级恢复演示用）。

### FR-7 语义召回与重排

- 长期召回：Milvus `search(query, filter=user_id)` 粗召回候选 → MySQL 主键批量回表 → memory_core recall 管线（过滤/重排/topK 规格统一见 §6.4）。
- 降级路径（依赖 FR-9）：Milvus 不可达时退化为 MySQL 全量扫 + 关键词重合度打分（jieba 分词，jieba 为硬依赖）。

### FR-8 记忆检查接口（教学透明化）

`GET /api/users/{user_id}/memories?include_archived=true` 返回 `{profile: [...], weekly: [...]}`，每条含全部字段（confidence/status/evidence/supersedes/hit_count）。归档行按 `__archived` 后缀识别并标注原 key。此接口同时是验收演示的观察窗口。

### FR-9 降级阶梯

| 依赖缺失 | 行为 |
|---|---|
| Milvus 不可达 | 长期召回退化关键词打分（FR-7），对话功能不受影响 |
| ALIYUN_API_KEY 未配置 | LLM 摘要/关系判断走规则降级（§6.4 双路径规格）；embedding 走 hash 种子伪向量（同文本同向量、可复现；能跑通 schema/过滤，但相似度无语义，需打印警示） |
| MySQL 不可达 | **不降级，直接报错**——MySQL 是本模块主题，属硬依赖；错误信息给出启动指引 |

### FR-10 写入端明确度评分（抽取置信度）

- 初始置信度不只由来源决定（§6.4 来源表回答"谁说的"），还需衡量"说得多确定"：`初始分 = 来源基准分 × 明确度系数`，写入时一次折算，不新增字段、不引入第二套评分。
- 明确度三档与判定规格见 §6.4"明确度系数"；`llm=None` 时按词表 + 疑问句确认模式确定性降级（§6.4 确定性约束）。
- LLM 判定时与新旧关系判断（§6.4）合并为同一次调用输出，不增加额外 LLM 往返。
- 低分入账即激活四档体系的上升通路：pending_verify 条目带"待确认"措辞注入（§6.5）→ 检索命中 touch → confirm 加分升档。
- **不做的事**：不为召回端增加第二套评分（recall 管线的重排分即"该不该用"的使用置信度）；不在表上加独立评分列（折算即写入，单一属主）。
- 展示不变：FR-8 / FR-1 meta 事件仍展示 confidence / eff；evidence[].kind 可选记录明确度取值供教学观察。

## 6. 数据模型

### 6.1 MySQL（3 张表，中期与长期合表）

```text
chat_thread
  thread_id   VARCHAR(64) PK
  user_id     VARCHAR(64)          INDEX idx_user(user_id)
  created_at / last_active_at  DATETIME(3)

chat_message
  id          BIGINT AUTO_INCREMENT PK
  thread_id   VARCHAR(64)          FK -> chat_thread
  user_id     VARCHAR(64)
  seq         INT NOT NULL         UNIQUE uk_thread_seq(thread_id, seq)
  role        ENUM('user','assistant')
  content     MEDIUMTEXT
  model       VARCHAR(64) NULL     -- assistant 行记录模型名
  created_at  DATETIME(3)
  -- 窗口读取：WHERE thread_id=? ORDER BY seq DESC LIMIT N

mem_entry                      -- 中期+长期合表：条目 schema 同源（防重复建模）
  user_id     VARCHAR(64) \
  mem_key     VARCHAR(128) /     PK(user_id, mem_key)
                                   -- 长期 = 语义键（name/goal/weak_point:条件边）
                                   -- 中期 = ISO 周键（2026-W45）
                                   -- 归档行 = {原key}__archived_{revision}（见 §6.4 upsert 语义）
  memory_type ENUM('long','mid')
  status      ENUM('active','archived')   -- 写时物化的派生列（§6.4 分档结果）
  content     MEDIUMTEXT
  source      VARCHAR(32)
  confidence  DECIMAL(3,2)         -- 原始分；有效分读时派生（FLOOR+衰减），不落库
  halflife_days INT                hit_count INT
  evidence    JSON                 supersedes JSON
  revision    INT
  created_at / updated_at / last_accessed_at  DATETIME(3)
  INDEX idx_user_type(user_id, memory_type, status)
```

设计说明：中期与长期条目结构与置信度体系完全同源（同一 entry schema），合表避免第二张结构重复的表；按 `memory_type` + 主键直达读取（中期读取 = 按 2 个周键主键查询）。

### 6.2 Milvus（1 个集合，瘦索引）

```text
mem_vectors
  pk        VARCHAR(193) PK  = "{user_id}:{mem_key}"   -- 天然 upsert；长度 = 64(user_id) + 1(分隔符) + 128(mem_key)
  user_id   VARCHAR(64)      -- 过滤
  mem_key   VARCHAR(128)
  content   VARCHAR(8192)    -- embed 输入（仅供检索调试展示）
  vector    FLOAT_VECTOR(1024)  AUTOINDEX / COSINE
  -- 不存 confidence/status/payload：权威字段一律回表取（D6）
```

embedding：阿里云 `text-embedding-v4`（1024 维，`check_embedding_ctx_length=False`），与全项目维度约定一致。

### 6.3 复用映射（防跨模块重复开发）

| 已有能力 | 来源 | 本模块动作 |
|---|---|---|
| Milvus 连接、降级检查模式 | `rag_examples/milvus_config.py` | 复用配置入口与 `check_connection` 模式 |
| Checkpointer / 时间旅行 / 断点恢复 | `09_checkpointing/` | 不重复实现，README 引用说明分界 |
| SSE / 调大超时 / 异步三模式 | `microservices_architecture_guide.md` §7.6 | 按"流式"模式实现，文档引用 |
| Redis 客户端用法 | `redis_example/` + 指南 §7.5 | M4 引入时复用模式 |

### 6.4 记忆算法层规格（memory_core.py，本模块自研）

**条目 schema**（MySQL `mem_entry` 行的内存形态）：条目身份 = (user_id, mem_key)；content、memory_type、confidence（原始分）、source、status、halflife_days、hit_count、evidence[]、supersedes[]（归档行 mem_key 字符串数组）、revision、created_at / updated_at / last_accessed_at。每条 evidence 记录 {thread_id, quote, at, kind}（可溯源；kind 可选记录明确度取值，见 FR-10）。

**来源 → 初始置信度 / 半衰期**（原则：亲口说的 > 客观数据 > 二手概括 > 机器推断 > 外部导入）：

| source | 初始分 | 半衰期 |
|---|---|---|
| user_correction（学生纠正） | 0.95 | 180 天 |
| user_stated（亲口陈述） | 0.90 | 90 天 |
| tool_result（测验/工具客观数据） | 0.75 | 60 天 |
| session_summary（周摘要 / 摘要升格，二手） | 0.60 | 30 天 |
| agent_inferred（Agent 推断） | 0.50 | 14 天 |
| imported（外部导入） | 0.40 | 30 天 |

未知来源按最低档（0.40）处理；中期条目（mid）source 固定取 session_summary，半衰期固定 7 天（覆盖来源表半衰期列）。

**明确度系数（写入端评分，FR-10）**：explicit 明确陈述/直接确认（第一人称自述；对提问答"对/是/没错"）×1.0；implicit 上下文间接可得（**默认档**）×0.8；hedged 含糊/猜测/弱确认（词表命中"可能/大概/好像/算是吧"）×0.6。LLM 判定与新旧关系判断合并为同一次调用输出；规则降级（`llm=None`）：hedged 词表命中 → hedged，疑问句得到"对/是"类短确认 → explicit，否则 implicit。半衰期仍按来源查表；confirm/supplement 加分作用于折算后的原始分。示例：user_stated+explicit = 0.90（trusted 边缘）、user_stated+hedged = 0.54（active，措辞不确定）、agent_inferred+hedged = 0.30（pending_verify，待核实）。

**有效置信度（读时派生，不落盘）**：`eff = FLOOR + (confidence − FLOOR) × 0.5^(age / halflife_days)`，age 基于 `last_accessed_at`（**用进废退**：检索命中 touch 刷新 last_accessed_at 与 hit_count，衰减重新起算）；FLOOR=0.05（**遗忘可逆**：不衰减到 0，归档记忆被再次提到可爬升）；CAP=0.95（永不给 1.0）。

**四档状态**（由 eff 分档，决定注入措辞）：≥0.75 trusted（可断言）/ ≥0.45 active（"我记得你…"措辞）/ ≥0.25 pending_verify（标"待确认"，适时核实）/ <0.25 archived（不注入不返回，数据保留可复活）。注入下限 0.25。

**新旧关系判断**（upsert 时对同 key 新旧内容判定，LLM 判定 + 规则降级双路径）：

| relation | 调分 | 处理 |
|---|---|---|
| confirm（新观察支持旧记忆） | +0.05~0.10 | 原始分加分（收敛于 CAP），追加证据，revision+1 |
| contradict（否定旧记忆） | 旧条目 −0.30 | 旧条目降分并**归档**（键规则与 supersedes 方向见下方统一约定）；新条目按来源起分并记 supersedes |
| supplement（补充细节） | +0.05 | 合并内容，confidence = min(max(旧分, 新来源初始分)+delta, CAP) |
| unrelated（主题无关） | 0 | **拒写**（防记忆污染） |

规则降级判定：否定词+话题重合 → contradict；高度一致 → confirm；关键词重合 → supplement；否则 unrelated。

**supersedes 方向（统一约定）**：由**新条目**指向被其取代的归档行，supersedes 存归档行 mem_key 字符串数组（如 `["goal__archived_2"]`）；contradict 产生的新条目 revision = 旧条目 revision + 1，归档行主键 = `{原key}__archived_{旧条目revision}`——同一键的归档后缀随 revision 单调递增、永不冲突；归档行自身不改写 supersedes 字段。

**摘要升格（mid→long，进阶选学，M3 选做）**：consolidate 写周摘要时识别与历史摘要重复出现的稳定事实（语义键一致或关键词重合达标），同一事实连续出现在 ≥2 个 ISO 周的周摘要中即升格为长期条目——memory_type='long'、source='session_summary'（初始分 0.60、半衰期 30 天）、revision 继承该事实的历史合并次数；原中期行不变、照常衰减。触发方式同 FR-4。

**recall 管线**：粗召回（`CANDIDATE_TOPK=50`）→ 有效置信度过滤（≥0.25）→ 相似度打分（向量 score，无则关键词重合度）→ 加权重排 `final = 0.6×sim + 0.3×eff + 0.1×recency` → 终选 `FINAL_TOPK=3` → 命中条目 touch 写回。

**确定性约束**（可测试性硬要求）：所有时间函数接受 `now=None` 注入；所有 LLM 函数接受 `llm=None` 并带规则降级；库内不 print（教学日志由图节点负责）。

### 6.5 注入渲染规格

记忆块进 SystemMessage（不污染对话历史）：长期画像每条带有效置信度数值 + 分档措辞提示；中期摘要按周标注；末尾附使用规则（≥0.75 可直接使用 / 0.45~0.75 用不确定措辞 / <0.45 先确认再更新 / 被纠正必须调 update_memory）。

## 7. 关键技术决策（ADR 摘要）

- **D1 用 MySQL 替代 PostgreSQL**：课程环境与学生技能栈要求。代价：LangGraph 无官方 MySQL Checkpointer → 引出 D2。补偿：事务 + 行锁满足读-改-写一致性。
- **D2 弃用 Checkpointer，MySQL 对话记录为短期记忆权威，服务无状态**：每请求从 MySQL 加载窗口，图按请求独立构建（`compile()` 不挂 checkpointer）。放弃的 interrupt/时间旅行已由 09 独立教授；此形态恰是"对话 API 服务"的企业标准形态。进阶选学：自定义 MySQLSaver（对照 09 教学）。
- **D3 Redis 非核心依赖，M4 仅以"异步巩固队列 broker"单一职责引入**：LLM 摘要/embedding/同步为秒级耗时，不应阻塞 SSE final 事件；MVP 用请求内同步执行并在 final 事件前完成（延迟诚实标注）。**明文禁止**：会话缓存（避免第二属主）、分布式锁（MySQL 行锁已覆盖）、语义缓存（超纲）。
- **D4 Milvus 只索引长期画像**：中期按周键主键直达、短期按 seq 窗口直达，均无语义检索需求；给它们建向量索引属于需求冗余。
- **D5 中期与长期合表 `mem_entry`**：条目 schema 与置信度体系同源，分表即重复建模；用 memory_type 区分，主键规则统一。
- **D6 检索两段式：Milvus 瘦索引出候选，MySQL 回表出权威**：对比"payload 快照制"（完整条目 JSON 冗余进向量库——省一次回查，但 touch 后快照陈旧、字段冗余），本模块采用回表制——Milvus 只存 pk/user_id/mem_key/content/vector，重排所需字段永远从 MySQL 读最新值。代价：每轮召回多一次主键批量点查（毫秒级）；收益：touch 不同步向量、派生副本最小化、索引可全量重建。
- **D7 技术栈**：PyMySQL（纯 Python、Windows 友好）+ 自写极简连接池（db.py 内：固定大小、无健康检查、≤50 行，不引入 SQLAlchemy 以控制教学噪音）；Milvus/pymilvus 与 embedding 复用课程既有约定（1024 维、text-embedding-v4）。
- **D8 记忆算法层自研 memory_core.py**：原计划复用的旧算法库已随教学版移除（且未入 git 版本历史），按 §6.4 规格全新实现；`MemoryBackend` 四方法接口的"算法与存储解耦"设计目标不变。

## 8. 非功能需求

- **环境变量**（模块内 `.env.example` 模板，禁止硬编码）：
  - `MYSQL_URI=mysql+pymysql://user:pass@host:3306/wolin_memory`
  - `MILVUS_URI`、`ALIYUN_API_KEY`（沿用全项目既有约定）
- **平台**：Windows 主环境；MySQL 与 Milvus 均以 Docker 运行（与 rag_examples/00_setup 指南一致）。
- **性能（粗标，教学口径）**：SSE 首 token 延迟不得因架构引入超过 200ms 额外开销；MVP 的巩固在 final 事件前同步执行（延迟 = 摘要生成耗时，README 标注），M4 异步化后 final 立即返回。
- **测试**：
  - 算法层：memory_core 确定性测试（时间注入 now、LLM 注入 None 走规则降级），覆盖来源赋分/明确度折算/衰减曲线/四档分界/四类关系/upsert 全分支/recall 管线。
  - 后端层：MySQLBackend/HybridBackend 测试（不可达时 pytest 优雅 skip，参照 rag_examples 的 `check_connection` 模式）。
  - 接口层：FastAPI TestClient 覆盖 FR-1/FR-2/FR-8。
  - 遗留处置：`langgraph_examples/tests/test_memory.py` 因导入目标已删除而失效，M1 落地时删除或改写为 memory_core 的新测试。

## 9. 里程碑与文件布局

模块目录：`langgraph_examples/10_memory/`（原目录已清空，就地重建，编号延续；README 说明新课程定位）

| 里程碑 | 内容 | 验收物 |
|---|---|---|
| M1 存储与对话最小闭环 | db（含配置）+ 消息读写 + graph（无记忆工具）+ service（SSE） | 能对话、落库、**重启服务对话不丢** |
| M2 长期记忆 + Milvus | memory_core + mem_entry + MySQLBackend + 记忆工具 + HybridBackend + 同步/重建 + 降级 | 跨 thread 认人、纠正归档、断 Milvus 仍可对话 |
| M3 中期周摘要 | consolidate 接入摘要与周合并（全量消息输入） | 同周三会话合并到同周键、下周可引用上周 |
| M4 进阶选学 | Redis 队列 + 巩固 Worker（异步化） | final 事件不再等待巩固；任务重启可恢复 |

文件布局（遵守课程"单模块 ≤ 7 文件"约束，配置并入 db.py）：

```text
10_memory/
  README.md          # 课程说明 + 架构总览 + 算法机制速览(指向 §6.4) + 检索两形态对比 + 生产触发说明
  memory_core.py     # 算法层：置信度/衰减/分档/关系判断/重排/摘要（§6.4 规格的实现）
  db.py              # 环境变量读取 + MySQL 连接池 + 建表 DDL + 消息读写
  mysql_backend.py   # MySQLBackend + HybridBackend（MemoryBackend 实现，事务+行锁）
  milvus_index.py    # 瘦索引封装：embed(伪向量降级) + upsert/delete/search + rebuild
  graph.py           # 5 节点记忆图
  service.py         # FastAPI：SSE 对话 / 会话 / 记忆检查 / health
```

M4 的 `worker.py` 超出 7 文件约束 → 作为"进阶选学"独立小节随 M4 文档给出。

## 10. 验收标准（演示脚本，全部可人工勾选）

1. 对话后 `chat_message` 出现 user + assistant 两行，SSE 流式输出正常，`final` 事件含工具轨迹。
2. thread_A 自我介绍 → `mem_entry` 出现 name/goal（user_stated，0.90），Milvus 出现对应向量行。
3. thread_B 全新会话问"我是谁" → `meta` 事件显示召回记录，回答包含"李明"。
4. 学生纠正 → 归档行 `{key}__archived_N` 出现，新条目 supersedes 指向该归档行，新条目 0.95。
5. 同周三次会话（summary_due 注入）→ 同一周键 revision=3、confidence 0.60→0.70；下周会话能取到上周摘要。
6. **重启 FastAPI 进程 → 对话记录与记忆完整保留**（对比教学版"重启即丢"的核心卖点）。
7. 停掉 Milvus → 对话仍通（关键词降级召回）；重启后执行 rebuild，索引与 MySQL 一致。
8. `pytest` 全绿（含不可达 skip）；无 Key 环境跑通降级路径。
9. 明确度评分：明确自述 → 0.90（user_stated × 1.0）；含糊表述（"好像/可能"类）→ 0.54（× 0.6）且注入措辞相应降档；明确确认后经 confirm 加分升档。

## 11. 开放问题

1. embedding 成本：每条长期记忆内容变化时调用一次，演示数据量下可忽略；生产注解"按租户核算 token 成本"（呼应微服务指南 §7.4）。
2. 进阶 MySQLSaver 路线是否值得做：取决于社区 `langgraph-checkpoint-mysql` 包的成熟度，M4 评审时调研。
3. 窗口 KEEP_LAST_N=8 是否需要按模型上下文自适应（教学固定值即可，生产注解）。

> 已关闭：PyMySQL vs SQLAlchemy——v0.5 评审定案，维持 D7（PyMySQL + 极简连接池）。
