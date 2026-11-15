# 10_memory — 企业标准三层记忆对话系统（MySQL + Milvus）

学生问答学习助手：Agent 跨会话记住学生画像（姓名 / 基础 / 目标 / 薄弱点），按周积累学习进度摘要，按置信度动态修正记忆。本模块是 09_checkpointing（Checkpointer 持久化）之后的进阶实战——**不用 Checkpointer 的企业标准对话服务形态**。

- 需求设计依据：`mysql_memory_dialogue_requirements.md`（v0.5，唯一设计依据）
- 概要开发文档：`outline_dev_doc.md`（技术栈 / 代码结构 / 实现流程 / 里程碑）

## 架构总览

```
学生 → FastAPI 对话服务（SSE，无状态，可多副本）
         │ ①短期窗口从 MySQL 加载 ②每轮消息一次性追加落库
         ▼
     记忆图（5 节点）recall → prepare → agent ⇄ tools → consolidate
         │                      │
   memory_core 算法层      MySQLBackend / HybridBackend
   （置信度/衰减/分档/       MySQL = 唯一权威存储（system of record）
    关系判断/明确度/重排）    Milvus = 派生瘦索引（可随时从 MySQL 全量重建）
```

三层记忆与数据职责：**短期** = chat_message 按 seq 派生的读取时视图（库内永远保存全量原文）；**中期** = mem_entry 按 ISO 周键的摘要（不进 Milvus，主键直达）；**长期** = mem_entry 语义键画像（Milvus 只索引这一层）。

## 算法机制速览（完整规格见需求文档 §6.4 / FR-10）

- **初始置信度** = 来源基准分 × 明确度系数：亲口说的（user_stated 0.90）> 客观数据（0.75）> 摘要升格（0.60）> 机器推断（0.50）> 外部导入（0.40）；明确度 explicit ×1.0 / implicit ×0.8 / hedged ×0.6。
- **有效置信度**（读时派生，不落盘）：`eff = 0.05 + (confidence − 0.05) × 0.5^(age/半衰期)`，检索命中 touch 刷新（用进废退），遗忘可逆。
- **四档状态**：≥0.75 trusted（可断言）/ ≥0.45 active（"我记得你…"）/ ≥0.25 pending_verify（待确认）/ <0.25 archived（不注入，可复活）。
- **新旧关系判断**（upsert，事务 + 行锁内）：confirm +0.05 / supplement 合并+0.05 / contradict 旧条目 −0.30 归档到 `{key}__archived_N`、新条目记 supersedes / unrelated **拒写**（防记忆污染）。LLM 判定 + 规则降级双路径，`llm=None` 即规则路径。

## 检索两形态对比（为什么 Milvus 只存瘦索引）

| 维度 | 瘦索引 + 回表（本模块采用，D6） | payload 快照制（完整条目冗余进向量库） |
|---|---|---|
| 检索路径 | Milvus 出候选 pk → MySQL 主键回表取权威 | 一次向量查询直接返回全部字段 |
| touch 之后 | 权威字段永远最新（无需同步向量） | 快照陈旧，需同步整个 payload |
| 字段一致性 | 单一属主：MySQL 是唯一权威 | 两处存储，易漂移 |
| 重建能力 | drop 集合后从 MySQL 全量重建即可 | 重建需重新构造全部 payload |
| 代价 | 每轮召回多一次主键批量点查（毫秒级） | 省一次回查 |

生产触发说明（重要）：图中的 `pending_facts` 与 `summary_due` **没有内部生产者**，由演示显式注入（`POST /api/chat` 请求体）以保证课堂确定性；生产环境由"会话结束触发的 LLM 事实抽取"填充。另：降级阶梯下（Milvus 不可达 / 无 Key 伪向量）跨会话认人在**小数据量演示**下成立——伪向量相似度无语义，候选集靠全量返回兜底，切勿外推为功能等价。

## 快速开始

```bash
# 1. 准备依赖（项目根 requirements.txt 已含 pymysql/jieba）
pip install pymysql jieba

# 2. 配置：项目根 .env 已包含 MYSQL_DB_* / MILVUS_* / ALIYUN_API_KEY / USE_REAL_EMBEDDING
#    （Redis 仅 M4 进阶选学需要）

# 3. 建库建表在服务启动时自动执行（库名 wolin_memory）
cd langgraph_examples/10_memory
uvicorn service:app --port 8000
```

演示脚本（对应需求 §10 验收）：

> 下面的 `$THREAD` 取自首轮的 `meta` 事件；Git Bash / Linux 下可直接
> `THREAD=$(curl -sN ... | head -2 | grep -o '"thread_id": "[^"]*"' | cut -d'"' -f4)`，
> PowerShell 或 Postman 里就手工把 `thread_id` 复制进去。

```bash
# ① 首轮对话：自我介绍（观察 SSE meta/delta/final 事件与工具轨迹）
curl -N -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "stu_001", "message": "我叫李明，我的目标是考研"}'

# ② 查看落库的记忆与消息
curl http://localhost:8000/api/users/stu_001/memories
curl http://localhost:8000/api/threads/<thread_id>/messages

# ③ 纠正：观察归档行 {key}__archived_N 与新条目 supersedes
curl -N -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "stu_001", "thread_id": "<thread_id>", "message": "不对，我的目标其实是考公"}'

# ④ 周摘要巩固（summary_due 演示注入）+ 记忆检查
curl -N -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "stu_001", "summary_due": true, "message": "这周学习总结一下"}'
curl "http://localhost:8000/api/users/stu_001/memories?include_archived=true"

# ⑤ 重启 uvicorn 进程 → 对话记录与记忆完整保留（对比教学版"重启即丢"）
```

### 课堂演示对话脚本（按讲解点排序）

每一条都对应一个可观察的现象，按顺序讲完即覆盖三层记忆与置信度机制。

**A. 跨会话记住学生（短期 → 长期）**

```json
{"user_id": "stu_demo", "message": "我叫李明，我的目标是考研，目前最头疼的是英语阅读"}
```
观察：`final.tool_trace` 出现 `save_memory(name)` / `save_memory(goal)` / `save_memory(weak_point)`。
再开一个**新会话**（不带 `thread_id`）问：

```json
{"user_id": "stu_demo", "message": "你还记得我是谁吗？"}
```
观察：`meta.recall` 命中 3 条，且 `recall` 字段为 `vector`（Milvus 路径）；回答里直接说出姓名与目标。

**B. 用进废退（hit_count 与衰减）**

同一个 `user_id` 连问三次"我的薄弱点是什么"，然后看记忆接口：
`hit_count` 递增、`eff` 随命中回升——衰减不是删除，遗忘可逆。

**C. 说了就改（contradict → 归档）**

```json
{"user_id": "stu_demo", "thread_id": "<同一会话>", "message": "我之前说错了，我其实不考研了，我准备考公"}
```
观察：`final.memory_log` 里 `goal` 的 `action` 为 `contradicted`；
`/api/users/stu_demo/memories?include_archived=true` 里同时出现 `goal`（新值）与 `goal__archived_1`（旧值，`original_key: "goal"`）。

**D. 明确度影响置信度（一句话的措辞决定可信度）**

| 说法 | 来源判定 | 初始 confidence |
|---|---|---|
| "我**一定**要考研" | user_stated + explicit | 0.90 |
| "我大概想考研" | user_stated + hedged | 0.54 |
| Agent 自己推断的结果 | agent_inferred | 0.50 |

同一句事实分别以不同措辞说一次，对比 `/api/users/{id}/memories` 中两条的 `eff` 与 `tier`：
前者 `trusted`，后者通常 `active`——回答时的措辞也跟着变（"你目标是考研" vs "我记得你好像想考研"）。

**E. 记忆工具直调（不经 LLM，确定性强）**

```json
{"user_id": "stu_demo", "message": "帮我记一下", "pending_facts": [
  {"mem_key": "school", "content": "李明就读于华北电力大学", "source": "user_stated"},
  {"mem_key": "weak_point:听力", "content": "英语听力是薄弱点", "source": "user_stated"}
]}
```
命名空间键 `weak_point:听力` 演示"同一维度可挂多条"，与单值键 `weak_point` 并存。

**F. 无 Key / 无 Milvus 的降级演示**

停掉 Milvus 或清空 `ALIYUN_API_KEY` 再跑 A：
- 对话照常出答案（`delta` 正常流出）；
- `meta.recall` 的 `recall` 变成 `keyword`，服务端打印 `[降级] Milvus 不可达…`；
- `/health` 里 `milvus: false`。

**G. 记忆检查的完整视图**

```bash
curl "http://localhost:8000/api/users/stu_demo/memories?include_archived=true"
```
返回 `profile`（长期）与 `weekly`（中期周摘要）两组，每条带 `confidence` / `eff` / `tier` /
`hit_count` / `evidence` / `supersedes`，是讲"置信度机制"最直观的窗口。

### 可视化演示页面（`memory_demo.html`）

纯前端单文件，双击用浏览器打开即可，不需要构建：

- **左栏对话**：SSE 流式输出；每轮下方显示本轮召回情况、`memory_log` 里的记忆动作（新建/确认/补充/冲突改写/拒写）与工具轨迹。
- **右栏学生画像**：每条长期记忆显示 `mem_key`、内容、四档状态徽标、`eff` 置信度刻度条、原始 `confidence`、`hit_count`、来源与半衰期；含归档条目开关。
- **召回预览**：`meta` 事件里本轮命中的条目及召回通道（向量 / 关键词），是讲"两段式检索"的直观窗口。
- **周摘要**：`summary_due` 勾选后生成的中期记忆随之展示。
- 顶部可切换 `user_id`、开新会话（保留长期记忆）、刷新记忆、一键重建索引。

页面通过 `fetch` 跨域调用 `http://localhost:8000`，服务端已用 `CORSMiddleware` 放开（课程环境全放行，生产按网关白名单收紧）。



Milvus 是**派生瘦索引**，写入口只有一处：`service.py` 在 SSE `final` 事件之后消费
`backend.pending_sync` 逐条同步。具体条件：

| 条件 | 说明 |
|---|---|
| 必须是长期条目 | `memory_type='long'`；**中期周摘要（mid）不进 Milvus**，按周键主键直达（D4） |
| 必须在本次请求内产生写入 | 来源是 `save_memory` / `update_memory` / `forget_memory` 工具调用，或 `pending_facts` 注入；只读检索不产生同步 |
| 必须跑到 `final` 之后 | 同步在流结束后执行。**客户端中途断开（Ctrl+C）不会同步**，本轮新记忆只在 MySQL 里，索引里没有 |
| 内容有变化才写 | 先比对 Milvus 里存的 content 快照，一致则跳过，避免重复 embed |
| Milvus 必须可达 | 同步失败不阻断对话，只打印降级日志；这些条目会**缺失索引**，需要重建修复 |

所以"MySQL 有、Milvus 没有"是正常状态，不是 bug。恢复方式（MySQL 永远是权威）：

```bash
# 一条命令从 MySQL 全量重建索引（drop 集合 → 扫 mem_entry 长期 active → 重新 embed）
curl -X POST http://localhost:8000/api/admin/rebuild-index
# → {"long_entries": 12, "indexed": 12, "collection": "mem_vectors"}
```

注意：跑 `pytest langgraph_examples/tests/test_backends.py` 时，重建用例会 **drop 整个集合**。
测试已改用独立集合 `mem_vectors_test`，不会动 `mem_vectors`；但如果你手动调了
`rebuild_from_entries` 并只传了部分条目，索引同样会被裁剪——重建后用上面的 admin 接口再刷一次即可。


降级阶梯：Milvus 不可达 → 长期召回退化为 jieba 关键词打分，对话不受影响；`ALIYUN_API_KEY` 未配置或 `USE_REAL_EMBEDDING=false` → LLM 判定/摘要走规则降级，embedding 走伪向量（打印警示）；MySQL 不可达 → 服务启动直接报错并给出指引（硬依赖，不降级）。

## 测试

```bash
# 项目根目录运行
pytest langgraph_examples/tests/test_memory_core.py -v   # 算法层：纯逻辑，无需任何服务
pytest langgraph_examples/tests/test_backends.py -v      # 后端层：MySQL 不可达自动 skip
pytest langgraph_examples/tests/test_service.py -v       # 接口层：FakeLLM 注入，不耗真实 API
```

## 文件说明（单模块 ≤ 7 文件约束）

| 文件 | 职责 |
|---|---|
| `db.py` | 环境配置、极简连接池（D7）、建表 DDL、消息读写 |
| `memory_core.py` | 算法层纯函数：置信度/衰减/分档/关系/明确度/重排/摘要/渲染 |
| `mysql_backend.py` | MySQLBackend（事务+行锁）与 HybridBackend（两段式检索+降级） |
| `milvus_index.py` | mem_vectors 瘦索引：embed（伪向量降级）/增删查/同步/重建 |
| `graph.py` | 5 节点记忆图 + save/search/update/forget_memory 四个记忆工具 |
| `service.py` | FastAPI：SSE 对话 / 会话记录 / 记忆检查 / health |

M4 进阶选学（Redis 队列 + 巩固 Worker 异步化）未包含在 MVP，随课程文档独立给出。
