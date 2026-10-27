# 09_checkpointing — Checkpoint 与断点恢复

## 学习目标（对应 5 个核心问题）

| 问题 | 答案要点 | 对应文件 |
|------|---------|---------|
| 1. 断点续跑/恢复如何实现？ | 挂 `Checkpointer` + `thread_id`，用 `interrupt()` 暂停、`Command(resume=...)` 恢复 | `01_checkpoint_basics.py` |
| 2. 运行状态如何持久化与恢复？ | 用 `SqliteSaver` 把检查点落盘到 `.sqlite` 文件 | `02_persistent_resume.py` |
| 3. Checkpoint 何时生成、存了什么？ | 每个 super-step 结束写一个；存 state 值 + 待执行节点 + `checkpoint_id` + 元信息 | `01_checkpoint_basics.py` |
| 4. 进程崩溃后如何找最新 Checkpoint 并恢复？ | 重启后同一 `thread_id` + `get_state_history` 找最新检查点，再 `invoke(None)` / `Command(resume=...)` | `02_persistent_resume.py` |
| 5. 中断/改需求后如何恢复对话状态？ | `update_state` 改写、`checkpoint_id` 时间旅行回滚、`fork` 派生独立分支 | `03_time_travel_fork.py` |

## 核心概念

- **Checkpointer**：LangGraph 的「记忆」。挂上它之后，图才会把每一步的状态存成 **Checkpoint（检查点）**，也才能断点续跑。不挂检查点的图是无状态的，跑完即忘。
- **thread_id**：一次会话的标识。同一 `thread_id` 的多次 `invoke` 共享同一份状态；不同 `thread_id` 互不干扰。
- **Checkpoint 的生成时机**：LangGraph 每次执行完一个 **super-step**（一批节点，直到需要暂停或等待输入的地方）就写入一个检查点；`interrupt()` 暂停处同样会写一个。
- **Checkpoint 存了什么**：各 state 字段的当前值（`values`）、下一步待执行的节点（`next`）、唯一 `checkpoint_id`、父检查点，以及元信息（`source`/`step`/`writes`）。

## 文件清单

```
09_checkpointing/
├── README.md                  ← 本文件
├── 01_checkpoint_basics.py    断点续跑 + Checkpoint 机制（InMemorySaver，内存）
├── 02_persistent_resume.py    状态持久化 + 进程崩溃恢复（SqliteSaver，落盘）
└── 03_time_travel_fork.py     改需求恢复：update_state / 时间旅行 / fork
```

## 依赖与运行

本模块**全部无需 LLM / API Key**，是确定性演示。

```bash
pip install langgraph langgraph-checkpoint-sqlite

# 01 断点续跑 + Checkpoint 机制
python 01_checkpoint_basics.py

# 02 崩溃恢复（分两次运行，模拟两个进程）
python 02_persistent_resume.py          # 第一个进程：运行到中断后崩溃
python 02_persistent_resume.py resume   # 第二个进程：从磁盘恢复继续

# 03 改需求恢复
python 03_time_travel_fork.py
```

## 两种 Checkpointer 对比

| 对比项 | `InMemorySaver` | `SqliteSaver` |
|--------|-----------------|---------------|
| 存储位置 | 进程内存 | 本地 `.sqlite` 文件 |
| 进程退出后 | 数据丢失 | 数据仍在 |
| 能否跨进程/崩溃恢复 | ❌ | ✅ |
| 适用场景 | 学习、单次运行的脚本 | 真实服务 / Worker 长时运行 |
| 额外依赖 | 无（langgraph 自带） | `pip install langgraph-checkpoint-sqlite` |

> 生产环境还可以用 `PostgresSaver`（`langgraph-checkpoint-postgres`）把检查点存进 PostgreSQL，供多实例共享。

## 与 08_student_helper 的关系

- `08_student_helper` 已经演示了「`interrupt()` + `MemorySaver` + `thread_id`」的最基础用法（同进程内的暂停→恢复）。
- 本模块（09）把话题展开：讲清 Checkpoint 的**机制**（何时生成、存了什么）、把检查点**落盘**（崩溃恢复）、以及**改写/回滚/分叉**（改需求后恢复）。
