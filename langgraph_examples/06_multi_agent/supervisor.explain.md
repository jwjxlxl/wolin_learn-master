# supervisor.py — 函数解释与流程图

## 文件概览

本文件演示 **Supervisor 模式**：一个主管 Agent 管理多个工人 Agent，工人完成后必须回到主管汇报，主管审查后再分配下一个任务，形成循环审查的工作流。

---

## 函数：`supervisor_with_llm()` — LLM 版 Supervisor

### 函数解释

**目标：** 用 LLM 做任务分配和审查。

**架构：**

```
START → 项目经理（LLM 结构化输出决定下一个工人）
            ├── researcher（LLM 调研）→ 回到项目经理
            ├── coder（LLM 编码）    → 回到项目经理
            └── reviewer（LLM 评审） → 回到项目经理 →（LLM 认为完成）→ END
```

**关键改进：**

1. **主管用结构化输出** — `SupervisorDecision(BaseModel)` 约束 LLM 返回 `next_worker`（Literal 类型，只能选 `researcher` / `coder` / `reviewer` / `FINISH`）和 `reasoning`，确保输出可控且可解释。

2. **动态分配策略** — LLM 不再按固定顺序，而是根据任务需求、已有进展、最新产出来智能决定下一个工人。例如：如果调研已经很充分，LLM 可能直接跳到 coder；如果觉得代码还需要改进，可能再派给 coder 一轮。

3. **工人用 LLM 扮演角色** — 每个工人节点给 LLM 不同的系统提示，让 LLM 以专业角色执行工作。

### 流程图

```
                          supervisor.py — supervisor_with_llm() 流程图

START
  │
  ▼
┌─────────────────────────────────────────────────┐
│           supervisor 节点（项目经理 - LLM）        │
│                                                  │
│  supervisor_model.with_structured_output(         │
│    SupervisorDecision                            │
│  )                                                │
│                                                  │
│  LLM 输入: 任务 + 已有进展 + 最新产出               │
│                                                  │
│  LLM 输出: {                                     │
│    next_worker: "researcher" | "coder" |          │
│                 "reviewer" | "FINISH",           │
│    reasoning: "分配理由"                           │
│  }                                                │
│                                                  │
│  Command(goto=decision.next_worker)               │
└────┬────────────┬────────────┬───────────────────┘
     │            │            │
     ▼            ▼            ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│researcher│ │  coder   │ │ reviewer │
│          │ │          │ │          │
│ LLM 调研  │ │ LLM 编码  │ │ LLM 评审  │
│          │ │          │ │          │
│ 基于任务   │ │ 参考调研   │ │ 审查全部   │
│ 推荐方案   │ │ 编写实现   │ │ 产出质量   │
│          │ │          │ │          │
│ Command  │ │ Command  │ │ Command  │
│ goto=    │ │ goto=    │ │ goto=    │
│supervisor│ │supervisor│ │supervisor│
└────┬─────┘ └────┬─────┘ └────┬─────┘
     │            │            │
     └────────────┴────────────┘
                  │
                  │ 回到 supervisor（循环审查）
                  │ LLM 重新评估：继续分配 or 交付
                  ▼
         ┌─────────────────┐
         │   supervisor     │ ← 再次 LLM 决策
         │  (审查 + 再分配)   │
         └────────┬────────┘
                  │ LLM 判断 FINISH
                  │ 或 iteration ≥ 5
                  ▼
                END

  执行序列示例（LLM 动态决定，不固定）：
  "搭建知识库问答Agent" → 项目经理 LLM 决策
    可能路径 A: 项目经理 → researcher → 项目经理 → coder → 项目经理 → reviewer → 项目经理 → END
    可能路径 B: 项目经理 → coder → 项目经理 → researcher → 项目经理 → coder → 项目经理 → reviewer → 项目经理 → END
    （LLM 根据实际产出质量决定是否需要补充环节）
```

**LLM 结构化输出示例：**

```python
# 主管决策模型 — 约束 LLM 只能选指定工人
class SupervisorDecision(BaseModel):
    next_worker: Literal["researcher", "coder", "reviewer", "FINISH"] = Field(
        description="下一个要分配的工人，或 FINISH 表示完成"
    )
    reasoning: str = Field(description="为什么分配给这个工人")

supervisor_model = model.with_structured_output(SupervisorDecision)
decision: SupervisorDecision = supervisor_model.invoke(
    f"你是项目经理。任务：{task}。已有进展：{all_products}。"
    f"请选择下一个工人并说明理由。"
)
# → decision.next_worker = "coder"
# → decision.reasoning = "调研已完成，现在需要编写代码实现"
# → Command(goto="coder")
```

---

## Supervisor 模式核心要点一图总结

```
Supervisor 模式本质：

            ┌──────────────────────────────────┐
            │         supervisor（主管）          │
            │  唯一决策者：分配 + 审查 + 决定结束   │
            └────┬─────────┬──────────┬────────┘
                 │         │          │
        分配任务  │         │          │  汇报结果
                 ▼         ▼          ▼
            ┌────────┐┌────────┐┌────────┐
            │worker A││worker B││worker C│
            │研究员   ││程序员   ││评审员   │
            └───┬────┘└───┬────┘└───┬────┘
                │         │         │
                └─────────┴─────────┘
                          │
              Command(goto="supervisor")
              工人不能自行结束，必须回到主管

关键特征：
  ✅ Supervisor 是唯一做决策的 Agent
  ✅ 所有 Worker 完成后必须回到 Supervisor（Command(goto="supervisor")）
  ✅ 可多次循环审查（不满意 → 再分配 → 再审查）
  ✅ 用 iteration 防止无限循环

与 Handoff 的关键区别：
  Handoff:        分诊护士 ──交出──→ 心内科 ──不再回来──→ END
                  （接力赛，交棒退出）
  Supervisor:     项目经理 ──分配──→ 研究员 ──汇报──→ 项目经理再分配
                  （包工头派活，工人干完回来）

与 Agent-as-Tool 的关键区别：
  Agent-as-Tool:  主 Agent ──调用──→ 子 Agent ──返回──→ 主 Agent 继续
                  （工具式：我问你答，主 Agent 掌控）
  Supervisor:     主管 ──分配──→ 工人 ──汇报──→ 主管审查 ──再分配
                  （工作流式：分配→执行→审查→再分配的完整管理循环）
```
