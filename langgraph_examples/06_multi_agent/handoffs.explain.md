# handoffs.py — 函数解释与流程图

## 文件概览

本文件演示 **Handoff 模式**：控制权从一个 Agent 完全转移到下一个 Agent，就像接力赛交棒，前一个 Agent 交出控制权后就不再参与。

---

## 函数：`handoff_with_llm()` — LLM 版 Handoff

### 函数解释

**目标：** 用真正的 LLM 做分诊和诊断决策。

**架构：**

```
START → 分诊护士（LLM 结构化输出判断科室）
            ├── 心内科（LLM 诊断 → 决定是否转放射科）
            ├── 神经内科（LLM 诊断 → END）
            ├── 放射科（LLM 模拟报告 → END）
            └── 全科医生（LLM 评估 → END）
```

**关键改进：**

1. **分诊用结构化输出** — `TriageDecision(BaseModel)` 约束 LLM 返回 `department`（Literal 类型，只能选指定科室）和 `reason`，确保输出可控。

2. **各科室用 LLM 扮演角色** — 每个科室节点给 LLM 不同的系统提示（"你是心内科医生..."），让 LLM 根据角色给出专业判断。

3. **心内科仍可二次转诊** — 心内科用 LLM 诊断后，判断是否需要放射科拍片，需要则返回 `Command(goto="radiology")`，不需要则 `Command(goto=END)`。

### 流程图

```
                          handoffs.py — handoff_with_llm() 流程图

START
  │
  ▼
┌─────────────────────────────────────────────────┐
│            triage_nurse 节点                     │  分诊护士（LLM 版）
│                                                  │
│  triage_model.with_structured_output(             │
│    TriageDecision                                │
│  )                                                │
│                                                  │
│  LLM 输出: {                                     │
│    department: "cardiology" | "neurology" |       │
│                "general" | "end",                │
│    reason: "分诊理由"                             │
│  }                                                │
│                                                  │
│  Command(goto=decision.department)                │
└──────┬────────────┬────────────┬─────────────────┘
       │            │            │
       ▼            ▼            ▼
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │cardiology│ │neurology │ │ general  │
  │          │ │          │ │          │
  │ LLM 诊断  │ │ LLM 诊断  │ │ LLM 评估  │
  │          │ │          │ │          │
  │ 需拍片?   │ │ → END    │ │ → END    │
  │  ↓        │ │          │ │          │
  │ RADIOLOGY │ │          │ │          │
  │ →radiology│ │          │ │          │
  │ 否→END    │ │          │ │          │
  └────┬─────┘ │          │ │          │
       │       │          │ │          │
       ▼       │          │ │          │
  ┌──────────┐ │          │ │          │
  │radiology │ │          │ │          │
  │          │ │          │ │          │
  │ LLM 报告  │ │          │ │          │
  │ → END    │ │          │ │          │
  └────┬─────┘ │          │ │          │
       │       │          │ │          │
       ▼       ▼          ▼ ▼          ▼
       ═══════════════════════════════════
                     END

  就诊路径示例（LLM 动态决定）：
  "胸闷、心悸一周，偶有胸痛" → 分诊 → 心内科 → 放射科 → END
  "头痛、眩晕三天"           → 分诊 → 神经内科 → END
  "发烧 38 度，伴随咳嗽"      → 分诊 → END
```

**LLM 结构化输出示例：**

```python
# 分诊决策模型 — 约束 LLM 只能选指定科室
class TriageDecision(BaseModel):
    department: Literal["cardiology", "neurology", "general", "end"] = Field(
        description="应该转到的科室"
    )
    reason: str = Field(description="分诊理由")

triage_model = model.with_structured_output(TriageDecision)
decision: TriageDecision = triage_model.invoke(f"患者症状：{symptom}。请判断转到哪个科室。")
# → decision.department = "cardiology"
# → Command(goto="cardiology")
```

---

## Handoff 模式核心要点一图总结

```
Handoff 模式本质：

START → 节点 A（用 Command(goto="B") 交出控制权）
           │
           ▼
        节点 B（接手后独立决策）
           │
           ├── Command(goto="C") → 再转给 C
           └── Command(goto=END) → 结束

关键特征：
  ✅ 用 Command(goto=...) 代替 add_conditional_edges
  ✅ 每个节点独立决策，前一个节点不参与后续
  ✅ 不需要在图构建时预设所有路由映射
  ✅ 适合"接力式"流程：A → B → C → END（线性或分支）

与 Agent-as-Tool 的关键区别：
  Agent-as-Tool:  主 Agent ──调用──→ 子 Agent ──返回──→ 主 Agent 继续
  Handoff:        节点 A ──交出──→ 节点 B ──不再回来──→ END
                  （像打电话问同事）   （像接力赛交棒）
```
