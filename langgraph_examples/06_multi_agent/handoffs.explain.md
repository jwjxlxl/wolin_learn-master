# handoffs.py — 函数解释与流程图

## 文件概览

本文件演示 **Handoff 模式**：控制权从一个 Agent 完全转移到下一个 Agent，就像接力赛交棒，前一个 Agent 交出控制权后就不再参与。

---

## 函数一：`hospital_triage_demo()` — 无 LLM 版医院分诊

### 函数解释

**目标：** 不用 LLM，用关键词匹配展示 `Command(goto=...)` 的接力效果，帮助理解 Handoff 的控制流。

**架构：**

```
START → 分诊护士（症状判断）
            ├── 心脏相关 → 心内科 →（需拍片？）→ 放射科 → END
            ├── 神经相关 → 神经内科 → END
            ├── 常见症状 → END（分诊直接处理）
            └── 其他 → 全科医生 → END
```

**核心 API：** `Command(goto="目标节点", update={状态更新})`
- `goto`：指定下一个要执行的节点名（或 `END`）
- `update`：更新 StateGraph 的状态（类似节点返回的字典）

**关键特征：**

1. **不需要 `add_conditional_edges`** — 与 02_state_and_branching 的条件分支不同，Handoff 中每个节点用 `Command(goto=...)` **自己决定**去哪，不需要在图构建时预设路由映射。

2. **控制权一旦交出就不会回来** — 分诊护士把患者转给心内科后，心内科独立决策（自己处理 or 再转放射科 or 结束），分诊护士不再参与。

3. **就诊路径可追踪** — 用 `path` 字段记录经过的节点，形成完整的就诊链路。

**五个节点：**

| 节点 | 职责 | 下一步 |
|------|------|--------|
| **triage（分诊护士）** | 关键词判断症状归属 | 心内科 / 神经内科 / 全科 / END |
| **cardiology（心内科）** | 心脏问题初步诊断 | 放射科（需拍片）或 END |
| **neurology（神经内科）** | 神经问题诊断 | END |
| **radiology（放射科）** | 拍片出报告 | END |
| **general（全科医生）** | 不确定症状综合评估 | END |

### 流程图

```
                        handoffs.py — hospital_triage_demo() 流程图

START
  │
  ▼
┌─────────────────────────────────────────────────┐
│            triage_nurse 节点                     │  分诊护士：关键词判断
│                                                  │
│  含"胸闷/心悸/心脏/胸痛" ──→ Command(goto="cardiology")
│  含"头痛/眩晕/麻木"     ──→ Command(goto="neurology")
│  含"感冒/发烧/咳嗽"     ──→ Command(goto=END)       直接处理
│  其他                   ──→ Command(goto="general")
└──────────┬──────────┬────────────┬───────────────┘
           │          │            │
           ▼          ▼            ▼
  ┌─────────────┐ ┌──────────┐ ┌──────────┐
  │ cardiology  │ │neurology │ │ general  │
  │ 心内科       │ │神经内科   │ │ 全科医生  │
  └──────┬──────┘ └────┬─────┘ └────┬─────┘
         │              │           │
         │ 需拍片?       │           │
         ▼              │           │
  ┌─────────────┐       │           │
  │ radiology   │       │           │
  │ 放射科       │       │           │
  └──────┬──────┘       │           │
         │              │           │
         ▼              ▼           ▼
       ═══════════════════════════════
                    END

  就诊路径示例：
  "胸闷、心悸" → 分诊 → 心内科 → 放射科 → END
  "头痛眩晕"   → 分诊 → 神经内科 → END
  "发烧38度"   → 分诊 → END
```

**构建图的特点（与条件分支对比）：**

```python
# 传统条件分支：需要在图构建时预设路由映射
.add_conditional_edges("triage", route, {
    "cardiology": "cardiology",
    "neurology": "neurology",
})

# Handoff 模式：不需要条件边！节点返回 Command(goto=...) 自己决定
.add_edge(START, "triage")  # 只需连起点
# 图构建完成。后续所有跳转都由 Command 在运行时决定
```

---

## 函数二：`handoff_with_llm()` — LLM 版 Handoff

### 函数解释

**目标：** 用真正的 LLM 做分诊和诊断决策，架构与示例 1 完全相同，关键词匹配换成 LLM 语义理解。

**架构：**

```
START → 分诊护士（LLM 结构化输出判断科室）
            ├── 心内科（LLM 诊断 → 决定是否转放射科）
            ├── 神经内科（LLM 诊断 → END）
            ├── 放射科（LLM 模拟报告 → END）
            └── 全科医生（LLM 评估 → END）
```

**与示例 1 的核心区别：**

| 维度 | 示例 1（无 LLM） | 示例 2（LLM 版） |
|------|-----------------|-----------------|
| **分诊决策** | `if "胸闷" in symptom` | `triage_model.with_structured_output(TriageDecision)` |
| **各科室诊断** | 硬编码返回固定文本 | LLM 扮演角色给出专业诊断 |
| **放射科** | 硬编码模拟报告 | LLM 根据症状生成真实报告 |
| **适合场景** | 教学理解控制流 | 真实生产环境 |

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

## 两个函数对比总结

| 对比项 | `hospital_triage_demo()` | `handoff_with_llm()` |
|-------|------------------------|---------------------|
| **分诊决策** | 关键词匹配（`if "胸闷" in symptom`） | LLM 结构化输出（`TriageDecision`） |
| **科室诊断** | 硬编码返回固定文本 | LLM 扮演角色给出诊断 |
| **需要 LLM** | 否，可立即运行 | 是（Ollama `qwen3.5:2b`） |
| **教学目的** | 理解 `Command(goto=...)` 的**接力控制流** | 理解**真实生产环境**的做法 |
| **图构建方式** | 完全相同（只需 `add_edge(START, "triage")`） | 完全相同 |
| **控制权归属** | 交出就不回来 | 交出就不回来 |

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
