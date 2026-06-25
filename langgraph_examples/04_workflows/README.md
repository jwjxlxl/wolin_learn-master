# 04_workflows — LangGraph 工作流编排模式

## 概述

本目录讲解 LangGraph 中的 **工作流（Workflow）编排模式**。

### Workflow vs Agent

| | 工作流 Workflow | 智能体 Agent |
|---|---|---|
| **本质** | 预定义的确定执行路径 | 自主决策的执行体 |
| **决策者** | 程序员在设计时写死 | LLM 运行时自己决定 |
| **核心** | 条件分支、循环、并行 | 思考 → 选工具 → 执行 → 观察 |
| **比喻** | 工厂流水线 | 自由职业者 |

> **何时选 Workflow**：问题和解决方案可预测，需要稳定的执行流程。
> **何时选 Agent**：问题不可预测，需要根据情况灵活选择工具和策略。

### 五大工作流模式

本目录覆盖其中四种（第五种在 05_practical 综合实战中体现）：

1. **提示链** — 顺序执行 + 条件分支
2. **并行化** — 多节点同时执行后聚合
3. **路由** — LLM 判断类型后选择不同路径
4. **评估器-优化器** — 生成 → 评估 → 不满意则重试
5. **协调器-工作者** — 先规划，再分配任务（见 05_practical）

---

## 文件说明

### 1. `prompt_chain.py` — 提示链（Prompt Chain）

**学什么：**
- 工作流的基本结构：StateGraph、节点、边、条件分支
- `with_structured_output(Pydantic模型)` 让 LLM 返回结构化 JSON
- 条件路由函数返回字符串，决定下一步走哪个节点

**流程图：**
```
START → generate_joke → evaluate_joke → (需要改进?)
                                            ├── improve_joke → polish_joke → END
                                            └── polish_joke → END
```

**代码案例：生成笑话并评估质量**
- `generate_joke`：为主题生成一个笑话
- `evaluate_joke`：使用结构化输出评估笑话是否有趣（返回 `needs_improvement: bool` + `reason: str`）
- `should_improve`：路由函数，根据评估结果决定走哪条路
- `improve_joke` / `polish_joke`：改进或润色

**关键代码片段：**
```python
class JokeEvaluation(BaseModel):
    needs_improvement: bool = Field(description="如果笑话需要改进返回 true")
    reason: str = Field(description="评价原因")

evaluator = model.with_structured_output(JokeEvaluation)

def should_improve(state: JokeState):
    if state["needs_improvement"]:
        return "improve_joke"
    else:
        return "polish_joke"
```

**实际应用场景：**
- 📝 **内容审核流水线**：用户发帖 → AI 审核（违规检测）→ 违规则打回修改 → 合规则发布
- 🌐 **翻译工作流**：翻译 → 术语校对 → 风格润色 → 输出
- 📊 **数据清洗管道**：原始数据 → 格式验证 → 异常值处理 → 标准化输出
- 🎨 **图片生成优化**：生成图片 → 质量评估 → 不满意则重绘 → 最终调色

---

### 2. `routing.py` — 路由（Routing）

**学什么：**
- 用 LLM 做智能分类（非硬编码关键词匹配）
- `Literal["technical", "philosophical", "creative"]` 约束 LLM 只能选指定选项
- 路由函数直接返回状态字段值，LangGraph 自动映射到对应节点

**流程图：**
```
START → router → (LLM判断问题类型)
                   ├── technical → 技术专家回答 → END
                   ├── philosophical → 哲学家回答 → END
                   └── creative → 创意专家回答 → END
```

**代码案例：LLM 判断问题类型后选择对应角色回答**
- `router_node`：让 LLM 判断输入问题属于哪个类型（技术/哲学/创意）
- `technical_answer` / `philosophical_answer` / `creative_answer`：不同角色以不同风格回答
- 测试了三种问题：
  - `"Python 中的 GIL 是什么？"` → 技术专家
  - `"人生的意义是什么？"` → 哲学家
  - `"如果月亮是一块巨大的奶酪，世界会怎样？"` → 创意专家

**关键代码片段：**
```python
class RouteDecision(BaseModel):
    question_type: Literal["technical", "philosophical", "creative"] = Field(
        description="问题类型"
    )

router = model.with_structured_output(RouteDecision)

# 路由函数直接返回类型名，LangGraph 自动映射到对应节点
def route_decision(state: RoutingState):
    return state["question_type"]
```

**路由 vs 条件分支的区别：**
| | 条件分支 | 路由 |
|---|---|---|
| **决策者** | 硬编码规则（关键词/数值） | LLM 语义理解 |
| **比喻** | 自动门（检测到人就开） | 前台接待（理解意图后分派） |
| **适合** | 逻辑明确、规则固定的场景 | 需要"理解语义"的场景 |

**实际应用场景：**
- 🏥 **智能客服路由**：用户描述问题 → AI 理解意图 → 转接技术/售后/账单部门
- 📚 **教育平台**：学生提问 → 判断学科（数学/物理/化学）→ 路由到对应专家模型
- 📧 **邮件自动分类**：收到邮件 → 判断类型（投诉/咨询/合作）→ 不同处理流程
- 🎯 **Prompt 路由**：用户输入 → 判断适合哪个 Prompt 模板 → 选择最优配置

---

### 3. `parallelization.py` — 并行化（Parallelization）

**学什么：**
- `Send()` API 的工作原理：动态创建并行分支
- `Annotated[list, operator.add]`：状态字段自动追加（多分支结果汇聚到一个列表）
- 扇出（Fan-out）→ 并行处理 → 扇入（Fan-in）聚合的完整流程
- 对比顺序执行和并行执行的效率差异

**三种边的对比：**
```
Edge         = 老板 → 一个员工（1对1，顺序执行）
Conditional  = 老板 → (判断) → 员工A 或 员工B（1对1，选择路径）
Send()       = 老板 → [员工A, 员工B, 员工C]（1对N，同时开工）
```

#### 示例 1：并行生成笑话

**流程图：**
```
START → planner → Send → generate_joke × 3（并行）
                            ↘ summarizer（汇总）→ END
```

为 3 个主题（程序员、产品经理、设计师）同时生成笑话，汇总后给出总评。

**关键代码片段：**
```python
class JokeState(TypedDict):
    topics: list[str]
    jokes: Annotated[list, operator.add]  # 自动追加所有并行分支的结果
    summary: str

def planner(state: JokeState):
    # 为每个主题创建一个 Send，LangGraph 自动并行执行
    return [Send("generate_joke", {"topics": [topic]}) for topic in state["topics"]]
```

#### 示例 2：多数据源搜索（无需 LLM）

**流程图：**
```
START → dispatcher → Send → search_news
                             → search_wiki
                             → search_forum
                             ↘ aggregator → END
```

模拟同时查询新闻库、百科库、论坛库，聚合所有匹配结果。用 `time.sleep()` 模拟网络延迟，演示 3 个搜索节点并行执行的效果。

**实际应用场景：**
- 🔍 **多源聚合搜索**：一个搜索请求 → 同时查数据库/向量库/搜索引擎 → 合并排序
- 📄 **批量文档处理**：上传 10 个 PDF → 同时做摘要/翻译/关键词提取
- 🗳️ **多候选投票**：对一个问题生成 5 个候选答案 → 独立评分 → 选最高分
- 🌐 **API 聚合**：同时调用天气 API + 新闻 API + 交通 API → 组合成综合回复
- 📊 **数据同步**：更新一个记录 → 同时同步到 MySQL / Redis / 搜索索引

---

### 4. `evaluator_optimizer.py` — 评估器-优化器

**学什么：**
- 迭代改进循环：生成 → 评估 → 不满意则改进 → 再评估 → ...
- 使用结构化输出做自动评估（打分 + 具体反馈）
- 退出保护：`max_iterations` 防止无限循环 + 质量阈值（分数达标即退出）
- 纯逻辑演示（猜数字游戏），无需 LLM 也能理解循环结构

**流程图：**
```
START → generate → evaluate → (达标?) ──是──→ END
                      ↑           │
                      │          否
                      └── optimize ←
```

#### 示例 1：代码优化器

让 AI 写一个 Python 函数，然后自动评估质量并改进。每次评估给出**具体反馈**（不只是"不好"，要说"哪里不好"），generate 节点根据反馈重新生成。

**关键代码片段：**
```python
class CodeEvaluation(BaseModel):
    score: int = Field(description="1-10 分，10 分为满分")
    is_good_enough: bool = Field(description="分数 >= 7 则为 true")
    issues: str = Field(description="具体问题和改进建议")

def should_continue(state: OptimizerState):
    if state.get("is_good_enough"):
        return END  # 质量达标，退出
    if state["iteration"] >= state["max_iterations"]:
        return END  # 达到最大次数，强制退出
    return "optimize"  # 继续循环

# ★ 循环的关键：optimize → generate 形成回环
graph.add_edge("optimize", "generate")
```

**退出条件设计（防止无限循环）：**
1. ✅ 质量达标（score >= 7）→ 正常退出
2. ⚠️ 达到最大迭代次数（max_iterations = 3）→ 强制退出

#### 示例 2：猜数字游戏（纯逻辑，无需 LLM）

用简单规则模拟"评估"和"优化"：AI 不断调整猜测直到猜中目标数字。帮助学生理解循环结构本身。

**实际应用场景：**
- 💻 **代码生成器**：写函数 → 自动评测（正确性/效率/边界）→ 改进 → 再测
- ✍️ **文案优化**：写广告语 → AI 打分评估 → 改进措辞 → 再评估 → 达标为止
- 🌐 **翻译质量提升**：初译 → 质量评估（流畅度/准确度）→ 修正 → 终审
- 📖 **学术论文润色**：初稿 → 评审反馈（逻辑/格式/引用）→ 修改 → 再评审
- 🎨 **设计迭代**：生成方案 → 评估打分 → 调整参数 → 再生成 → 满意为止
- 🤖 **RLHF / 自我改进**：模型输出 → 人类/AI 评估 → 反馈训练 → 再输出

---

## 模式对比总结

| 模式 | 核心特征 | 何时使用 | 代表文件 |
|------|---------|---------|----------|
| **提示链** | 顺序执行 + 条件分支 | 流程固定，可能需要分支调整 | `prompt_chain.py` |
| **并行化** | 同时执行 + 结果聚合 | 多任务独立，需要提高效率 | `parallelization.py` |
| **路由** | LLM 分类 + 多路径选择 | 输入类型不确定，需要智能分派 | `routing.py` |
| **评估器-优化器** | 循环迭代 + 质量门槛 | 需要反复打磨直到达标 | `evaluator_optimizer.py` |
| **协调器-工作者** | 先规划 + 分配任务 | 复杂任务需要分解再执行 | 见 05_practical |

## 组合使用

实际项目中这些模式通常会**组合使用**：

```
用户提问
  ↓
[路由] 判断问题类型
  ↓
[提示链] 按流程处理：检索 → 生成 → 验证
  ↓
[并行化] 遇到多数据需求 → 同时查询多个来源
  ↓
[评估器-优化器] 输出前 → 质量评估 → 不满意则改进
  ↓
返回结果
```

## 运行方式

```bash
# 确保 Ollama 正在运行
ollama serve

# 下载模型（首次运行）
ollama pull qwen3.5:2b

# 运行各个示例
python prompt_chain.py
python routing.py
python parallelization.py
python evaluator_optimizer.py
```

每个文件都可以独立运行，建议按顺序学习：提示链 → 路由 → 并行化 → 评估器-优化器。
