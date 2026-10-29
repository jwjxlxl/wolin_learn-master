# 04_workflows — 三种核心工作流

工作流的**图结构预先定义**；节点可以调用 LLM，但模型只能在图允许的路径内做判断。Agent 则可以在运行时自主决定工具调用和后续行动。

前置知识：`01_introduction/simple_graph.py` 讲顺序执行，`02_state_and_branching/conditional_branch.py` 讲规则分支。本章不再单独重复“提示链”；它的基本结构就是 `START → 生成 → 润色 → END`。

| 案例 | 新增的 LangGraph 能力 | 是否需要模型 |
|---|---|---|
| [routing.py](routing.py) | LLM 分类后走预设条件边 | 是 |
| [parallelization.py](parallelization.py) | `Send` 动态扇出、reducer 汇聚 | 否 |
| [evaluator_optimizer.py](evaluator_optimizer.py) | 根据反馈循环，达标或到次数上限后退出 | 是 |

## 1. 智能路由

`START → classify → 售前 / 售后 → END`

`routing.py` 用 `with_structured_output(RouteDecision)` 将问题归为售前或售后，再用 `add_conditional_edges` 进入对应处理节点。与前置章节的规则分支相比，**变化的是分类者，不是图的条件边机制**。分支仅模拟转接，不调用真实客服系统；结构化输出约束格式，不能保证分类永远正确。

## 2. 并行汇聚

`START → Send("lookup", ...) × 3 → summarize → END`

`parallelization.py` 从输入的三个数据源动态创建 `Send`。每个 Worker 收到自己的 `WorkerState`；`Annotated[list[str], operator.add]` 将结果合并，汇总节点在 Worker 完成后运行。数据是本地模拟值，目的在于理解扇出与扇入，**不做速度对比，也不保证结果到达顺序**。代码在展示前排序，便于观察。

注意：`Send` 列表由 `add_conditional_edges(START, dispatch, ...)` 注册的路由函数返回，不是普通节点返回的状态更新。

## 3. 评估与重试

`START → generate → evaluate → 达标则 END；否则回到 generate`

`evaluator_optimizer.py` 让模型生成代码并给出分数与反馈；是否达标由代码依据 `score >= 7` 判断，`max_iterations = 3` 保证循环终止。下一次生成直接读取上次的反馈，不需要空的“优化”节点。

本例**不会运行生成的代码**。同一模型给自己的输出打分只是反馈循环教学，不能代替语法检查、单元测试或人工审查。

## 运行

在仓库根目录运行。并行案例只需 `langgraph`；另外两例需要本地 Ollama `qwen3.5:2b` 支持结构化输出，且要安装项目依赖：

```powershell
python -m pip install langgraph langchain-core langchain-ollama langchain-openai python-dotenv pydantic
ollama pull qwen3.5:2b
python langgraph_examples/04_workflows/parallelization.py
python langgraph_examples/04_workflows/routing.py
python langgraph_examples/04_workflows/evaluator_optimizer.py
```

运行模型示例前确保 Ollama 服务已启动。默认使用本地模型，不需要 API Key；若使用阿里云 Qwen，可在入口传入 `provider="qwen"` 并配置 `ALIYUN_API_KEY`。三个示例都不强制生成 PNG 图，避免额外的图形渲染依赖。
