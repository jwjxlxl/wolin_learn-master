# Langfuse 可观测性 Demo

给一个 LangGraph 智能问答 Agent 接入 [Langfuse](https://langfuse.com) 链路追踪，展示"业务代码零改动、只加追踪"的接入方式。

## 这是什么

`qa_agent_with_tracing.py` 是一个标准 LangGraph ReAct Agent（和 `langgraph_examples/05_practical/search_qa_agent.py` 结构一致），
自带三个工具（知识库搜索 / 计算 / 天气查询）。本文件的重点不是"怎么搭图"，而是"怎么给已有的图接上 Langfuse"：

- `langfuse.langchain.CallbackHandler` 挂到 `graph.invoke(..., config={"callbacks": [...]})`，
  LangGraph 内部每一次 LLM 调用、每一次工具调用会自动变成 Langfuse 里的一个 Span，代码逻辑完全不用改。
- 用 `langfuse.start_as_current_observation()` 包一层根 Span，把 trace 顶层的 input/output
  收敛成"用户问题 / 最终回答"，而不是 LangGraph 内部完整的 messages 状态（更符合 Langfuse 官方最佳实践）。
- 通过 `metadata` 里的 `langfuse_session_id` / `langfuse_user_id` / `langfuse_tags`，
  让同一次多轮对话在 Sessions 视图里聚合展示，并支持按标签筛选。

## 运行前准备

1. 安装依赖（项目已有 langgraph / langchain-core / langchain-openai，只需额外装 langfuse）：
   ```bash
   pip install langfuse
   ```
2. 复制 `.env.example` 里的变量到项目根目录 `.env`：
   - `ALIYUN_API_KEY` — 调用云端 Qwen 模型（工具调用场景本地小模型不够稳定，同 `03_agent_loop`/`05_practical` 的选择）
   - `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` — 登录 [cloud.langfuse.com](https://cloud.langfuse.com) 后在项目 `Settings -> API Keys` 获取
   - `LANGFUSE_BASE_URL` — 你的账号所在数据区域（欧盟/美国/日本）或私有部署地址
3. 没有 Langfuse 账号也能跑：`.env` 里不配置 `LANGFUSE_*` 时 Agent 正常运行，只是不上报 trace。

## 运行

```bash
python langfuse_demo/qa_agent_with_tracing.py
```

跑完之后终端会打印 Langfuse 控制台地址，去 **Traces** 视图能看到 3 条 trace（一个问题一条），
去 **Sessions** 视图能看到这 3 条按同一个 `session_id` 聚合成一次完整对话。

## Trace 结构（已实际运行并核对过）

```
[SPAN] qa-agent-weather                    ← 根 span：干净的 input/output（问题 / 回答）
  [AGENT] agent-execution                  ← LangGraph 图的完整执行
    [CHAIN] llm                            ← "llm" 节点
      [CHAIN] route_check                  ← 路由判断（是否需要调用工具）
      [GENERATION] ChatOpenAI              ← 一次真实的 LLM 调用（model / token 用量自动记录）
    [CHAIN] tool                           ← "tool" 节点
      [TOOL] get_weather                   ← 工具调用（自动标注为 TOOL 类型，不是泛泛的 span）
    [CHAIN] llm
      [CHAIN] route_check
      [GENERATION] ChatOpenAI              ← 拿到工具结果后的第二次 LLM 调用
```

已对照 Langfuse 官方 [Best Practices](https://langfuse.com/docs/observability/best-practices) 核对过：
- LLM 调用类型正确标为 `generation`，工具调用正确标为 `tool`，不会挂在 trace 根节点上（正确嵌套）
- 同一次多步骤里的两次 LLM 调用是两个独立的 `generation`，没有被合并成一个
- 每个 `generation` 都带 `model`（`qwen-plus`）和 `usageDetails`（输入/输出/缓存 token）
- 根节点 input/output 是"问题/回答"这种一眼能看懂的内容，不是原始 JSON

## 已知的可选优化

- `qwen-plus` 不在 Langfuse 默认定价表里，所以 `totalCost` 目前是空的。
  想看到自动计费，可以去 Langfuse 控制台 `Settings -> Models` 手动添加一个自定义模型价格。
- `route_check` 会作为一个独立的 `CHAIN` 节点出现在 trace 里 —— 这是 LangGraph 把路由函数也包成
  可追踪 Runnable 的默认行为，保留了它是因为它的 output（`"tool"` / `"__end__"`）正好展示了
  Agent 这一步为什么继续调工具还是直接结束，属于有信息量的节点，不是需要过滤掉的噪音。

## 下一步可以探索

- **Traces 视图**：点开任意一条 trace，看完整 Span 树、每一步耗时、模型输入输出
- **Sessions 视图**：按 `session_id` 看一次多轮对话的全貌
- **按 tags 过滤**：`knowledge` / `calculate` / `weather` 三个标签对应三种问题类型
- 想看更多用法（数据集、评估、Prompt 管理等）可以参考项目里装好的 Langfuse Skill：
  `.claude/skills/langfuse/`（来自 [langfuse/skills](https://github.com/langfuse/skills)）
