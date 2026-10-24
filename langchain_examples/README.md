# LangChain 核心概念速览

LangChain 是连接模型、提示词、文档和工具的 Python 组件库，不是开发大模型应用的必选项。简单调用模型时，直接使用服务商 SDK 通常更轻；需要组合多个组件时，再考虑 LangChain。

## 先理解什么

- **Chat Model**：统一的模型调用接口；`invoke()` 一次返回结果，`stream()` 逐块返回。
- **Messages**：System 规定角色，Human 提供问题，AI 保存回复。模型本身不会跨请求记住对话，必须再次传入历史。
- **Prompt Template**：把变量填入固定的提示词格式。
- **Output Parser**：把模型回复转换成字符串或结构化数据；解析不等于保证模型一定输出合法格式。
- **LCEL**：用 `|` 将提示词、模型、解析器连成流水线。
- **Retriever**：先找相关资料，再交给模型回答；本课程只用关键词演示流程，向量检索见 `rag_examples/`。
- **Tool / Agent**：工具是可调用的函数；Agent 由模型决定何时调用工具，再用工具结果回答。执行效果取决于模型的工具调用能力。

## 最短学习路径

| 顺序 | 文件 | 只看一个重点 | 是否需要 Ollama |
|---|---|---|---|
| 1 | [first_chain.py](01_introduction/first_chain.py) | 调用本地模型 | 是 |
| 2 | [llm_basic.py](02_llm_call/llm_basic.py) | 消息与流式输出 | 是 |
| 3 | [prompt_template.py](03_prompt/prompt_template.py) | 提示词模板 | 否 |
| 4 | [json_parser.py](04_output_parser/json_parser.py) | JSON 输出解析 | 否 |
| 5 | [conversation_memory.py](05_memory/conversation_memory.py) | 显式传递对话历史 | 是 |
| 6 | [simple_chain.py](06_chains/simple_chain.py) | LCEL 三段式 | 是 |
| 7 | [retrieval_demo.py](07_retrieval/retrieval_demo.py) | 加载、切片、关键词检索 | 否 |
| 8 | [agent.py](09_agent/agent.py) | 一个模拟订单工具与 Agent | 是 |

本课程有意不展开旧式 Memory 类、复杂链、Agent 中间件及生产级 RAG。这些主题分别可继续阅读 `langgraph_examples/` 与 `rag_examples/`。

## 运行

在仓库根目录执行：

```powershell
python -m pip install -r langchain_examples/requirements.txt
python langchain_examples/03_prompt/prompt_template.py
python langchain_examples/04_output_parser/json_parser.py
python langchain_examples/07_retrieval/retrieval_demo.py
```

需要模型的示例先安装并启动 Ollama，下载 `qwen3.5:2b`，再运行对应脚本：

```powershell
ollama pull qwen3.5:2b
python langchain_examples/01_introduction/first_chain.py
```

`09_agent/agent.py` 还要求本地模型支持工具调用。模拟订单状态是固定数据，不会连接真实企业系统。全部示例均不需要 API Key。

> 示例使用 LangChain 1.x API。课程以 `.py` 为准；旧测试报告与重复的学习路线已移除。
