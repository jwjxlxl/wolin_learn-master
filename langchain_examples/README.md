# LangChain 基础教程

> 为 Python 基础薄弱的 AI 初学者设计的渐进式教程

## 课程目标

完成本教程后，你将能够：

- ✅ 理解 LangChain 的核心概念和用途
- ✅ 使用 LangChain 调用 LLM（本地 Ollama 或云端 API）
- ✅ 创建和管理 Prompt 模板
- ✅ 解析 LLM 输出为结构化数据
- ✅ 构建简单的对话应用
- ✅ 理解 RAG（检索增强生成）的基本原理

## 快速开始

### 1. 安装依赖

**Windows 用户（推荐）**：
```bash
# 运行一键安装脚本
install_deps.bat
```

**手动安装**：
```bash
pip install -r requirements.txt
```

**使用国内镜像（可选，加速下载）**：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 配置环境变量

```bash
# 复制 .env.example 为 .env
cp .env.example .env

# 编辑 .env 文件，填入你的 API Key
# 如果只用本地 Ollama 模型，不需要配置 API Key
```

### 3. 运行示例

```bash
# 每个示例文件都可以独立运行
python 01_introduction/first_chain.py
```

## 课程目录

### 📚 第一部分：LangChain 简介

| 文件 | 说明 | 难度 |
|------|------|------|
| [what_is_langchain.py](01_introduction/what_is_langchain.py) | 什么是 LangChain | ⭐ |
| [first_chain.py](01_introduction/first_chain.py) | 第一个 Chain | ⭐ |

### 📚 第二部分：LLM 调用

| 文件 | 说明 | 难度 |
|------|------|------|
| [llm_basic.py](02_llm_call/llm_basic.py) | 基础 LLM 调用 | ⭐⭐ |
| [chat_model.py](02_llm_call/chat_model.py) | Chat Model 对话 | ⭐⭐ |
| [streaming_output.py](02_llm_call/streaming_output.py) | 流式输出 | ⭐⭐ |

### 📚 第三部分：Prompt 工程

| 文件 | 说明 | 难度 |
|------|------|------|
| [prompt_template.py](03_prompt/prompt_template.py) | Prompt 模板基础 | ⭐⭐ |
| [few_shot_prompt.py](03_prompt/few_shot_prompt.py) | Few-Shot 示例 | ⭐⭐⭐ |
| [pipeline_prompt.py](03_prompt/pipeline_prompt.py) | Pipeline 组合 | ⭐⭐⭐ |

### 📚 第四部分：输出解析

| 文件 | 说明 | 难度 |
|------|------|------|
| [string_parser.py](04_output_parser/string_parser.py) | 字符串解析 | ⭐⭐ |
| [json_parser.py](04_output_parser/json_parser.py) | JSON 结构化输出 | ⭐⭐⭐ |
| [pydantic_parser.py](04_output_parser/pydantic_parser.py) | Pydantic 强类型解析 | ⭐⭐⭐ |

### 📚 第五部分：记忆

| 文件 | 说明 | 难度 |
|------|------|------|
| [conversation_memory.py](05_memory/conversation_memory.py) | 对话记忆 | ⭐⭐⭐ |
| [buffer_memory.py](05_memory/buffer_memory.py) | 缓冲区记忆 | ⭐⭐⭐ |

### 📚 第六部分：链

| 文件 | 说明 | 难度 |
|------|------|------|
| [simple_chain.py](06_chains/simple_chain.py) | 简单链 | ⭐⭐⭐ |
| [sequential_chain.py](06_chains/sequential_chain.py) | 顺序链 | ⭐⭐⭐⭐ |
| [router_chain.py](06_chains/router_chain.py) | 路由链 | ⭐⭐⭐⭐ |

### 📚 第七部分：RAG 检索

| 文件 | 说明 | 难度 |
|------|------|------|
| [document_loader.py](07_retrieval/document_loader.py) | 文档加载 | ⭐⭐⭐ |
| [vector_store.py](07_retrieval/vector_store.py) | 向量存储 | ⭐⭐⭐⭐ |
| [rag_basic.py](07_retrieval/rag_basic.py) | RAG 基础示例 | ⭐⭐⭐⭐ |

### 📚 第八部分：实战项目

| 文件 | 说明 | 难度 |
|------|------|------|
| [qna_bot.py](08_project/qna_bot.py) | 问答机器人 | ⭐⭐⭐⭐ |
| [research_assistant.py](08_project/research_assistant.py) | 研究助手 | ⭐⭐⭐⭐⭐ |

## 学习建议

### 🎯 零基础学员
1. 按顺序学习，不要跳过前面的章节
2. 每个示例都要亲手运行一遍
3. 遇到错误先阅读错误信息，再查看示例中的错误处理部分

### 🎯 有基础学员
1. 可以直接跳转到感兴趣的部分
2. 重点关注代码组织和最佳实践
3. 尝试修改示例代码，添加自己的功能

## 核心概念速查

### LangChain 是什么？
> **LangChain = AI 应用的乐高积木**
>
> 它提供标准化的"积木块"，让你可以像搭乐高一样构建 AI 应用

### LLM vs Chat Model
- **LLM**: 文本补全（像续写句子）
- **Chat Model**: 对话交互（像聊天）

### Prompt Template
> **Prompt Template = 填空题模板**
>
> `"请用{语言}解释{概念}"`

### Memory
> **Memory = 外部记事本**
>
> LLM 本身没有记忆，Memory 帮它记住历史对话

### RAG (检索增强生成)
> **RAG = 先查资料，再回答问题**
>
> 流程：用户问题 → 检索相关文档 → 拼接问题 + 文档 → LLM 回答

## 常见问题

### Q: 我需要付费 API 才能学习吗？
A: 不需要！使用本地 Ollama 模型可以免费学习全部功能。

### Q: 运行示例需要什么配置？
A: 每个文件开头都有"运行前检查"清单，按照提示准备即可。

### Q: 代码报错怎么办？
A: 每个示例都包含错误处理代码，会显示友好的错误信息和建议。

## 参考资料

- [LangChain 官方文档](https://python.langchain.com/docs/introduction/)
- [LangChain 中文文档](https://langchain-doc.cn/)
- [Ollama 官网](https://ollama.ai)
