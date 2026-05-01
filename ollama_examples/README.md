# Ollama Python 大语言模型调用示例

## 📚 目录说明

本目录包含 Ollama Python 库的教学示例代码，适用于人工智能课程教学。

```
ollama_examples/
├── llm_basic_call.py        # 基础调用示例
├── llm_advanced_features.py # 高级功能示例
└── README.md                # 本说明文件
```

## 🔧 环境准备

### 1. 安装 Ollama 服务

访问官网下载安装：https://ollama.ai

### 2. 下载模型

在终端执行以下命令下载模型：

```bash
# 下载 qwen3.5:2b 模型（当前本地已安装）
ollama pull qwen3.5:2b
```

### 3. 安装 Python 库

```bash
pip install ollama
```

### 4. 启动 Ollama 服务

```bash
ollama serve
```

## 📖 文件说明

### llm_basic_call.py - 基础调用示例

| 示例编号 | 功能名称 | 说明 |
|---------|---------|------|
| 示例 1 | basic_chat_completion | 最简单的文本生成调用 |
| 示例 2 | streaming_chat | 流式输出（打字机效果） |
| 示例 3 | multi_turn_conversation | 多轮对话（保持上下文） |
| 示例 4 | chat_with_options | 使用不同的模型参数 |
| 示例 5 | list_local_models | 列出本地可用模型 |
| 示例 6 | chat_with_error_handling | 错误处理示例 |

### llm_advanced_features.py - 高级功能示例

| 示例编号 | 功能名称 | 说明 |
|---------|---------|------|
| 示例 1 | chat_with_system_prompt | 系统提示词设定角色 |
| 示例 2 | function_calling_example | 函数调用（Function Calling） |
| 示例 3 | generate_embeddings | 文本嵌入（Embeddings） |
| 示例 4 | generate_streaming | Generate API 流式输出 |
| 示例 5 | structured_json_output | 结构化 JSON 输出 |
| 示例 6 | show_model_info | 模型信息获取 |

## 🚀 快速开始

```python
import ollama

# 最简单的调用
response = ollama.chat(
    model='qwen2.5:7b',
    messages=[{'role': 'user', 'content': '你好！'}]
)

print(response['message']['content'])
```

## 📝 常用模型推荐

| 模型名称 | 大小 | 特点 | 适用场景 |
|---------|------|------|---------|
| qwen3.5:2b | 4B | 中文支持好，最新一代 | 通用对话、文本生成 |
| llama3.2:3b | 3B | 轻量快速 | 资源受限环境 |
| mistral:7b | 7B | 代码能力强 | 编程辅助 |
| gemma2:9b | 9B | 推理能力强 | 逻辑推理任务 |

## ⚠️ 注意事项

1. 确保 Ollama 服务正在运行
2. 模型名称必须与本地下载的名称一致
3. 大模型需要足够的内存和 CPU/GPU 资源
4. 首次调用可能会较慢（模型加载）

## 🔗 参考链接

- 官方 GitHub：https://github.com/ollama/ollama-python
- Ollama 官网：https://ollama.ai
- 模型库：https://ollama.ai/library
