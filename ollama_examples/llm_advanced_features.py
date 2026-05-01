# =============================================================================
# Ollama Python 高级功能示例
# =============================================================================
#  
# 用途：教学演示 - 展示 Ollama 的高级功能
# =============================================================================

# -----------------------------------------------------------------------------
# 重要：设置 UTF-8 编码和输出缓冲
# -----------------------------------------------------------------------------
import sys
import io
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer,
    encoding='utf-8',
    errors='replace',
    line_buffering=True
)
# -----------------------------------------------------------------------------

import ollama
import json

# 创建 Ollama 客户端，禁用环境变量代理（避免 localhost 被代理出去）
client = ollama.Client(host='http://localhost:11434', trust_env=False)


# =============================================================================
# 示例 1: 使用系统提示词（System Prompt）
# =============================================================================
def chat_with_system_prompt():
    """
    系统提示词示例

    system 角色可以设定助手的行为模式、专业领域或回答风格
    这在构建特定角色的机器人时非常有用

    常见的消息角色分类：
    - system: 系统提示词，设定助手的行为模式和专业领域（可选）
    - user: 用户发送的消息或问题
    - assistant: 助手的回复（用于多轮对话历史）
    - tool: 工具调用的返回结果（用于函数调用场景）
    """
    print("=" * 60)
    print("示例 1: 系统提示词 - 设定助手角色")
    print("=" * 60)

    response = client.chat(
        model='qwen3.5:2b',
        messages=[
            # 系统提示词：设定助手的角色和行为
            {
                'role': 'system',
                'content': '你是一位耐心的小学数学老师，擅长用简单易懂的方式解释概念。'
            },
            # 用户问题
            {
                'role': 'user',
                'content': '什么是分数？'
            }
        ]
    )

    # 处理 Qwen 模型的思考模式
    content = response.message.content if response.message.content else response.message.thinking
    print(content if content else '（无输出）')
    print()


# =============================================================================
# 示例 2: 函数调用（Function Calling）
# =============================================================================
def function_calling_example():
    """
    函数调用示例

    大模型可以识别用户意图并返回结构化数据，用于调用外部函数
    这是构建 AI 助手的核心技术
    """
    print("=" * 60)
    print("示例 2: 函数调用 - 天气查询演示")
    print("=" * 60)

    # 定义可用的函数列表
    # 每个函数包含名称、描述和参数定义（JSON Schema 格式）
    tools = [{
        'type': 'function',
        'function': {
            'name': 'get_weather',  # 函数名称
            'description': '获取指定城市的天气信息',  # 函数描述
            'parameters': {  # 参数定义（JSON Schema）
                'type': 'object',
                'properties': {
                    'city': {
                        'type': 'string',
                        'description': '城市名称，如：北京、上海'
                    },
                    'date': {
                        'type': 'string',
                        'description': '日期，格式：YYYY-MM-DD'
                    }
                },
                'required': ['city']  # 必填参数
            }
        }
    }]

    # 用户请求
    response = client.chat(
        model='qwen3.5:2b',
        messages=[
            {'role': 'user', 'content': '帮我查一下北京明天的天气'}
        ],
        tools=tools  # 传入可用工具列表
    )

    # 检查模型是否返回了工具调用
    message = response.message

    if message.tool_calls:
        print("模型决定调用以下函数：")
        for tool_call in message.tool_calls:
            func = tool_call.function
            print(f"  函数名：{func.name}")
            print(f"  参数：{func.arguments}")

            # 实际应用中，这里会调用真实的 API
            # weather = get_weather(**func['arguments'])
    else:
        print("普通回复：", message.content if message.content else message.thinking)

    print()


# =============================================================================
# 示例 3: 文本嵌入（Embeddings）
# =============================================================================
def generate_embeddings():
    """
    文本嵌入生成示例

    Embedding 将文本转换为向量表示，用于：
    - 语义搜索
    - 文本相似度计算
    - 知识库检索（RAG）

    注意：需要使用支持 embeddings 的模型，如：
    - nomic-embed-text
    - mxbai-embed-large
    - all-minilm
    """
    print("=" * 60)
    print("示例 3: 文本嵌入（Embeddings）")
    print("=" * 60)

    try:
        # 生成文本的向量表示
        # 注意：qwen3.5:2b 不支持 embeddings，需使用专门模型
        # 安装方法：ollama pull nomic-embed-text
        response = client.embeddings(
            model='nomic-embed-text',  # 或使用 mxbai-embed-large
            prompt='人工智能是计算机科学的一个分支'
        )

        # embedding 是一个高维向量（通常是几百到几千维）
        embedding = response.embedding

        print(f"生成的向量维度：{len(embedding)}")
        print(f"向量前 10 个值：{embedding[:10]}")

    except Exception as e:
        print(f"无法生成 embeddings: {e}")
        print("提示：请先安装支持 embeddings 的模型，例如：")
        print("  ollama pull nomic-embed-text")
        print("  ollama pull mxbai-embed-large")

    print()


# =============================================================================
# 示例 4: 生成并流式处理
# =============================================================================
def generate_streaming():
    """
    使用 generate API 进行流式输出

    generate 是更底层的 API，直接生成文本而非对话格式
    """
    print("=" * 60)
    print("示例 4: Generate API 流式输出")
    print("=" * 60)

    stream = client.generate(
        model='qwen3.5:2b',
        prompt='写一首关于春天的五言绝句：',
        stream=True
    )

    for chunk in stream:
        print(chunk['response'], end='', flush=True)

    print("\n")


# =============================================================================
# 示例 5: 结构化 JSON 输出
# =============================================================================
def structured_json_output():
    """
    结构化输出示例

    通过格式化提示词，可以让模型返回 JSON 格式的数据
    便于程序解析和处理
    """
    print("=" * 60)
    print("示例 5: 结构化 JSON 输出")
    print("=" * 60)

    response = client.chat(
        model='qwen3.5:2b',
        messages=[
            {
                'role': 'user',
                'content': '''请分析以下句子，并以 JSON 格式返回：
                句子："小明喜欢打篮球和游泳。"

                请返回以下字段：
                - subject: 主语
                - hobbies: 爱好列表（数组）

                只返回 JSON，不要其他文字。'''
            }
        ]
    )

    # 处理 Qwen 模型的思考模式
    content = response.message.content if response.message.content else response.message.thinking
    print("模型输出：", content)

    # 尝试解析 JSON（实际应用中建议添加错误处理）
    try:
        # 清理可能的 markdown 标记
        content = content.replace('```json', '').replace('```', '').strip()
        data = json.loads(content)
        print(f"解析后的主语：{data.get('subject')}")
        print(f"解析后的爱好：{data.get('hobbies')}")
    except json.JSONDecodeError:
        print("JSON 解析失败，模型可能返回了额外文本")

    print()


# =============================================================================
# 示例 6: 模型信息获取
# =============================================================================
def show_model_info():
    """
    显示模型详细信息
    """
    print("=" * 60)
    print("示例 6: 模型信息")
    print("=" * 60)

    # 获取指定模型的详细信息
    info = client.show('qwen3.5:2b')

    print(f"模型：qwen3.5:2b")
    print(f"参数数量：{info.modelinfo.get('general.parameter_count', '未知')}")
    print(f"量化等级：{info.details.quantization_level}")
    print()


# =============================================================================
# 主程序入口
# =============================================================================
if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  Ollama Python 高级功能示例")
    print("  说明：以下示例演示了 Ollama 库的高级用法")
    print("=" * 70 + "\n")

    print("【运行前检查】")
    print("  1. 已安装 Ollama 服务")
    print("  2. 已下载模型：ollama pull qwen3.5:2b（当前本地模型）")
    print("  3. Ollama 服务正在运行")
    print()

    # 取消注释以运行相应示例
    # chat_with_system_prompt()      # 示例 1: 系统提示词
    # function_calling_example()     # 示例 2: 函数调用
    generate_embeddings()          # 示例 3: 文本嵌入
    # generate_streaming()           # 示例 4: Generate 流式
    # structured_json_output()       # 示例 5: JSON 输出
    # show_model_info()              # 示例 6: 模型信息

    print("提示：取消注释相应的函数调用来运行示例。")
