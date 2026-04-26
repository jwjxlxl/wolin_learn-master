# =============================================================================
# Ollama Python 大语言模型基本调用示例
# =============================================================================
#  
# 用途：教学演示 - 展示如何使用 Ollama Python 库调用本地大语言模型
# 文档：https://github.com/ollama/ollama-python
# =============================================================================

# -----------------------------------------------------------------------------
# 重要：设置 UTF-8 编码和输出缓冲
# -----------------------------------------------------------------------------
# 这段代码解决两个问题：
#
# 1. UTF-8 编码：Windows 命令行默认使用 GBK 编码，无法显示中文，
#    这会导致包含中文的 print() 语句报错。
#
# 2. 输出缓冲（line_buffering=True）：Python 默认会缓冲输出内容，
#    导致所有 print() 语句的结果先存储在内存中，直到程序结束才一次性显示。
#    这会让程序看起来"卡住"，然后突然打印所有内容。
#    line_buffering=True 确保每行 print() 后立即显示，学生能看到程序执行进度。
#
# 简单理解：
#   - 没有 line_buffering=True：像 batching，攒一批再显示
#   - 有 line_buffering=True：像实时流，说一句显示一句
# -----------------------------------------------------------------------------
import sys
import io
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer,           # 获取底层二进制缓冲区
    encoding='utf-8',            # 使用 UTF-8 编码（支持中文）
    errors='replace',            # 遇到无法编码的字符时替换而非报错
    line_buffering=True          # 行缓冲：遇到换行符就立即输出（关键！）
)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 1. 安装依赖
# -----------------------------------------------------------------------------
# 在终端中执行以下命令安装 ollama 库：
#   pip install ollama
#
# 前置要求：
#   - 已安装 Ollama 服务 (https://ollama.ai)
#   - 已下载模型，例如：ollama pull qwen3:4b
# -----------------------------------------------------------------------------

import ollama

# 创建 Ollama 客户端，禁用环境变量代理（避免 localhost 被代理出去）
client = ollama.Client(host='http://localhost:11434', trust_env=False)

def ollama_chat(messages, model='qwen3:4b', stream=False, options=None):
    """
    封装 Ollama chat API 调用

    Args:
        messages: 消息列表，如 [{'role': 'user', 'content': '你好'}]
        model: 模型名称
        stream: 是否使用流式输出
        options: 模型参数配置

    Returns:
        非流式模式返回完整响应字典
        流式模式返回生成器，逐个返回响应块
    """
    kwargs = {
        'model': model,
        'messages': messages,
        'stream': stream
    }
    if options:
        kwargs['options'] = options

    # ollama.chat() 直接返回响应对象
    # stream=True 时返回一个可迭代的生成器
    return client.chat(**kwargs)


# =============================================================================
# 示例 1: 最简单的文本生成调用
# =============================================================================
def basic_chat_completion():
    """
    基础对话调用示例

    这是最常用的调用方式，适用于单次对话场景
    """
    print("=" * 60)
    print("示例 1: 基础对话调用")
    print("=" * 60)

    # 调用聊天接口
    # model: 指定使用的模型名称（需确保本地已下载该模型）
    # messages: 消息列表，每条消息包含 role 和 content
    response = ollama_chat(
        model='qwen3:4b',  # 模型名称，可替换为 llama3.2、mistral 等
        messages=[
            {
                'role': 'user',      # 消息角色：'user'（用户）或 'assistant'（助手）
                'content': '你好，请简单介绍一下自己。'  # 消息内容
            }
        ]
    )

    # 提取并打印回复内容
    # response 是一个字典，包含 'message'、'done' 等字段
    # response['message']['content'] 存储助手的回复文本
    print(response['message']['content'])
    print()


# =============================================================================
# 示例 2: 流式输出（Streaming）
# =============================================================================
def streaming_chat():
    """
    流式对话示例

    流式输出可以逐步显示模型的回复，提升用户体验
    类似于打字机效果，用户无需等待完整回复生成
    """
    print("=" * 60)
    print("示例 2: 流式对话（打字机效果）")
    print("=" * 60)

    # stream=True 启用流式输出
    # ollama.chat() 返回一个可迭代的生成器，逐个获取响应块
    stream = ollama_chat(
        model='qwen3:4b',
        messages=[{'role': 'user', 'content': '请用 1000 字介绍人工智能。'}],
        stream=True  # 关键参数：启用流式模式
    )

    # 遍历生成器，逐个打印响应块
    # end='' 确保不换行，flush=True 确保立即显示
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)

    print("\n")  # 流式输出结束后换行


# =============================================================================
# 示例 3: 多轮对话（保持上下文）
# =============================================================================
def multi_turn_conversation():
    """
    多轮对话示例

    通过将历史对话消息添加到 messages 列表，实现上下文记忆
    这是构建对话机器人的核心方法
    """
    print("=" * 60)
    print("示例 3: 多轮对话（保持上下文）")
    print("=" * 60)

    # 初始化消息历史列表
    # 这个列表会保存整个对话历史
    messages = [{
        'role': 'user',
        'content': '我的名字叫小明，今年 10 岁。请记住这个信息。'
    }]

    # 第一轮对话

    response = ollama_chat(model='qwen3:4b', messages=messages)
    assistant_reply = response['message']['content']
    print(f"助手回复：{assistant_reply}")

    # 将助手回复也添加到历史记录中
    messages.append({'role': 'assistant', 'content': assistant_reply})

    # 第二轮对话 - 测试模型是否记住之前的信息
    messages.append({
        'role': 'user',
        'content': '我叫什么名字？今年几岁？'
    })

    response = ollama_chat(model='qwen3:4b', messages=messages)
    print(f"助手回复：{response['message']['content']}")
    print()


# =============================================================================
# 示例 4: 使用不同的模型参数
# =============================================================================
def chat_with_options():
    """
    带参数配置的对话示例

    可以通过 options 参数调整模型行为：
    - temperature: 控制输出随机性（0-1，越高越有创意）
    - num_predict: 限制最大生成长度
    - top_p: 核采样参数
    """
    print("=" * 60)
    print("示例 4: 带参数的对话调用")
    print("=" * 60)

    response = ollama_chat(
        model='qwen3:4b',
        messages=[
            {
                'role': 'user',
                'content': '请用一句话解释什么是机器学习。'
            }
        ],
        # options 字典可以配置各种模型参数
        options={
            'temperature': 0.7,      # 温度：0.7 表示平衡创意和稳定性
            'num_predict': 200,      # 最多生成 200 个 token
            'top_p': 0.9,            # 核采样：从累积概率 90% 的词中采样
        }
    )

    # 注意：某些模型（如 Qwen）有思考模式，content 可能为空
    # 优先使用 message.content，如果为空则尝试思考内容
    content = response.message.content if response.message.content else response.message.thinking
    print(content if content else '（无输出）')
    print()


# =============================================================================
# 示例 5: 列出本地可用模型
# =============================================================================
def list_local_models():
    """
    列出本地已下载的模型

    在调用前检查可用模型是个好习惯
    """
    print("=" * 60)
    print("示例 5: 列出本地可用模型")
    print("=" * 60)

    # 使用 client.list() 获取本地所有模型列表
    # 返回的是 ListResponse 对象，models 属性是模型列表
    models = client.list()

    # 遍历并打印每个模型的信息
    # 每个模型是对象，使用 .model 获取名称，.size 获取大小
    for model in models.models:
        print(f"模型名称：{model.model}")
        print(f"模型大小：{model.size}")
        print("-" * 40)

    print()


# =============================================================================
# 示例 6: 错误处理（推荐的生产环境写法）
# =============================================================================
def chat_with_error_handling():
    """
    带错误处理的对话示例

    在实际应用中，应该始终处理可能的异常情况：
    - 模型不存在
    - Ollama 服务未启动
    - 网络连接问题
    """
    print("=" * 60)
    print("示例 6: 带错误处理的调用")
    print("=" * 60)

    try:
        response = ollama_chat(
            model='qwen3:4b',
            messages=[{'role': 'user', 'content': '测试消息'}]
        )
        print(response['message']['content'])

    except client.ResponseError as e:
        # HTTP 响应错误
        print(f"HTTP 响应错误：{e}")
        if e.status_code == 404:
            print("错误：模型不存在，请先使用 'ollama pull <模型名>' 下载模型")

    except client.ConnectionError as e:
        # 连接错误
        print(f"连接错误：{e}")
        print("请检查 Ollama 服务是否已启动（运行 'ollama serve'）")

    except Exception as e:
        # 其他未知错误
        print(f"未知错误：{e}")

    print()


# =============================================================================
# 主程序入口
# =============================================================================
if __name__ == '__main__':
    # 程序入口说明
    print("\n" + "=" * 70)
    print("  Ollama Python 大语言模型调用示例")
    print("  说明：以下示例演示了 Ollama 库的各种基本用法")
    print("=" * 70 + "\n")

    # 提示：运行前请确保
    print("【运行前检查】")
    print("  1. 已安装 Ollama 服务")
    print("  2. 已下载模型：ollama pull qwen3:4b（当前本地模型）")
    print("  3. Ollama 服务正在运行")
    print()

    # 取消注释以运行相应示例
    # 注意：运行这些示例需要本地有对应的模型

    # basic_chat_completion()        # 示例 1: 基础对话
    # streaming_chat()               # 示例 2: 流式输出
    # multi_turn_conversation()      # 示例 3: 多轮对话
    # chat_with_options()            # 示例 4: 带参数调用
    # list_local_models()            # 示例 5: 列出模型
    chat_with_error_handling()     # 示例 6: 错误处理

    print("提示：取消注释相应的函数调用来运行示例。")
