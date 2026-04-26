# =============================================================================
# vLLM Python 基本调用示例
# =============================================================================
#  
# 用途：教学演示 - 展示如何使用 vLLM 调用部署的大语言模型
#
# vLLM vs Ollama 核心区别：
# -------------------------
# 1. 定位不同：
#    - Ollama: 一键式模型工具，适合本地快速部署和测试
#    - vLLM: 高性能推理引擎，适合生产环境高并发场景
#
# 2. API 兼容性：
#    - Ollama: 自有 API 格式
#    - vLLM: 兼容 OpenAI API 格式，可直接使用 openai 库调用
#
# 3. 启动方式：
#    - Ollama: ollama serve（后台自动运行）
#    - vLLM: 需要手动启动服务，指定模型和端口
#
# 4. 性能：
#    - Ollama: 适合单人使用和开发测试
#    - vLLM: 支持高并发，吞吐量更高，适合生产部署
#
# 5. 模型格式：
#    - Ollama: 使用 GGUF 量化格式，节省显存
#    - vLLM: 使用原始 HuggingFace 格式，精度更高
# =============================================================================

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 安装依赖
# -----------------------------------------------------------------------------
# pip install openai
#
# 注意：vLLM 兼容 OpenAI API，所以使用 openai 库调用
# -----------------------------------------------------------------------------

from openai import OpenAI

# -----------------------------------------------------------------------------
# 客户端配置
# -----------------------------------------------------------------------------
# vLLM 启动命令示例（在 WSL 中执行）：
#   python -m vllm.entrypoints.openai_api_server \
#       --model Qwen/Qwen2.5-7B-Instruct \
#       --host 0.0.0.0 \
#       --port 8000
#
# Ollama 启动命令对比：
#   ollama serve  （默认端口 11434）
# -----------------------------------------------------------------------------

# 创建 OpenAI 兼容客户端，指向 vLLM 服务地址
client = OpenAI(
    api_key="not-needed",           # vLLM 不需要 API key，但必须传入任意值
    base_url="http://localhost:8000/v1"  # vLLM 默认端口 8000，Ollama 是 11434
)


def vllm_completion(prompt, model='Qwen/Qwen2.5-1.5B-Instruct', stream=False, max_tokens=1024, temperature=0.7, top_p=0.9):
    """
    封装 vLLM chat completion 调用（适用于有 chat template 的指令模型）

    与 Ollama 的区别：
    - Ollama: ollama.chat(model='xxx', messages=[...])
    - vLLM:   client.chat.completions.create(...)

    Args:
        prompt: 文本提示，如 '你好'
        model: 模型名称（需与 vLLM 启动时指定的一致）
        stream: 是否使用流式输出
        max_tokens: 最大生成 token 数
        temperature: 温度，控制输出随机性
        top_p: 核采样参数

    Returns:
        非流式模式返回完整响应
        流式模式返回生成器
    """
    kwargs = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'stream': stream,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'top_p': top_p
    }

    if stream:
        # 流式模式：返回生成器
        return client.chat.completions.create(**kwargs)
    else:
        # 非流式模式：返回完整响应
        return client.chat.completions.create(**kwargs)


# =============================================================================
# 示例 1: 最简单的文本生成调用
# =============================================================================
def basic_chat_completion():
    """
    基础对话调用示例

    这是最常用的调用方式
    """
    print("=" * 60)
    print("示例 1: 基础对话调用")
    print("=" * 60)

    response = vllm_completion(
        model='Qwen/Qwen2.5-1.5B-Instruct',
        prompt='你好，请简单介绍一下自己。'
    )

    # 提取回复内容
    # vLLM chat completion 返回格式：response.choices[0].message.content
    print(response.choices[0].message.content)
    print()


# =============================================================================
# 示例 2: 流式输出（Streaming）
# =============================================================================
def streaming_completion():
    """
    流式输出示例

    流式输出可以逐步显示模型的回复，提升用户体验
    类似于打字机效果
    """
    print("=" * 60)
    print("示例 2: 流式输出（打字机效果）")
    print("=" * 60)

    # stream=True 启用流式输出
    stream = vllm_completion(
        model='Qwen/Qwen2.5-1.5B-Instruct',
        prompt='请用 100 字介绍人工智能。',
        stream=True
    )

    # 遍历流式响应
    # chat completion 流式格式：chunk.choices[0].delta.content
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end='', flush=True)

    print("\n")


# =============================================================================
# 示例 3: 多轮对话（保持上下文）
# =============================================================================
def multi_turn_conversation():
    """
    多轮对话示例

    completion 接口不支持真正的多轮对话（因为没有 chat template）
    这里演示如何将多轮对话拼接成单个 prompt
    """
    print("=" * 60)
    print("示例 3: 多轮对话（拼接 prompt 模拟）")
    print("=" * 60)

    # 手动拼接对话历史
    prompt = """用户：我的名字叫小明，今年 10 岁。请记住这个信息。
助手：好的，我已经记住了。你叫小明，今年 10 岁。
用户：我叫什么名字？今年几岁？
助手："""

    response = vllm_completion(
        model='Qwen/Qwen2.5-1.5B-Instruct',
        prompt=prompt,
        max_tokens=100
    )

    print(f"助手回复：{response.choices[0].message.content}")
    print()


# =============================================================================
# 示例 4: 使用不同的模型参数
# =============================================================================
def completion_with_options():
    """
    带参数配置的 completion 示例

    常见参数说明：
    - temperature: 控制输出随机性（0-1，越高越有创意）
    - max_tokens: 限制最大生成长度
    - top_p: 核采样参数
    - presence_penalty: 存在惩罚，降低重复话题
    """
    print("=" * 60)
    print("示例 4: 带参数的 completion 调用")
    print("=" * 60)

    response = vllm_completion(
        model='Qwen/Qwen2.5-1.5B-Instruct',
        prompt='请用一句话解释什么是机器学习。',
        max_tokens=200,
        temperature=0.7,
        top_p=0.9
    )

    print(response.choices[0].text)
    print()


# =============================================================================
# 示例 5: 列出可用模型
# =============================================================================
def list_available_models():
    """
    列出 vLLM 服务可用的模型

    Ollama 对比：ollama.list() 或 client.list()
    """
    print("=" * 60)
    print("示例 5: 列出可用模型")
    print("=" * 60)

    # vLLM 使用 models.list()
    # Ollama 使用 client.list()
    models = client.models.list()

    for model in models.data:
        print(f"模型 ID: {model.id}")
        print(f"创建时间：{model.created}")
        print(f"拥有者：{model.owned_by}")
        print("-" * 40)

    print()


# =============================================================================
# 示例 6: 错误处理
# =============================================================================
def completion_with_error_handling():
    """
    带错误处理的 completion 示例

    常见错误场景：
    - vLLM 服务未启动
    - 模型名称不匹配
    - 网络连接问题
    """
    print("=" * 60)
    print("示例 6: 带错误处理的调用")
    print("=" * 60)

    try:
        response = vllm_completion(
            model='Qwen/Qwen2.5-1.5B-Instruct',
            prompt='测试消息'
        )
        print(response.choices[0].text)

    except Exception as e:
        print(f"请求失败：{e}")
        print("\n可能的原因：")
        print("  1. vLLM 服务未启动")
        print("  2. 模型名称不匹配")
        print("  3. 端口或地址错误")
        print("\n请检查 WSL 中的 vLLM 服务是否正常运行")

    print()


# =============================================================================
# 主程序入口
# =============================================================================
if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  vLLM Python 基本调用示例")
    print("  说明：以下示例演示了 vLLM 库的基本用法")
    print("=" * 70 + "\n")

    print("【运行前检查】")
    print("  1. WSL 中已安装 vLLM: pip install vllm")
    print("  2. WSL 中已启动 vLLM 服务（见下方启动命令）")
    print("  3. 模型已下载且名称与配置一致")
    print()

    print("【WSL 中启动 vLLM 服务】")
    print("  python -m vllm.entrypoints.openai_api_server \\")
    print("      --model Qwen/Qwen2.5-1.5B-Instruct \\")
    print("      --host 0.0.0.0 \\")
    print("      --port 8000")
    print()

    # 取消注释以运行相应示例
    basic_chat_completion()        # 示例 1: 基础对话
    streaming_completion()         # 示例 2: 流式输出
    # multi_turn_conversation()      # 示例 3: 多轮对话
    # completion_with_options()      # 示例 4: 带参数调用
    # list_available_models()        # 示例 5: 列出模型
    # completion_with_error_handling()  # 示例 6: 错误处理

    print("提示：取消注释相应的函数调用来运行示例。")
