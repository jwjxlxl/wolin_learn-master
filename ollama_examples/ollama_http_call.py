# =============================================================================
# Ollama HTTP 原生调用示例
# =============================================================================
#  
# 用途：教学演示 - 展示 Ollama 底层 HTTP 调用原理
# =============================================================================

# -----------------------------------------------------------------------------
# 核心知识：Ollama 的调用层次
# -----------------------------------------------------------------------------
#
# 学生常问：Ollama 是通过 HTTP 调用的吗？
#
# 答案：是的！Ollama Python 库底层就是通过 HTTP 请求与 Ollama 服务通信。
#
# 调用层次结构：
# ┌─────────────────────────────────────────────────────────────┐
# │                    用户代码层                                │
# │  方式 1: ollama.chat()  ← 高级封装（方便，但隐藏细节）       │
# │  方式 2: HTTP 请求      ← 底层原生（灵活，可自定义）         │
# └─────────────────────────────────────────────────────────────┘
#                            ↓
# ┌─────────────────────────────────────────────────────────────┐
# │                   Ollama 服务层                              │
# │  ollama serve 运行在 localhost:11434                        │
# └─────────────────────────────────────────────────────────────┘
#                            ↓
# ┌─────────────────────────────────────────────────────────────┐
# │                   推理引擎层                                 │
# │  llama.cpp 加载模型、执行推理                                │
# └─────────────────────────────────────────────────────────────┘
#
# 为什么要学 HTTP 调用？
# 1. 理解底层原理，不依赖特定库
# 2. 可以跨语言调用（任何能发 HTTP 的语言都可以）
# 3. 方便调试和自定义请求
# 4. 理解 API 工作原理
#
# -----------------------------------------------------------------------------

# 设置 UTF-8 编码（Windows 专用）
import sys
import io
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer,
    encoding='utf-8',
    errors='replace',
    line_buffering=True
)

# -----------------------------------------------------------------------------
# 使用 httpx 库发送 HTTP 请求
# -----------------------------------------------------------------------------
# 安装：pip install httpx
#
# 为什么用 httpx 而不是 requests？
# - httpx 支持异步，性能更好
# - 语法更现代，支持上下文管理器
# - 类型提示更完善
# -----------------------------------------------------------------------------
import httpx
import json


# =============================================================================
# 方式 1: 使用 httpx 发送 POST 请求（推荐）
# =============================================================================
def http_chat_with_httpx():
    """
    使用 httpx 库发送 HTTP POST 请求调用 Ollama

    Ollama API 端点：
    - POST http://localhost:11434/api/chat  ← 对话接口（推荐）
    - POST http://localhost:11434/api/generate  ← 生成接口（无对话历史）

    请求格式：
    {
        "model": "qwen3.5:2b",
        "messages": [
            {"role": "user", "content": "你好"}
        ],
        "stream": false
    }

    响应格式：
    {
        "model": "qwen3.5:2b",
        "message": {
            "role": "assistant",
            "content": "你好！我是..."
        },
        "done": true
    }
    """
    print("=" * 60)
    print("方式 1: 使用 httpx 发送 HTTP 请求")
    print("=" * 60)

    # 请求地址和参数
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "qwen3.5:2b",
        "messages": [
            {"role": "user", "content": "你好，请用一句话介绍你自己。"}
        ],
        "stream": False  # 非流式模式，等待完整响应
    }

    # 发送 POST 请求
    # with 语句确保连接正确关闭，避免资源泄漏
    with httpx.Client() as client:
        response = client.post(url, json=payload, timeout=120.0)

        # 检查响应状态
        response.raise_for_status()

        # 解析 JSON 响应
        result = response.json()

        # 提取回复内容
        print(result['message']['content'])

    print()


# =============================================================================
# 方式 2: 使用 urllib（标准库，无需额外安装）
# =============================================================================
def http_chat_with_urllib():
    """
    使用 Python 标准库 urllib 发送 HTTP 请求

    优点：
    - 无需安装额外库
    - 适合理解 HTTP 协议底层

    缺点：
    - 代码相对繁琐
    - 需要手动处理编码和解析
    """
    print("=" * 60)
    print("方式 2: 使用 urllib 标准库")
    print("=" * 60)

    import urllib.request
    import urllib.parse

    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "qwen3.5:2b",
        "messages": [
            {"role": "user", "content": "你好，请用一句话介绍你自己。"}
        ],
        "stream": False
    }

    # 将字典转换为 JSON 字节
    data = json.dumps(payload).encode('utf-8')

    # 创建请求对象
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            'Content-Type': 'application/json'  # 告诉服务器发送的是 JSON 数据
        }
    )

    # 发送请求并读取响应
    with urllib.request.urlopen(req, timeout=120) as response:
        result = json.loads(response.read().decode('utf-8'))
        print(result['message']['content'])

    print()


# =============================================================================
# 方式 3: HTTP 流式调用（打字机效果）
# =============================================================================
def http_streaming_chat():
    """
    流式 HTTP 调用

    流式原理：
    1. 设置 stream: true
    2. 服务器会持续返回多个 JSON 块（chunk）
    3. 每个 chunk 包含一部分生成内容
    4. 客户端逐个接收并显示，形成打字机效果

    流式响应格式（多个 JSON 行）：
    {"model":"qwen3.5:2b","message":{"role":"assistant","content":"你"},"done":false}
    {"model":"qwen3.5:2b","message":{"role":"assistant","content":"好"},"done":false}
    {"model":"qwen3.5:2b","message":{"role":"assistant","content":"！"},"done":false}
    ...
    {"model":"qwen3.5:2b","done":true}  ← 最后一个块标记结束
    """
    print("=" * 60)
    print("方式 3: HTTP 流式调用（打字机效果）")
    print("=" * 60)

    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "qwen3.5:2b",
        "messages": [
            {"role": "user", "content": "请用 500 字介绍人工智能。"}
        ],
        "stream": True  # 启用流式模式
    }

    with httpx.Client() as client:
        # stream=True 启用流式接收
        with client.stream("POST", url, json=payload, timeout=120.0) as response:
            # 逐行读取响应
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    # 流式模式下，content 在 message 字段中
                    content = chunk.get('message', {}).get('content', '')
                    if content:
                        print(content, end='', flush=True)

    print("\n")


# =============================================================================
# 方式 4: 多轮对话（保持上下文）
# =============================================================================
def http_multi_turn_chat():
    """
    HTTP 多轮对话

    关键点：
    - 将历史对话消息按顺序添加到 messages 数组
    - 每次请求都发送完整的对话历史
    - 服务器根据完整历史生成回复

    messages 数组格式：
    [
        {"role": "user", "content": "我叫小明"},
        {"role": "assistant", "content": "你好小明"},
        {"role": "user", "content": "我叫什么？"}
    ]
    """
    print("=" * 60)
    print("方式 4: HTTP 多轮对话")
    print("=" * 60)

    url = "http://localhost:11434/api/chat"

    # 初始化对话历史
    messages = [
        {"role": "user", "content": "我的名字叫小明，今年 10 岁。请记住这个信息。"}
    ]

    # 第一轮对话
    payload = {
        "model": "qwen3.5:2b",
        "messages": messages,
        "stream": False
    }

    with httpx.Client() as client:
        response = client.post(url, json=payload, timeout=120.0)
        result = response.json()
        assistant_reply = result['message']['content']
        print(f"第一轮回复：{assistant_reply}")

    # 将助手回复添加到历史
    messages.append({"role": "assistant", "content": assistant_reply})

    # 第二轮对话
    messages.append({"role": "user", "content": "我叫什么名字？今年几岁？"})

    payload = {
        "model": "qwen3.5:2b",
        "messages": messages,
        "stream": False
    }

    with httpx.Client() as client:
        response = client.post(url, json=payload, timeout=120.0)
        result = response.json()
        print(f"第二轮回复：{result['message']['content']}")

    print()


# =============================================================================
# 方式 5: 列出本地模型（GET 请求）
# =============================================================================
def http_list_models():
    """
    使用 HTTP GET 请求列出本地模型

    Ollama API 端点：
    - GET http://localhost:11434/api/tags  ← 获取模型列表

    响应格式：
    {
        "models": [
            {"name": "qwen3.5:2b", "size": 2497293931, ...},
            {"name": "llama3.2", "size": ...}
        ]
    }
    """
    print("=" * 60)
    print("方式 5: HTTP GET 获取模型列表")
    print("=" * 60)

    url = "http://localhost:11434/api/tags"

    with httpx.Client() as client:
        response = client.get(url, timeout=30.0)
        result = response.json()

        for model in result['models']:
            print(f"模型名称：{model['name']}")
            print(f"模型大小：{model.get('size', '未知')}")
            print("-" * 40)

    print()


# =============================================================================
# 主程序入口
# =============================================================================
if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  Ollama HTTP 原生调用示例")
    print("  说明：通过 HTTP 请求直接调用 Ollama 服务，不依赖 ollama 库")
    print("=" * 70 + "\n")

    print("【运行前检查】")
    print("  1. Ollama 服务已启动：ollama serve")
    print("  2. 模型已下载：ollama pull qwen3.5:2b")
    print("  3. 安装依赖：pip install httpx")
    print()

    print("【HTTP 调用核心知识】")
    print("  - Ollama 服务运行在 http://localhost:11434")
    print("  - /api/chat: 对话接口（支持多轮对话）")
    print("  - /api/generate: 生成接口（单次生成）")
    print("  - /api/tags: 获取模型列表")
    print("  - stream: true 启用流式输出")
    print()

    # 取消注释以运行相应示例
    # http_chat_with_httpx()       # 方式 1: httpx
    # http_chat_with_urllib()    # 方式 2: urllib 标准库
    # http_streaming_chat()      # 方式 3: 流式调用
    http_multi_turn_chat()     # 方式 4: 多轮对话
    # http_list_models()         # 方式 5: 获取模型列表

    print("提示：取消注释相应的函数调用来运行示例。")
