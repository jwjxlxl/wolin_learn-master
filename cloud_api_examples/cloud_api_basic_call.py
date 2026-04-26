# =============================================================================
# 云 API 基本调用示例 - Qwen 和 DeepSeek
# =============================================================================
#  
# 用途：教学演示 - 展示如何使用 API Key 调用云端大模型服务
# =============================================================================

# -----------------------------------------------------------------------------
# 核心知识：云 API 调用 vs 本地部署
# -----------------------------------------------------------------------------
#
# 本地部署 (Ollama/vLLM)：
#   - 优点：数据私密、无网络费用、可离线使用
#   - 缺点：需要 GPU 硬件、显存限制、模型质量受限
#
# 云 API 调用 (Qwen/DeepSeek)：
#   - 优点：无需 GPU、最新最强模型、按使用付费
#   - 缺点：需要网络、数据上传到云端、持续使用成本高
#
# 适用场景：
#   - 本地部署：开发测试、敏感数据、高频使用
#   - 云 API：生产环境、大型模型、低频使用
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
# 安装依赖
# -----------------------------------------------------------------------------
# pip install openai python-dotenv
#
# 说明：
# - openai: Qwen 和 DeepSeek 都兼容 OpenAI API 格式
# - python-dotenv: 从.env 文件加载环境变量
# -----------------------------------------------------------------------------

import os
from openai import OpenAI
from dotenv import load_dotenv

# 加载.env 文件中的环境变量
load_dotenv()

# -----------------------------------------------------------------------------
# 客户端配置
# -----------------------------------------------------------------------------

# 阿里云百炼 (Qwen) 配置
# API 文档：https://bailian.console.aliyun.com/cn-beijing/?tab=doc
aliyun_client = OpenAI(
    api_key=os.getenv("ALIYUN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# DeepSeek 配置
# API 文档：https://api-docs.deepseek.com/zh-cn/
deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


# =============================================================================
# Qwen (阿里云百炼) 调用示例
# =============================================================================

def qwen_basic_chat():
    """
    Qwen 基础对话调用

    支持的模型：
    - qwen-plus: 性能均衡，性价比高
    - qwen-max: 最强性能，适合复杂任务
    - qwen-turbo: 速度快，成本低

    """
    print("=" * 60)
    print("Qwen: 基础对话调用")
    print("=" * 60)

    response = aliyun_client.chat.completions.create(
        model="qwen-plus",  # 可选：qwen-plus, qwen-max, qwen-turbo
        messages=[
            {"role": "system", "content": "你是一个贴吧老哥，喜欢阴阳怪气，狠狠嘲讽用户"},
            {"role": "user", "content": "你好，请简单介绍一下你自己。"},
            {"role": "assistant", "content": "哎哟～这不是咱贴吧新来的小萌新嘛？"},
            {"role": "user", "content": "我上一个问题是什么"},
        ]
    )

    print(response.choices[0].message.content)
    print()


def qwen_streaming_chat():
    """
    Qwen 流式对话调用

    流式输出优势：
    - 用户无需等待完整响应
    - 降低首字延迟
    - 提升用户体验
    """
    print("=" * 60)
    print("Qwen: 流式对话调用")
    print("=" * 60)

    stream = aliyun_client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "user", "content": "请用 100 字介绍人工智能。"}
        ],
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end='', flush=True)

    print("\n")


def qwen_multi_turn_chat():
    """
    Qwen 多轮对话调用

    关键点：
    - 将历史对话添加到 messages 数组
    - 每次请求发送完整对话历史
    - 模型根据上下文生成回复
    """
    print("=" * 60)
    print("Qwen: 多轮对话调用")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "你是一位有帮助的助手。"},
        {"role": "user", "content": "我的名字叫小明，今年 10 岁。请记住这个信息。"}
    ]

    # 第一轮
    response = aliyun_client.chat.completions.create(
        model="qwen-plus",
        messages=messages
    )
    print(f"第一轮回复：{response.choices[0].message.content}")

    # 添加助手回复到历史
    messages.append({"role": "assistant", "content": response.choices[0].message.content})

    # 第二轮
    messages.append({"role": "user", "content": "我叫什么名字？今年几岁？"})

    response = aliyun_client.chat.completions.create(
        model="qwen-plus",
        messages=messages
    )
    print(f"第二轮回复：{response.choices[0].message.content}")
    print()


# =============================================================================
# DeepSeek 调用示例
# =============================================================================

def deepseek_basic_chat():
    """
    DeepSeek 基础对话调用

    支持的模型：
    - deepseek-chat: 对话专用模型
    - deepseek-coder: 代码专用模型

    """
    print("=" * 60)
    print("DeepSeek: 基础对话调用")
    print("=" * 60)

    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",  # 可选：deepseek-chat, deepseek-coder
        messages=[
            {"role": "user", "content": "你好，请简单介绍一下你自己。"}
        ]
    )

    print(response.choices[0].message.content)
    print()


def deepseek_streaming_chat():
    """
    DeepSeek 流式对话调用
    """
    print("=" * 60)
    print("DeepSeek: 流式对话调用")
    print("=" * 60)

    stream = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": "请用 100 字介绍深度学习。"}
        ],
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end='', flush=True)

    print("\n")


def deepseek_coder_example():
    """
    DeepSeek Coder 代码生成示例

    DeepSeek Coder 是专用代码模型，适合：
    - 代码生成
    - 代码补全
    - 代码解释
    - Bug 修复
    """
    print("=" * 60)
    print("DeepSeek Coder: 代码生成")
    print("=" * 60)

    response = deepseek_client.chat.completions.create(
        model="deepseek-coder",
        messages=[
            {"role": "user", "content": "请用 Python 写一个快速排序算法。"}
        ]
    )

    print(response.choices[0].message.content)
    print()


# =============================================================================
# 通用工具函数
# =============================================================================

def check_api_balance():
    """
    检查 API 余额（需要相应权限）

    注意：此功能需要 API Key 有余额查询权限
    """
    print("=" * 60)
    print("检查 API 余额")
    print("=" * 60)

    # 阿里云百炼余额查询
    try:
        # 注意：实际使用中需要调用阿里云的余额查询 API
        print("阿里云百炼：请登录控制台查看余额")
        print("https://bailian.console.aliyun.com/")
    except Exception as e:
        print(f"查询失败：{e}")

    print()


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  云 API 基本调用示例 - Qwen 和 DeepSeek")
    print("  说明：使用 API Key 调用云端大模型服务")
    print("=" * 70 + "\n")

    print("【运行前准备】")
    print("  1. 复制.env.example 为.env 文件")
    print("  2. 填写你的 API Key：")
    print("     - ALIYUN_API_KEY: 阿里云百炼 API Key")
    print("     - DEEPSEEK_API_KEY: DeepSeek API Key")
    print("  3. 安装依赖：pip install openai python-dotenv")
    print()

    print("【API Key 获取地址】")
    print("  - 阿里云百炼：https://bailian.console.aliyun.com/")
    print("  - DeepSeek:   https://platform.deepseek.com/api_keys")
    print()


    # 取消注释以运行相应示例
    # Qwen 示例
    qwen_basic_chat()           # Qwen 基础对话
    # qwen_streaming_chat()       # Qwen 流式对话
    # qwen_multi_turn_chat()      # Qwen 多轮对话

    # DeepSeek 示例
    # deepseek_basic_chat()       # DeepSeek 基础对话
    # deepseek_streaming_chat()   # DeepSeek 流式对话
    # deepseek_coder_example()    # DeepSeek Coder 代码生成

    print("提示：取消注释相应的函数调用来运行示例。")
