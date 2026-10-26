# =============================================================================
# 第一个 LangChain 程序 — 5 分钟体验成功感
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 用 3 行代码调用本地 AI 模型
#   ✅ 实现打字机效果的流式输出
#   ✅ 理解 invoke() 和 stream() 的区别
# =============================================================================

# -----------------------------------------------------------------------------
# 运行前检查
# -----------------------------------------------------------------------------
# 1. 已安装 Ollama 服务（下载地址：https://ollama.ai）
# 2. 已下载模型：ollama pull qwen3.5:2b
# 3. Ollama 服务正在运行（终端输入 ollama serve）
# -----------------------------------------------------------------------------

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from utils.model_utils import setup_encoding
setup_encoding()

from langchain_ollama import ChatOllama


# =============================================================================
# 示例 1: 最简单的调用 — 3 行代码让 AI 回答你的问题
# =============================================================================

def simplest_call():
    """
    演示最简短的 LangChain 调用方式。

    核心概念：
    - ChatOllama: LangChain 中用于连接 Ollama 本地模型的"接口卡"
    - invoke(): 把消息发给模型，等待完整回复后一次性返回（像发短信—等对方打完再显示）
    - .content: 从模型返回的"信封"中取出"信纸"（纯文本内容）

    """

    # 第 1 行：创建模型实例
    model = ChatOllama(model="qwen3.5:2b")

    # 第 2 行：发送消息并等待回复
    response = model.invoke("你好，请用一句话介绍你自己。")

    # 第 3 行：取出纯文本内容并打印
    print(response.content)
    print()


# =============================================================================
# 示例 2: 使用云端 API（阿里云 Qwen 等）— 需要 API Key
# =============================================================================

def cloud_api_call():
    """
    演示如何使用云端大模型 API（而非本地 Ollama）。

    核心概念：
    - ChatOpenAI: LangChain 中用于连接 OpenAI 兼容 API 的"万能接口卡"
      （不仅限于 OpenAI，任何兼容 OpenAI 格式的 API 都能用这个接口，如阿里云/DeepSeek）
    - base_url: API 的"门牌号"（不同服务商地址不同）
    - 环境变量: 把 API Key 存在 .env 文件中，代码通过 os.getenv() 读取，避免泄露
    """


    try:
        from langchain_openai import ChatOpenAI
        import os
        from dotenv import load_dotenv

        load_dotenv()

        # 检查是否配置了 API Key
        api_key = os.getenv("ALIYUN_API_KEY")
        if not api_key or api_key == "sk-your-aliyun-api-key-here":
            print("⚠️ 未配置 API Key，跳过此示例")
            return

        # 创建云端模型实例
        # base_url 指向阿里云 DashScope 的 OpenAI 兼容接口
        cloud_model = ChatOpenAI(
            model="qwen-plus",
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        response = cloud_model.invoke("你好，请用一句话介绍你自己。")
        print(f"云端模型回复：{response.content}")
        print()

    except ImportError:
        print("⚠️ 未安装 langchain-openai，跳过此示例")
        print("提示：pip install langchain-openai")
        print()


# =============================================================================
# 示例 3: 流式输出 — 像打字机一样逐字显示
# =============================================================================

def streaming_call():
    """
    演示流式输出（streaming）——内容边生成边显示，而不是等全部完成才出现。

    核心概念：
    - invoke() vs stream():
      invoke() = 等人把话全说完再转述给你（等得着急）
      stream() = 对方边说边转述（每说一个字就显示一个字）

    为什么需要流式？
    1. 降低等待焦虑——用户能看到"AI 正在工作"
    2. 首字延迟低——0.5 秒就能看到第一个字
    3. 体验更好——像真人在打字

    """

    model = ChatOllama(model="qwen3.5:9b")

    # stream() 返回一个生成器，每次 yield 一小块内容
    print("AI 正在输入：", end="", flush=True)
    for chunk in model.stream("请用 50 字左右介绍人工智能。"):
        # chunk.content: 当前这一小块文本
        print(chunk.content, end='', flush=True)

    print("\n")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    # ★ 默认运行最简单的示例 — 让学生第一次就成功
    simplest_call()

    # 有 API Key 可取消注释体验云端模型
    cloud_api_call()
