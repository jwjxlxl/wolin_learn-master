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

# 设置 UTF-8 编码（Windows 中文环境必需，防止 emoji/中文乱码）
import sys
import io

sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer,
    encoding='utf-8',
    errors='replace',
    line_buffering=True
)

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

    生活化比喻：
    ChatOllama(model="qwen3.5:2b") = 拨通电话给一个叫"qwen3.5:2b"的 AI 客服
    model.invoke("...")             = 你对着话筒说话
    response.content               = 听筒里传来的回复
    """
    print("=" * 60)
    print("示例 1: 最简单的调用（3 行代码）")
    print("=" * 60)

    # 第 1 行：创建模型实例（就像拿起电话拨号）
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

    生活化比喻：
    本地 Ollama = 家里的厨房（免费但菜式有限）
    云端 API     = 外面的餐厅（付费但选择多、味道好）
    """
    print("=" * 60)
    print("示例 2: 使用云端 API（阿里云 Qwen）")
    print("=" * 60)

    try:
        from langchain_openai import ChatOpenAI
        import os
        from dotenv import load_dotenv

        load_dotenv()

        # 检查是否配置了 API Key
        api_key = os.getenv("ALIYUN_API_KEY")
        if not api_key or api_key == "sk-your-aliyun-api-key-here":
            print("⚠️ 未配置 API Key，跳过此示例")
            print("提示：复制 .env.example 为 .env，填写你的 ALIYUN_API_KEY")
            print()
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

    生活化比喻：
    invoke() = 下载完整个视频再看
    stream() = 边缓冲边播放（youtube 那样）
    """
    print("=" * 60)
    print("示例 3: 流式输出（打字机效果）")
    print("=" * 60)

    model = ChatOllama(model="qwen3.5:2b")

    # stream() 返回一个生成器，每次 yield 一小块内容
    print("AI 正在输入：", end="", flush=True)
    for chunk in model.stream("请用 50 字左右介绍人工智能。"):
        # chunk.content: 当前这一小块文本
        print(chunk.content, end='', flush=True)

    print("\n")


# =============================================================================
# 示例 4: 带错误处理的调用 — 写生产代码必需
# =============================================================================

def call_with_error_handling():
    """
    演示如何优雅地处理模型调用可能出现的异常。

    常见失败原因：
    1. Ollama 服务没启动 → 运行 'ollama serve'
    2. 模型还没下载 → 运行 'ollama pull qwen3.5:2b'
    3. 网络问题 → 检查是否能访问 Ollama 端口

    生活化比喻：
    错误处理 = 汽车的安全气囊
    平时用不到，但一旦出问题，它能防止"车毁人亡"（程序崩溃）
    """
    print("=" * 60)
    print("示例 4: 带错误处理的调用")
    print("=" * 60)

    model = ChatOllama(model="qwen3.5:2b")

    try:
        response = model.invoke("测试消息")
        print(f"调用成功：{response.content}")

    except Exception as e:
        # 出了任何问题都进入这里，不会让程序直接崩溃
        print(f"❌ 调用失败：{e}")
        print()
        print("可能的原因：")
        print("  1. Ollama 服务未启动 → 终端运行 ollama serve")
        print("  2. 模型未下载 → 终端运行 ollama pull qwen3.5:2b")
        print("  3. 网络连接问题 → 检查防火墙/代理设置")

    print()


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  第一个 LangChain 程序")
    print("  用最简单的代码体验 LangChain 的核心能力")
    print("=" * 70 + "\n")

    print("【运行前检查】")
    print("  1. 已安装 Ollama 服务")
    print("  2. 已下载模型：ollama pull qwen3.5:2b")
    print("  3. Ollama 服务正在运行")
    print()

    # ★ 默认运行最简单的示例 — 让学生第一次就成功
    simplest_call()
    streaming_call()
    call_with_error_handling()

    # 有 API Key 可取消注释体验云端模型
    # cloud_api_call()

    print("=" * 70)
    print("  恭喜！你已完成第一个 LangChain 程序！")
    print("  接下来学习：02_llm_call/llm_basic.py（深入 LLM 调用）")
    print("=" * 70 + "\n")
