# =============================================================================
# 流式输出
# =============================================================================
#  
# 用途：教学演示 - 实现打字机效果的流式输出
#
# 核心概念：
#   - 什么是流式输出？（打字机效果）
#   - 为什么需要流式？（降低等待焦虑）
# =============================================================================

# -----------------------------------------------------------------------------
# 运行前检查
# -----------------------------------------------------------------------------
# 1. 已安装 Ollama 服务
# 2. 已下载模型：ollama pull qwen3.5:2b
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


# =============================================================================
# 第一部分：理解流式输出
# =============================================================================
"""
什么是流式输出？

🐢 非流式（普通模式）
   用户提问 → 等待 5 秒 → 一次性显示完整答案
   体验：感觉"卡住"了，不知道模型在工作还是出错了

   [-----等待 5 秒-----] → 完整答案突然出现！

⚡ 流式（打字机效果）
   用户提问 → 0.5 秒后开始逐字显示 → 持续显示直到完成
   体验：看到内容在不断生成，知道模型在工作

   你 → 好 → ！→ 我 → 是 → A → I → ...

为什么需要流式？
1. 降低首字延迟（0.5 秒就能看到内容）
2. 减少等待焦虑（看到内容在生成）
3. 提升用户体验（像真人在打字）
"""


# =============================================================================
# 示例 1: 最简单的流式调用
# =============================================================================

def simplest_streaming():
    """
    最简单的流式调用

    只需要把 invoke() 改成 stream()
    """
    print("=" * 60)
    print("示例 1: 最简单的流式调用")
    print("=" * 60)

    from langchain_ollama import ChatOllama

    model = ChatOllama(model="qwen3.5:2b")

    # 关键区别：
    # invoke()  → 等待完整回复，一次性返回
    # stream()  → 逐字返回，返回一个生成器

    print("AI 正在输入：", end="", flush=True)

    # stream() 返回一个生成器，可以逐个获取内容块
    for chunk in model.stream("请用 100 字左右介绍人工智能。"):
        # chunk.content: 当前内容块的文本
        print(chunk.content, end='', flush=True)

    print("\n")


# =============================================================================
# 示例 2: 对比流式和非流式
# =============================================================================

def streaming_vs_non_streaming():
    """
    对比流式和非流式的区别

    实际感受两种模式的差异
    """
    print("=" * 60)
    print("示例 2: 流式 vs 非流式对比")
    print("=" * 60)

    from langchain_ollama import ChatOllama
    import time

    model = ChatOllama(model="qwen3.5:2b")
    question = "请用三句话介绍你自己。"

    # 非流式
    print("【非流式模式】")
    print("开始调用...", end="", flush=True)
    start = time.time()
    response = model.invoke(question)
    elapsed = time.time() - start
    print(f" 耗时：{elapsed:.2f}秒")
    print(f"AI 回复：{response.content}")
    print()

    # 流式
    print("【流式模式】")
    print("AI 正在输入：", end="", flush=True)
    start = time.time()
    for chunk in model.stream(question):
        print(chunk.content, end='', flush=True)
    elapsed = time.time() - start
    print(f"\n耗时：{elapsed:.2f}秒")
    print()


# =============================================================================
# 示例 3: 收集流式输出内容
# =============================================================================

def collect_streaming_output():
    """
    收集流式输出的完整内容

    有时需要边显示边保存完整内容
    """
    print("=" * 60)
    print("示例 3: 收集流式输出内容")
    print("=" * 60)

    from langchain_ollama import ChatOllama

    model = ChatOllama(model="qwen3.5:2b")

    # 用列表收集所有内容块
    full_response = []

    print("AI 正在输入：", end="", flush=True)

    for chunk in model.stream("请用 100 字介绍机器学习。"):
        print(chunk.content, end='', flush=True)
        full_response.append(chunk.content)

    # 拼接完整回复
    complete_text = "".join(full_response)
    print(f"\n\n完整回复长度：{len(complete_text)} 字符")
    print()


# =============================================================================
# 示例 4: 流式输出 + 错误处理
# =============================================================================

def streaming_with_error_handling():
    """
    流式输出时处理可能的错误

    实际应用中应该添加错误处理
    """
    print("=" * 60)
    print("示例 4: 流式输出 + 错误处理")
    print("=" * 60)

    from langchain_ollama import ChatOllama

    model = ChatOllama(model="qwen3.5:2b")

    try:
        print("AI 正在输入：", end="", flush=True)

        for chunk in model.stream("你好！"):
            print(chunk.content, end='', flush=True)

        print("\n")

    except Exception as e:
        print(f"\n调用失败：{e}")
        print()
        print("可能的原因：")
        print("  1. Ollama 服务未启动")
        print("  2. 模型未下载")
        print("  3. 网络连接问题")


# =============================================================================
# 示例 5: 实用场景 - 聊天机器人
# =============================================================================

def chat_bot_with_streaming():
    """
    实用的聊天机器人示例

    结合流式输出和多轮对话
    """
    print("=" * 60)
    print("示例 5: 流式聊天机器人")
    print("=" * 60)
    print("（输入 'quit' 退出）\n")

    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage, AIMessage

    model = ChatOllama(model="qwen3.5:2b")
    messages = []

    while True:
        user_input = input("你：").strip()
        if user_input.lower() == 'quit':
            break

        # 添加用户消息
        messages.append(HumanMessage(content=user_input))

        # 流式输出 AI 回复
        print("AI: ", end='', flush=True)

        full_response = []
        for chunk in model.stream(messages):
            print(chunk.content, end='', flush=True)
            full_response.append(chunk.content)

        print()  # 换行

        # 添加 AI 回复到历史记录
        messages.append(AIMessage(content="".join(full_response)))


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  LangChain 流式输出")
    print("  说明：打字机效果的实现")
    print("=" * 70 + "\n")

    print("【运行前检查】")
    print("  1. 已安装 Ollama 服务")
    print("  2. 已下载模型：ollama pull qwen3.5:2b")
    print()

    # 运行示例
    simplest_streaming()
    # streaming_vs_non_streaming()
    # collect_streaming_output()

    # 聊天机器人（交互式，按需运行）
    # chat_bot_with_streaming()

    print("=" * 70)
    print("  接下来学习：03_prompt/prompt_template.py")
    print("=" * 70 + "\n")
