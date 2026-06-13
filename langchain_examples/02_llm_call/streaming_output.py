# =============================================================================
# 流式输出 — 像打字机一样逐字显示
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解 invoke()（一次性返回）和 stream()（逐字返回）的区别
#   ✅ 实现打字机效果的流式输出
#   ✅ 边显示边收集完整内容
# =============================================================================

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)


"""
invoke() vs stream():

  invoke() = 等人把话全说完再转述给你
     用户提问 → 等待 5 秒 → 一次性显示完整答案
     体验：感觉"卡住"了

  stream() = 对方边说边转述，每出一个字就显示一个字
     用户提问 → 0.5 秒开始逐字显示 → 持续到完成
     体验：看到内容在生成，知道模型在工作
"""


# =============================================================================
# 示例 1: 最简单的流式调用
# =============================================================================

def simplest_streaming():
    """
    把 invoke() 换成 stream()，就得到了流式输出。

    stream() 返回一个生成器，每次 yield 一小块文本。
    chunk.content 是当前这一小块的内容。
    """
    from langchain_ollama import ChatOllama

    print(f"\n-- 示例 1: 最简单的流式调用")

    model = ChatOllama(model="qwen3.5:2b")

    print("AI 正在输入: ", end="", flush=True)
    for chunk in model.stream("请用 50 字左右介绍人工智能。"):
        print(chunk.content, end='', flush=True)
    print()


# =============================================================================
# 示例 2: 流式 vs 非流式 — 对比体验
# =============================================================================

def streaming_vs_invoke():
    """
    用计时器直观对比两种模式的差异。

    注意：总耗时差不多，但流式模式"首字延迟"极低——
    用户 0.5 秒就能看到第一个字，不用干等。
    """
    from langchain_ollama import ChatOllama
    import time

    print(f"\n-- 示例 2: 流式 vs 非流式对比")

    model = ChatOllama(model="qwen3.5:2b")
    question = "请用三句话介绍你自己。"

    # 非流式
    print("[非流式] 开始调用...", end="", flush=True)
    start = time.time()
    response = model.invoke(question)
    print(f" 耗时 {time.time() - start:.1f}秒")
    print(f"回复: {response.content}\n")

    # 流式
    print("[流式] AI 正在输入: ", end="", flush=True)
    start = time.time()
    for chunk in model.stream(question):
        print(chunk.content, end='', flush=True)
    print(f"\n耗时 {time.time() - start:.1f}秒")


# =============================================================================
# 示例 3: 收集流式输出 — 边显示边保存
# =============================================================================

def collect_while_streaming():
    """
    有时需要一边流式显示给用户看，一边把完整内容保存下来。

    做法：用一个列表收集所有 chunk，最后 join 起来即可。
    """
    from langchain_ollama import ChatOllama

    print(f"\n-- 示例 3: 边显示边收集完整内容")

    model = ChatOllama(model="qwen3.5:2b")
    collected = []

    print("AI 正在输入: ", end="", flush=True)
    for chunk in model.stream("请用一句话介绍机器学习。"):
        print(chunk.content, end='', flush=True)
        collected.append(chunk.content)

    full_text = "".join(collected)
    print(f"\n\n完整回复: {full_text}")
    print(f"总字符数: {len(full_text)}")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n>>> 02_llm_call/streaming_output — 流式输出\n")

    simplest_streaming()
    streaming_vs_invoke()
    collect_while_streaming()

    # 接下来学习: 03_prompt/prompt_template.py（Prompt 模板）
