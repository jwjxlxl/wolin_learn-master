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
from llm_basic import get_model

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)


# =============================================================================
# 示例 1: 最简单的流式调用
# =============================================================================

def simplest_streaming():
    """
    把 invoke() 换成 stream()，就得到了流式输出。

    stream() 返回一个生成器，每次 yield 一小块文本。
    chunk.content 是当前这一小块的内容。
    """

    print(f"\n-- 示例 1: 最简单的流式调用")

    model = get_model("qwen")
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
    import time

    print(f"\n-- 示例 2: 流式 vs 非流式对比")

    model = get_model("qwen")
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
        time.sleep(0.1)
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
    print(f"\n-- 示例 3: 边显示边收集完整内容")

    model = get_model("qwen")
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

    simplest_streaming()
    streaming_vs_invoke()
