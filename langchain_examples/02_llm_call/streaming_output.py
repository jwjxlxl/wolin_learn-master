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
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from utils.model_utils import get_model, setup_encoding
setup_encoding()


# =============================================================================
# 示例 1: 最简单的流式调用
# =============================================================================

def simplest_streaming():
    """
    把 invoke() 换成 stream()，就得到了流式输出。

    stream() 返回一个生成器，每次 yield 一小块文本。
    chunk.content 是当前这一小块的内容。
    """

    model = get_model("qwen")
    print("AI 正在输入: ", end="", flush=True)
    for chunk in model.stream("请用 50 字左右介绍人工智能。"):
        print(chunk.content, end='', flush=True)
    print()



# =============================================================================
# 示例 2: 收集流式输出 — 边显示边保存
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
