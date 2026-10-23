# =============================================================================
# 07_recursive_chunking — 递归切片（Recursive Chunking）
# =============================================================================
# 用途：学习"递归切片"策略 — LangChain RecursiveCharacterTextSplitter 的核心原理
# 难度：⭐⭐（2 星）
# =============================================================================
# 核心思想（按分隔符优先级"从大到小"递归降级）：
#   固定切片无视语义边界，容易把句子/段落拦腰切断；
#   递归切片尽量在"最自然"的边界切分：
#     ① 优先按段落（\n\n）切 —— 语义最完整
#     ② 段落太大 → 降级按行（\n）切
#     ③ 还是太大 → 降级按句子（。）切
#     ④ 再不行 → 按逗号（，）、空格切
#     ⑤ 全都用不上 → 按固定长度硬切（兜底）
#   切出的小段再贪心合并到接近 chunk_size，兼顾"块够大"与"边界自然"。
#
# 对标：LangChain 的 RecursiveCharacterTextSplitter，本文件手写其核心逻辑，
#       并用 trace 记录"递归降级"的完整过程，便于看清每一步。
# =============================================================================

# 分隔符优先级：从大到小（段落 → 行 → 句子 → 逗号 → 空格）
SEPARATORS = ["\n\n", "\n", "。", "，", " "]

# 边界名称（用于结果标注）
SEP_NAMES = {"\n\n": "段落", "\n": "行", "。": "句子", "，": "逗号", " ": "空格"}

# 演示文本：2 个短段 + 1 个超长段（内含两行，最长约 200 字）+ 2 个中段
DEMO_TEXT = (
    "RAG 是检索增强生成的缩写，核心思路是先检索后生成。"
    "\n\n"
    "它能缓解大模型幻觉，让答案可追溯、可验证。"
    "\n\n"
    "文档切片是 RAG 流程的第一步，切片质量直接决定检索的上限。"
    "常用的切片方法包括固定字符切片、滑动窗口切片、按段落切片、递归切片以及借助大模型辅助的语义切片，"
    "不同方法在实现难度、语义完整性与计算成本上各有取舍，需要结合文档结构与业务场景灵活选择。\n"
    "切片太大时向量会被无关内容稀释导致检索不准，切片太小时语义不完整导致生成质量下降。"
    "工程实践中通常从三百到五百字符起步，再配合百分之十到二十的重叠率，在两者之间取得平衡。"
    "\n\n"
    "向量化把文本转换成向量，语义相近的内容在向量空间中彼此靠近，检索时用余弦相似度衡量这种接近程度。"
    "\n\n"
    "检索阶段用问题向量在库中找出最相似的若干块，按分数排序截断后送入生成环节。"
)


def recursive_split(text, separators=None, chunk_size=100, trace=None, depth=0):
    """递归切片：优先在大边界切，切不开就降级用小边界，最后硬切兜底

    参数:
        text: 输入文本
        separators: 分隔符列表（优先级从高到低），默认用模块级 SEPARATORS
        chunk_size: 单块长度上限
        trace: 可选列表，传入后逐条记录降级过程（教学演示用）
        depth: 当前递归深度（控制 trace 缩进，内部参数）
    返回:
        [(块文本, 边界类型), ...]：边界类型标明该块由哪级分隔符封块
    """
    if separators is None:
        separators = SEPARATORS

    # 块已足够小 → 直接返回（仅当整段文本都不超限时触发）
    if len(text) <= chunk_size:
        return [(text, "整体")]

    # ① 找出文本中实际存在的最高优先级分隔符
    sep = next((s for s in separators if s in text), None)

    # ② 没有任何分隔符可用 → 按固定长度硬切（兜底）
    if sep is None:
        return [(text[i:i + chunk_size], "硬切")
                for i in range(0, len(text), chunk_size)]

    name = SEP_NAMES.get(sep, sep)  # 本级边界名称，如 "段落" / "句子"
    segments = [s for s in text.split(sep) if s]

    if trace is not None:  # 记录本次切分（换行符转义后可见）
        disp = sep.replace("\n", "\\n")
        trace.append(f"{'  ' * depth}按『{disp}』切出 {len(segments)} 段（上限 {chunk_size} 字）")

    # ③ 逐段处理：小段贪心合并，大段用"剩余分隔符"递归再切
    rest = separators[separators.index(sep) + 1:]  # 优先级更低的分隔符
    chunks, buffer = [], ""
    for seg in segments:
        if len(seg) > chunk_size:  # 段超限 → 封存缓冲，降级重切
            if trace is not None:
                trace.append(f"{'  ' * depth}  其中 {len(seg)} 字超限 → 降级")
            if buffer:
                chunks.append((buffer, name))
                buffer = ""
            chunks.extend(recursive_split(seg, rest, chunk_size, trace, depth + 1))
        elif len(buffer) + len(seg) + len(sep) <= chunk_size:
            buffer = f"{buffer}{sep}{seg}" if buffer else seg  # 贪心合并相邻小段
        else:
            if buffer:
                chunks.append((buffer, name))  # 放不下 → 封块
            buffer = seg
    if buffer:
        chunks.append((buffer, name))
    return chunks


def show_chunk(chunk, label, index):
    """单行展示一个切块（换行转义，超长截断）"""
    text = chunk[:38].replace("\n", "\\n")
    more = "…" if len(chunk) > 38 else ""
    print(f"  {index}. [{len(chunk):>3} 字 | {label}] {text}{more}")


def demo_degradation():
    """示例 1：长文档的完整递归降级过程"""
    print(f"\n-- 示例 1: 递归降级过程（chunk_size=60）")

    trace = []
    chunks = recursive_split(DEMO_TEXT, chunk_size=60, trace=trace)

    paragraphs = [p for p in DEMO_TEXT.split("\n\n") if p]
    print(f"原文 {len(DEMO_TEXT)} 字 / {len(paragraphs)} 个段落"
          f"（最长段落 {max(len(p) for p in paragraphs)} 字），切出 {len(chunks)} 块\n")

    print("降级过程：")
    for line in trace:
        print(f"  {line}")

    print("\n最终切块（标注封块边界）：")
    for i, (c, label) in enumerate(chunks, 1):
        show_chunk(c, label, i)

    print("\n  要点：两个短段自动合并成一块；超长段逐级降级（段落 → 行 → 句子 → 逗号）；")
    print("  每块的边界都落在段落 / 句子 / 逗号等自然语义位置，避免无意义硬切。")


def demo_chunk_size():
    """示例 2：chunk_size 如何影响降级深度"""
    print(f"\n-- 示例 2: chunk_size 越小，降级越深")

    for size in (60, 100):
        chunks = recursive_split(DEMO_TEXT, chunk_size=size)
        counts = {}
        for _, label in chunks:
            counts[label] = counts.get(label, 0) + 1
        detail = " / ".join(f"{name} {n} 块" for name, n in counts.items())
        print(f"  chunk_size={size:<4} 共 {len(chunks)} 块：{detail}")

    print("\n  上限越小 → 越多段落装不下 → 降级到更小的边界，块数变多；")
    print("  上限越大 → 更多内容在段落 / 句子级直接封块，上下文更完整。")
    print("  工程常用 300-500 字起步，再配合 10%-20% 重叠（见 02_sliding_window.py）。")


if __name__ == "__main__":
    print("=" * 60)
    print("  递归切片（Recursive Chunking）")
    print("=" * 60)

    demo_degradation()   # 示例 1: 递归降级过程
    # demo_chunk_size()    # 示例 2: chunk_size 与降级深度

    print("=" * 60)
    print("  优先大边界、逐级降级 = 递归切片的核心")
    print("  下一步：03_retrieval_methods/（检索方法）")
    print("=" * 60)
