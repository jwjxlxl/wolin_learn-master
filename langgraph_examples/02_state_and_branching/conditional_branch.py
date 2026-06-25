# =============================================================================
# 条件分支 — 根据状态选择不同路径
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解条件边（Conditional Edge）的工作原理
#   ✅ 用 add_conditional_edges 实现分支路由
#   ✅ 写路由函数：根据状态返回下一步节点名
#
# 运行前检查：
# 1. 已安装依赖：pip install langgraph langchain-core
# 2. 无需 API Key（本示例不调用模型，纯本地执行）
# =============================================================================

import sys
import os
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# =============================================================================
# 核心概念：条件边（Conditional Edge）
# =============================================================================
"""
什么是条件边？

  固定边（Edge）：A → B（无论什么情况，都走这条路）
  条件边（Conditional Edge）：A → (判断) → B 或 C 或 D（根据条件选择）

  .add_conditional_edges(源节点, 路由函数, 映射字典)

  执行过程：
    1. 流程走到"源节点"后，调用路由函数 route(state)
    2. route 返回一个字符串，比如 "path_b"
    3. 在映射字典中查找 "path_b" 对应的目标节点
    4. 流程走向那个目标节点

  生活化比喻：
    固定边  = 单行道（只能直走）
    条件边  = 十字路口（根据红绿灯选择方向）
    路由函数 = 交警（判断当前该往哪走）
"""


# =============================================================================
# 示例: 情感分类 — 根据文本内容走不同路径
# =============================================================================

def conditional_branch_graph():
    """
    带条件分支的图：根据消息内容走不同路径。

    START → 分类 → (正面/负面/中性) → END

    关键点：
      - 路由函数返回的是字符串，必须和映射字典中的 key 对应
      - 映射字典的值可以是节点名或 END
      - 所有分支最终都走到 END（不走回头路）
    """
    print(f"\n-- 示例: 条件分支 — 情感分类")

    from typing import TypedDict
    from langgraph.graph import StateGraph, START, END

    # 定义State
    class SentimentState(TypedDict):
        text: str
        category: str
        reply: str

    def classify(state: SentimentState):
        """分类节点：简单判断情感倾向（关键词匹配）"""
        positive_words = ["好", "棒", "开心", "喜欢", "谢谢", "不错"]
        negative_words = ["差", "糟", "难过", "讨厌", "失望", "不好"]

        text = state["text"]
        if any(w in text for w in positive_words):
            category = "positive"
        elif any(w in text for w in negative_words):
            category = "negative"
        else:
            category = "neutral"

        print(f"  [节点: classify] 文本 '{text}' → 分类: {category}")
        return {"category": category}

    def positive_reply(state: SentimentState):
        """正面回复节点"""
        print("  [节点: positive_reply] 生成正面回复")
        return {"reply": "太好了！听到你这么说我很开心！😊"}

    def negative_reply(state: SentimentState):
        """负面回复节点"""
        print("  [节点: negative_reply] 生成安慰回复")
        return {"reply": "别担心，一切都会好起来的！有什么我可以帮你的吗？"}

    def neutral_reply(state: SentimentState):
        """中性回复节点"""
        print("  [节点: neutral_reply] 生成中性回复")
        return {"reply": "好的，我明白了。"}

    def route(state: SentimentState):
        """条件路由函数：根据分类结果决定下一步"""
        cat = state["category"]
        if cat == "positive":
            return "positive_reply"
        elif cat == "negative":
            return "negative_reply"
        else:
            return "neutral_reply"

    # 构建图
    graph = (
        StateGraph(SentimentState)
        # 添加节点
        .add_node("classify", classify)
        .add_node("positive_reply", positive_reply)
        .add_node("negative_reply", negative_reply)
        .add_node("neutral_reply", neutral_reply)
        .add_edge(START, "classify")
        # 条件边：路由函数决定走哪个回答节点
        .add_conditional_edges("classify", route, {
            "positive_reply": "positive_reply",
            "negative_reply": "negative_reply",
            "neutral_reply": "neutral_reply",
        })
        .add_edge("positive_reply", END)
        .add_edge("negative_reply", END)
        .add_edge("neutral_reply", END)
        .compile()
    )

    # 保存图为 PNG
    images_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images')
    os.makedirs(images_dir, exist_ok=True)
    png_path = os.path.join(images_dir, 'conditional_branch.png')
    with open(png_path, 'wb') as f:
        f.write(graph.get_graph().draw_mermaid_png())
    print(f"  图已保存到: {png_path}\n")

    # 测试不同输入
    print("【测试 1: 正面输入】")
    r1 = graph.invoke({"text": "今天天气真好，心情不错！", "category": "", "reply": ""})
    print(f"  回复: {r1['reply']}\n")

    print("【测试 2: 负面输入】")
    r2 = graph.invoke({"text": "今天的体验太差了，很失望。", "category": "", "reply": ""})
    print(f"  回复: {r2['reply']}\n")

    print("【测试 3: 中性输入】")
    r3 = graph.invoke({"text": "我明天要去开会。", "category": "", "reply": ""})
    print(f"  回复: {r3['reply']}\n")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  条件分支 — 根据状态选择不同路径")
    print("  理解条件边（Conditional Edge）的工作原理")
    print("=" * 70 + "\n")

    conditional_branch_graph()

    print("=" * 70)
    print("  接下来学习：agent_react.py（Agent 循环 — ReAct 模式）")
    print("=" * 70 + "\n")
