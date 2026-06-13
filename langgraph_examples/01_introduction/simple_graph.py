# =============================================================================
# 最简图 — 两个节点的顺序执行
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解 StateGraph / Node / Edge / State 四个核心概念
#   ✅ 用 add_node + add_edge 构建最简单的图
#   ✅ 用 invoke() 执行图并获取结果
#
# 运行前检查：
# 1. 已安装依赖：pip install langgraph langchain-core
# 2. 无需 API Key（本示例不调用模型，纯本地执行）
# =============================================================================

import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# =============================================================================
# 核心概念：StateGraph 的四个基本元素
# =============================================================================
"""
LangGraph 最核心的四个概念（记住就能上手）：

┌──────────────────────────────────────────────────────────────┐
│  元素          │  比喻          │  代码                        │
├──────────────────────────────────────────────────────────────┤
│  StateGraph    │  整张地铁图    │  StateGraph(MyState)         │
│  Node（节点）  │  地铁站        │  .add_node("站名", 函数)     │
│  Edge（边）    │  轨道          │  .add_edge("站A", "站B")    │
│  State（状态） │  乘客（数据）  │  TypedDict 定义              │
│  START / END   │  起点站/终点站 │  常量，无需定义              │
│  compile()     │  开通运营      │  graph.compile()             │
└──────────────────────────────────────────────────────────────┘

生活化比喻：
  把 LangGraph 想象成"地铁线路图"
  - 站点（Node）= 每个处理步骤
  - 线路（Edge）= 站点间的连接
  - 乘客（State）= 在图中传递的数据
  - 换乘站（Conditional Edge）= 根据条件选择不同线路
"""


# =============================================================================
# 示例: 两个节点的顺序执行
# =============================================================================

def simple_two_node_graph():
    """
    最简单的 LangGraph：两个节点顺序执行。

    START → 问候 → 回应 → END

    关键点：
      - ⚠️ 节点函数返回的是"状态更新"，不是"替换"——只会更新你返回的字段
      - add_edge 添加的是固定边（无条件，一定执行）
      - invoke() 的参数是初始状态
    """
    print(f"\n-- 示例: 最简图 — 两个节点顺序执行")

    from typing import TypedDict
    from langgraph.graph import StateGraph, START, END

    # 1. 定义状态：图中传递的数据结构
    #    TypedDict 是 Python 的类型提示，LangGraph 用它定义"贯穿整张图的状态"
    class GraphState(TypedDict):
        """图的状态：存储消息和步骤计数"""
        messages: list[str]
        step_count: int

    # 2. 定义节点函数：每个节点接收 State，返回部分更新的 State
    def greet(state: GraphState):
        """节点 1：添加问候消息"""
        print("  [节点: greet] 添加问候语")
        return {
            "messages": ["你好！很高兴见到你！"],
            "step_count": 1
        }

    def respond(state: GraphState):
        """节点 2：添加回应消息，读取上一步的 step_count 并 +1"""
        print("  [节点: respond] 添加回应")
        return {
            "messages": ["你好！我也很高兴！"],
            "step_count": state["step_count"] + 1
        }

    # 3. 构建图：连接节点
    graph = (
        StateGraph(GraphState)
        .add_node("greet", greet)
        .add_node("respond", respond)
        .add_edge(START, "greet")       # 起点 → greet
        .add_edge("greet", "respond")   # greet → respond
        .add_edge("respond", END)       # respond → 终点
        .compile()
    )

    # 4. 执行图
    print("【执行图】")
    result = graph.invoke({"messages": [], "step_count": 0})
    print(f"  最终消息: {result['messages']}")
    print(f"  总步骤数: {result['step_count']}")
    print()


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  最简图 — 两个节点顺序执行")
    print("  理解 StateGraph 的四个核心元素")
    print("=" * 70 + "\n")

    print("【运行前检查】")
    print("  1. 已安装依赖：pip install langgraph langchain-core")
    print("  2. 无需 API Key（本示例不调用模型，纯本地执行）")
    print()

    simple_two_node_graph()

    print("=" * 70)
    print("  接下来学习：conditional_branch.py（条件分支）")
    print("=" * 70 + "\n")
