# =============================================================================
# 什么是 LangGraph — 用通俗语言理解核心概念
# =============================================================================
#
# 本文件不需要安装任何依赖，直接运行即可。
# 目的：在动手写代码之前，先建立"LangGraph 是什么、能做什么"的整体认知。
# =============================================================================

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)


def without_langgraph():
    """
    演示不用 LangGraph 时写 AI Agent 有多繁琐。

    想象你要做一个"能自动调用工具的 AI 助手"：
    手动管理状态、写循环逻辑、处理条件分支……
    每一步都要手写样板代码。
    """
    print(f"\n-- 没有 LangGraph 时，写一个 ReAct Agent 需要...")

    print("""
    # 1. 手动管理消息历史（每次都自己拼接）
    messages = [{"role": "user", "content": "3加4是多少？"}]

    # 2. 手动写思考-行动循环（while True + 各种判断）
    while True:
        response = model.invoke(messages)
        if not response.tool_calls:
            break  # 没有工具调用了才结束
        for tc in response.tool_calls:
            result = execute_tool(tc.name, tc.args)
        messages.append(result)  # 手动追加 ToolMessage

    # 3. 想加条件分支？自己写 if/elif/else
    # 4. 想加并行？自己写多线程
    # 5. 想可视化？自己画流程图
    """)

    print("结论：每个 AI Agent 都重复造这些轮子 → LangGraph 帮你做掉")


def what_is_langgraph():
    """
    用生活化比喻解释 LangGraph = "地铁线路图"。

    不用地铁图（不用框架）：自己找路、换乘、记站名 — 复杂、易出错
    用地铁图（用 LangGraph）：清晰的站点和线路 — 一目了然

    LangGraph 的核心概念：
      StateGraph — 整张"地铁图"（有状态的工作流图）
      Node（节点）— 地铁站点（每个处理步骤）
      Edge（边）— 地铁线路（站点间的连接）
      State（状态）— 乘客（在图中传递的数据）
      Conditional Edge — 换乘站（根据条件选择不同线路）
    """
    print(f"\n-- LangGraph = 地铁线路图")

    concepts = {
        "StateGraph（状态图）":    {"比喻": "整张地铁图",              "代码": "StateGraph(MyState)"},
        "Node（节点）":           {"比喻": "地铁站：每站做一件事",    "代码": ".add_node(\"站名\", 处理函数)"},
        "Edge（边）":             {"比喻": "轨道：连接两个站",        "代码": ".add_edge(\"站A\", \"站B\")"},
        "START / END":           {"比喻": "起点站 / 终点站",         "代码": "START, END"},
        "State（状态）":          {"比喻": "乘客：在图中传递的数据",   "代码": "TypedDict 定义的数据结构"},
        "Conditional Edge（条件边）": {"比喻": "换乘站：根据方向选线路", "代码": ".add_conditional_edges(...)"},
        "compile()":             {"比喻": "开通运营：让线路跑起来",   "代码": "graph.compile()"},
    }

    for name, info in concepts.items():
        print(f"  {name}")
        print(f"    比喻: {info['比喻']}")
        print(f"    示例: {info['代码']}")


def langgraph_vs_langchain():
    """LangGraph 和 LangChain 的关系 — 互补而非替代。"""
    print(f"\n-- LangGraph 和 LangChain 的关系")

    print("""
    LangChain = "直线流水线"（A → B → C）
      适合：翻译、摘要、格式化输出等线性任务
      特点：数据单向流动，一次性完成

    LangGraph = "地铁网络图"（A → B → (条件判断) → C 或 D → 回到 A）
      适合：Agent 循环、多步骤推理、条件分支
      特点：数据可循环流动，支持复杂路由

    两者可以组合使用！
      - 用 LangChain 的 Prompt / Model / Parser 做组件
      - 用 LangGraph 的 StateGraph 做编排
    """)


def langgraph_applications():
    """LangGraph 能做什么应用？列举典型场景。"""
    print(f"\n-- LangGraph 的典型应用场景")

    cases = [
        ("ReAct Agent",     "思考→行动→观察→再思考的循环",      "StateGraph + Conditional Edge + Tool"),
        ("多步骤推理",       "先规划、再执行、评估后修正",        "Prompt Chain + Evaluator-Optimizer"),
        ("智能客服路由",     "判断问题类型后分配给对应专家",      "Routing + 多个回答节点"),
        ("并行处理",         "多个任务同时执行后聚合结果",        "Send API + 聚合节点"),
        ("人在回路审批",     "关键操作需要人类确认才能继续",      "interrupt() + Command(resume=...)"),
    ]

    for name, desc, comps in cases:
        print(f"  {name}: {desc}")
        print(f"    用到: {comps}")


if __name__ == '__main__':
    print("\n>>> 什么是 LangGraph？—— 在写代码之前先建立整体认知\n")

    without_langgraph()
    what_is_langgraph()
    langgraph_vs_langchain()
    langgraph_applications()

    print(f"\n-- 概念理解完成！接下来运行 simple_graph.py 体验真正的代码")
