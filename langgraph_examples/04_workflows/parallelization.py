# =============================================================================
# 并行化（Parallelization）— 多节点同时执行后聚合
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解 Send() API 的工作原理：动态创建并行分支
#   ✅ 实现扇出（Fan-out）：一个节点分发任务到多个 Worker
#   ✅ 实现扇入（Fan-in）：收集所有 Worker 结果后聚合
#   ✅ 对比顺序执行和并行执行的效率差异
#
# 运行前检查：
# 1. 已安装依赖：pip install langgraph langchain-core langchain-openai
# 2. 配置 .env 中的 ALIYUN_API_KEY（使用云端 Qwen 模型）
# =============================================================================

import sys
import os
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 确保项目根目录可导入（用于 utils/ 共享模块）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import time
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from utils.model_utils import get_qwen_client


# =============================================================================
# 核心概念：Send() API — LangGraph 的并行利器
# =============================================================================
"""
什么是 Send() ？

  Send() 是 LangGraph 中实现"条件扇出"的 API。

  正常边（Edge）：       A → B（1 对 1，固定目标）
  条件边（Conditional）： A → (判断) → B 或 C（1 对 1，动态选择）
  Send()：               A → [B, B, B, ...]（1 对 N，动态数量）

  工作流程：
    1. 一个节点返回 Send() 列表（每个 Send 指定目标节点和参数）
    2. LangGraph 为每个 Send 创建一个并行执行分支
    3. 所有分支完成后，结果自动合并到状态中

  生活化比喻：
    Edge         = 老板 → 一个员工（顺序执行）
    Conditional  = 老板 → (判断技能) → 设计师 或 程序员（选择路径）
    Send()       = 老板 → [员工A, 员工B, 员工C]（同时开工，都完成后汇总）

  典型场景：
    - 对多篇文档同时做摘要
    - 同时查询多个数据源后合并结果
    - 对一个问题生成多个候选答案后投票
"""


# =============================================================================
# 示例: Send() 基础 — 并行生成笑话
# =============================================================================

def parallel_joke_generation():
    """
    演示 Send() 的基本用法：为多个主题并行生成笑话。

    图结构：
      START → planner（分派任务）→ 条件边 Send → generate_joke × 3（并行）
                                                      ↘ summarizer（汇总）→ END

    关键点：
      - 条件边返回 [Send("generate_joke", {"topics": [t]}) for t in topics]
      - 3 个 generate_joke 节点可以同时运行
      - summarizer 在所有 generate_joke 完成后才执行
    """
    print(f"\n-- 示例: Send() 基础 — 并行生成笑话")

    model = get_qwen_client()
    if model is None:
        print("  【跳过】请先在 .env 配置 ALIYUN_API_KEY")
        return

    # 1. 定义状态
    class JokeState(TypedDict):
        topics: list[str]                     # 所有主题（输入）
        jokes: Annotated[list, operator.add]  # 收集所有笑话（自动追加）
        summary: str                          # 最终汇总

    # 2. 定义节点
    def planner(state: JokeState):
        """规划节点：本身只打印提示；真正的扇出由下方 add_conditional_edges 里的 Send 列表完成。"""
        print(f"  [节点: planner] 为 {len(state['topics'])} 个主题分配任务")
        return {}

    def generate_joke(state: JokeState):
        """Worker 节点：为分配到的主题生成笑话"""
        topic = state["topics"][0]
        print(f"  [节点: generate_joke] 为主题 '{topic}' 生成笑话...")
        time.sleep(0.3)  # 模拟 LLM 调用延迟
        response = model.invoke(f"写一个关于 {topic} 的简短中文笑话，20字以内。")
        joke = f"[{topic}] {response.content}"
        print(f"    完成: {joke[:40]}...")
        return {"jokes": [joke]}

    def summarizer(state: JokeState):
        """汇总节点：所有并行分支完成后，汇总结果"""
        jokes = state.get("jokes", [])
        if not jokes:
            print(f"  [节点: summarizer] 等待笑话生成...")
            return {"summary": "（无笑话可汇总）"}
        print(f"  [节点: summarizer] 汇总 {len(jokes)} 个笑话")
        response = model.invoke(
            f"以下是几个笑话，请给一个幽默的总评（15字以内）：\n" +
            "\n".join(jokes)
        )
        return {"summary": response.content}

    # 3. 条件扇出：从 planner 返回 Send 列表（条件边直接定义在图中）

    # 4. 构建图
    graph = (
        StateGraph(JokeState)
        .add_node("planner", planner)
        .add_node("generate_joke", generate_joke)
        .add_node("summarizer", summarizer)
        .add_edge(START, "planner")
        # ★ Send 条件边：这里的 lambda 返回 Send 列表，LangGraph 为每个 Send 开一个并行分支
        .add_conditional_edges("planner", lambda state: [Send("generate_joke", {"topics": [topic]}) for topic in state["topics"]], ["generate_joke"])
        .add_edge("generate_joke", "summarizer")
        .add_edge("summarizer", END)
        .compile()
    )

    # 保存图为 PNG
    images_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images')
    os.makedirs(images_dir, exist_ok=True)
    png_path = os.path.join(images_dir, 'parallelization.png')
    with open(png_path, 'wb') as f:
        f.write(graph.get_graph().draw_mermaid_png())
    print(f"  图已保存到: {png_path}\n")

    # 5. 执行
    print("  【测试：3 个主题并行生成笑话】")
    start = time.time()
    result = graph.invoke({
        "topics": ["程序员", "产品经理", "设计师"],
        "jokes": [],
        "summary": ""
    })
    elapsed = time.time() - start

    print(f"\n  【结果】")
    for joke in result["jokes"]:
        print(f"    {joke}")
    print(f"  【总评】{result['summary']}")
    print(f"  【耗时】{elapsed:.1f} 秒（并行执行，3 个任务几乎同时完成）")
    print()


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  并行化（Parallelization）— Send() API 多节点并行执行")
    print("  Fan-out（扇出）→ 并行处理 → Fan-in（扇入）")
    print("=" * 70 + "\n")

    parallel_joke_generation()

    print("=" * 70)
    print("  五大工作流模式已全部覆盖：")
    print("    1. 提示链（prompt_chain.py）")
    print("    2. 并行化（parallelization.py）← 本文件")
    print("    3. 路由（routing.py）")
    print("    4. 评估器-优化器（evaluator_optimizer.py）")
    print("    5. 协调器-工作者（见 Send API 进阶用法）")
    print("=" * 70 + "\n")
