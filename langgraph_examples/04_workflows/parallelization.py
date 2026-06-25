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
# 1. 已安装依赖：pip install langgraph langchain-core langchain-ollama
# 2. 已安装 Ollama 并下载模型：ollama pull qwen3.5:2b
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
from utils.model_utils import get_model


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
# 示例 1: Send() 基础 — 并行生成笑话
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
    print(f"\n-- 示例 1: Send() 基础 — 并行生成笑话")

    model = get_model("qwen")
    if model is None:
        print("  【跳过】请安装 Ollama 并下载模型：ollama pull qwen3.5:2b")
        return

    # 1. 定义状态
    class JokeState(TypedDict):
        topics: list[str]                     # 所有主题（输入）
        jokes: Annotated[list, operator.add]  # 收集所有笑话（自动追加）
        summary: str                          # 最终汇总

    # 2. 定义节点
    def planner(state: JokeState):
        """
        规划节点：为每个主题创建一个 Send()，派发给 generate_joke 节点。

        条件边（add_conditional_edges）负责返回 Send 列表：
          Send("generate_joke", {"topics": ["程序员"]})
          Send("generate_joke", {"topics": ["产品经理"]})
          Send("generate_joke", {"topics": ["设计师"]})

        每个 Send 会在目标节点中作为一个独立执行分支，
        传入的参数会合并到该分支的状态中。
        """
        print(f"  [节点: planner] 为 {len(state['topics'])} 个主题分配任务")
        # 节点本身不做路由，由条件边负责扇出
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
        # ★ Send 条件边：planner 节点返回 Send 列表，LangGraph 自动并行执行
        # 注意：planner 函数的返回值就是 Send 列表，不需要额外的 continue_to_jokes
        .add_conditional_edges("planner", lambda state: [Send("generate_joke", {"topics": [topic]}) for topic in state["topics"]], ["generate_joke"])
        .add_edge("generate_joke", "summarizer")
        .add_edge("summarizer", END)
        .compile()
    )

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


# =============================================================================
# 示例 2: 扇出-扇入 — 多数据源查询后聚合
# =============================================================================

def multi_source_search():
    """
    实用案例：同时查询多个"数据源"，聚合结果。

    模拟场景：一个搜索请求 → 同时搜索新闻库、百科库、论坛库 → 聚合

    图结构：
      START → dispatcher → Send → search_news
                                 → search_wiki
                                 → search_forum
                                 ↘ aggregator → END

    关键点：
      - 不需要 LLM：用模拟数据演示 Send() 的编排能力
      - 所有 search 节点同时执行
      - aggregator 等待所有 search 完成后聚合
    """
    print(f"\n-- 示例 2: 扇出-扇入 — 多数据源查询后聚合")

    from typing import Annotated
    import operator

    class SearchState(TypedDict):
        query: str
        results: Annotated[list, operator.add]  # 自动追加
        final_answer: str

    # 模拟数据源
    NEWS_DB = ["AI 技术突破：大模型进入多模态时代",
               "OpenAI 发布最新推理模型",
               "LangGraph 2.0 发布新特性"]
    WIKI_DB = ["人工智能：计算机科学的分支",
               "大语言模型：基于 Transformer 架构",
               "LangGraph：基于图结构的 Agent 框架"]
    FORUM_DB = ["网友热议：AI 会取代程序员吗？",
                "实战分享：用 LangGraph 搭建客服系统",
                "对比评测：LangChain vs LangGraph"]

    def dispatcher(state: SearchState):
        """调度节点：无操作，仅触发扇出"""
        print(f"  [节点: dispatcher] 查询: '{state['query']}'")
        return {}

    def search_news(state: SearchState):
        """搜索新闻库"""
        print(f"  [节点: search_news] 搜索中...")
        time.sleep(0.2)  # 模拟网络延迟
        matched = [r for r in NEWS_DB if any(w in r for w in state["query"].split())]
        return {"results": matched or ["（新闻库）无匹配"]}

    def search_wiki(state: SearchState):
        """搜索百科库"""
        print(f"  [节点: search_wiki] 搜索中...")
        time.sleep(0.2)
        matched = [r for r in WIKI_DB if any(w in r for w in state["query"].split())]
        return {"results": matched or ["（百科库）无匹配"]}

    def search_forum(state: SearchState):
        """搜索论坛库"""
        print(f"  [节点: search_forum] 搜索中...")
        time.sleep(0.2)
        matched = [r for r in FORUM_DB if any(w in r for w in state["query"].split())]
        return {"results": matched or ["（论坛库）无匹配"]}

    def aggregator(state: SearchState):
        """聚合节点：汇总所有搜索结果"""
        print(f"  [节点: aggregator] 聚合 {len(state['results'])} 条结果")
        return {"final_answer": f"找到 {len(state['results'])} 条相关结果：\n" +
                "\n".join(f"  - {r}" for r in state["results"])}

    def fan_out(state: SearchState):
        """扇出函数：返回 Send 列表，每个 Send 指向不同数据源"""
        # 必须把 query 传入每个 Send，否则 worker 分支拿不到原始查询
        return [
            Send("search_news", {"query": state["query"]}),
            Send("search_wiki", {"query": state["query"]}),
            Send("search_forum", {"query": state["query"]}),
        ]

    graph = (
        StateGraph(SearchState)
        .add_node("dispatcher", dispatcher)
        .add_node("search_news", search_news)
        .add_node("search_wiki", search_wiki)
        .add_node("search_forum", search_forum)
        .add_node("aggregator", aggregator)
        .add_edge(START, "dispatcher")
        # ★ 从 dispatcher 扇出到 3 个数据源（并行）
        .add_conditional_edges("dispatcher", fan_out, [
            "search_news", "search_wiki", "search_forum"
        ])
        # 所有数据源 → 聚合器（扇入）
        .add_edge("search_news", "aggregator")
        .add_edge("search_wiki", "aggregator")
        .add_edge("search_forum", "aggregator")
        .add_edge("aggregator", END)
        .compile()
    )

    start = time.time()
    result = graph.invoke({
        "query": "AI LangGraph",
        "results": [],
        "final_answer": ""
    })
    elapsed = time.time() - start

    print(f"\n{result['final_answer']}")
    print(f"  【耗时】{elapsed:.1f} 秒（3 个数据源并行查询）")
    print()


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  并行化（Parallelization）— Send() API 多节点并行执行")
    print("  Fan-out（扇出）→ 并行处理 → Fan-in（扇入）")
    print("=" * 70 + "\n")

    # 示例 2 不需要 LLM，先运行
    multi_source_search()

    # 示例 1 需要 LLM
    # parallel_joke_generation()

    print("=" * 70)
    print("  五大工作流模式已全部覆盖：")
    print("    1. 提示链（prompt_chain.py）")
    print("    2. 并行化（parallelization.py）← 本文件")
    print("    3. 路由（routing.py）")
    print("    4. 评估器-优化器（evaluator_optimizer.py）")
    print("    5. 协调器-工作者（见 Send API 进阶用法）")
    print("=" * 70 + "\n")
