# =============================================================================
# 路由（Routing）— 基于 LLM 判断选择不同路径
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 用 LLM 做智能路由（非硬编码关键词匹配）
#   ✅ 使用 Literal 类型约束 LLM 的输出选项
#   ✅ 理解路由模式和条件分支的区别（路由 = LLM 做判断）
#
# 运行前检查：
# 1. 已安装依赖：pip install langgraph langchain-core langchain-ollama pydantic
# 2. 已安装 Ollama 并下载模型：ollama pull qwen3.5:2b
# 3. （可选）配置 .env 中的 ALIYUN_API_KEY 使用云端模型
# =============================================================================

import sys
import os
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 确保项目根目录可导入（用于 utils/ 共享模块）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from typing import TypedDict
from typing_extensions import Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from utils.model_utils import get_model


# =============================================================================
# 模式说明：路由 vs 条件分支
# =============================================================================
"""
路由模式 vs 条件分支

  条件分支（conditional_branch.py）：
    路由函数 = 硬编码规则（关键词匹配、数值比较等）
    → 适合逻辑明确的场景

  路由模式（本文件）：
    路由函数 = LLM 判断（用结构化输出做分类）
    → 适合需要"理解语义"的场景

  生活化比喻：
    条件分支 = 自动门（检测到人就开，规则固定）
    路由模式 = 前台接待（理解访客意图后分派到不同部门）
"""


# =============================================================================
# 示例: LLM 判断问题类型 → 选择对应角色回答
# =============================================================================

def routing_demo():
    """
    路由：LLM 判断问题类型后选择对应回答路径。

    START → router → (判断类型)
                        ├── technical → 技术专家回答 → END
                        ├── philosophical → 哲学家回答 → END
                        └── creative → 创意专家回答 → END

    关键点：
      - Literal["technical", "philosophical", "creative"] 约束 LLM 只能三选一
      - 路由决策由 LLM 完成（非硬编码关键词匹配）
      - 所有分支最终都走到 END（不走回头路）
    """
    print(f"\n-- 示例: 路由 — LLM 判断问题类型后选择回答路径")

    model = get_model()
    if model is None:
        print("  【跳过】请安装 Ollama 并下载模型：ollama pull qwen3.5:2b")
        return

    # 1. 结构化路由：让 LLM 判断问题类型
    class RouteDecision(BaseModel):
        """路由决策：判断问题属于哪种类型"""
        question_type: Literal["technical", "philosophical", "creative"] = Field(
            description="问题类型：technical(技术问题)、philosophical(哲学问题)、creative(创意问题)"
        )

    router = model.with_structured_output(RouteDecision)

    # 2. 定义状态
    class RoutingState(TypedDict):
        question: str
        question_type: str
        answer: str

    # 3. 定义节点
    def router_node(state: RoutingState):
        """路由节点：让 LLM 判断问题类型"""
        print(f"  [节点: router] 判断问题类型: '{state['question']}'")
        decision: RouteDecision = router.invoke(f"判断这个问题的类型：{state['question']}")
        q_type = decision.question_type
        print(f"    判断结果: {q_type}")
        return {"question_type": q_type}

    def technical_answer(state: RoutingState):
        """技术专家回答"""
        print(f"  [节点: technical_answer] 以技术专家身份回答")
        response = model.invoke(f"以技术专家的身份，简洁专业地回答：{state['question']}")
        return {"answer": response.content}

    def philosophical_answer(state: RoutingState):
        """哲学家回答"""
        print(f"  [节点: philosophical_answer] 以哲学家身份回答")
        response = model.invoke(f"以哲学家的身份，深入思辨地回答：{state['question']}")
        return {"answer": response.content}

    def creative_answer(state: RoutingState):
        """创意专家回答"""
        print(f"  [节点: creative_answer] 以创意专家身份回答")
        response = model.invoke(f"以创意专家的身份，富有想象力地回答：{state['question']}")
        return {"answer": response.content}

    def route_decision(state: RoutingState):
        """路由函数：根据 LLM 判断的 question_type 决定走哪个回答节点"""
        return state["question_type"]

    # 4. 构建图
    workflow = (
        StateGraph(RoutingState)
        .add_node("router", router_node)
        .add_node("technical_answer", technical_answer)
        .add_node("philosophical_answer", philosophical_answer)
        .add_node("creative_answer", creative_answer)
        .add_edge(START, "router")
        .add_conditional_edges("router", route_decision, {
            "technical": "technical_answer",
            "philosophical": "philosophical_answer",
            "creative": "creative_answer",
        })
        .add_edge("technical_answer", END)
        .add_edge("philosophical_answer", END)
        .add_edge("creative_answer", END)
        .compile()
    )

    # 5. 测试不同类型的问题
    questions = [
        "Python 中的 GIL 是什么？",                # → technical
        "人生的意义是什么？",                       # → philosophical
        "如果月亮是一块巨大的奶酪，世界会怎样？",  # → creative
    ]

    for q in questions:
        print(f"【问题】{q}")
        result = workflow.invoke({"question": q, "question_type": "", "answer": ""})
        print(f"【回答】{result['answer']}")
        print()


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  路由（Routing）— 基于 LLM 判断选择路径")
    print("  用结构化输出让 LLM 做智能分类")
    print("=" * 70 + "\n")

    print("【运行前检查】")
    print("  1. 已安装依赖：pip install langgraph langchain-core langchain-ollama pydantic")
    print("  2. 已安装 Ollama 并下载模型：ollama pull qwen3.5:2b")
    print("  3. 云端 API（可选）：代码改用 get_model(use_cloud=True)")
    print()

    routing_demo()

    print("=" * 70)
    print("  接下来学习：../05_practical/search_qa_agent.py（综合实战）")
    print("=" * 70 + "\n")
