# =============================================================================
# 提示链（Prompt Chain）— 顺序执行 + 条件分支
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解工作流（Workflow）和智能体（Agent）的区别
#   ✅ 掌握五大工作流模式之一：提示链 + 条件分支
#   ✅ 使用 with_structured_output() 让 LLM 返回结构化数据
#   ✅ 用 Pydantic 模型定义输出格式
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
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from utils.model_utils import get_model


# =============================================================================
# 模式说明：Workflow vs Agent
# =============================================================================
"""
工作流（Workflow）vs 智能体（Agent）

  工作流 = 预定义的执行路径
    问题和解决方案可预测，按特定顺序执行
    包含：条件分支、循环、并行

  智能体 = 自主决策的执行体
    Agent 自己决定使用哪些工具、何时结束
    核心循环：思考 → 选择工具 → 执行 → 观察 → 再思考

五大常见工作流模式：
  1. 提示链（Prompt Chain）— 顺序执行 + 条件分支  ← 本文件
  2. 并行化（Parallelization）— 多个节点同时执行后聚合
  3. 路由（Routing）— LLM 判断类型后选择不同路径
  4. 协调器-工作者（Orchestrator-Worker）— 先规划，再分配任务
  5. 评估器-优化器（Evaluator-Optimizer）— 生成 → 评估 → 不满意则重试
"""


# =============================================================================
# 示例: 生成笑话 → 评估质量 → 改进 → 润色
# =============================================================================

def prompt_chain_demo():
    """
    提示链：生成笑话并评估质量，必要时改进。

    START → generate_joke → evaluate_joke → (需要改进?)
                                                ├── improve_joke → polish_joke → END
                                                └── polish_joke → END

    关键点：
      - with_structured_output() 让 LLM 返回固定格式（Pydantic 模型）
      - 每个节点函数返回的是"状态更新"（只更新部分字段）
      - 条件路由函数返回字符串，映射到对应的节点名
    """
    print(f"\n-- 示例: 提示链 — 生成笑话 → 评估 → 改进 → 润色")

    # 1. 结构化输出：让 LLM 返回固定格式的 JSON
    class JokeEvaluation(BaseModel):
        """判断笑话是否有趣的结构化输出"""
        needs_improvement: bool = Field(description="如果笑话需要改进返回 true")
        reason: str = Field(description="评价原因")

    model = get_model()
    if model is None:
        print("  【跳过】请安装 Ollama 并下载模型：ollama pull qwen3.5:2b")
        return

    evaluator = model.with_structured_output(JokeEvaluation)

    # 2. 定义状态
    class JokeState(TypedDict):
        topic: str
        joke: str
        needs_improvement: bool
        reason: str
        final_joke: str

    # 3. 定义节点函数
    def generate_joke(state: JokeState):
        """生成笑话节点"""
        print(f"  [节点: generate_joke] 为主题 '{state['topic']}' 生成笑话")
        response = model.invoke(f"写一个关于 {state['topic']} 的简短中文笑话")
        return {"joke": response.content}

    def evaluate_joke(state: JokeState):
        """评估节点：使用结构化输出判断笑话质量"""
        print(f"  [节点: evaluate_joke] 评估笑话质量")
        evaluation: JokeEvaluation = evaluator.invoke(f"评价这个笑话是否有趣：{state['joke']}")
        print(f"    需要改进: {evaluation.needs_improvement}")
        print(f"    原因: {evaluation.reason}")
        return {
            "needs_improvement": evaluation.needs_improvement,
            "reason": evaluation.reason,
        }

    def improve_joke(state: JokeState):
        """改进节点：根据反馈重新生成"""
        print(f"  [节点: improve_joke] 根据反馈改进笑话")
        response = model.invoke(
            f"改进这个笑话，使其更有趣。原笑话：{state['joke']}，反馈：{state['reason']}"
        )
        return {"joke": response.content}

    def polish_joke(state: JokeState):
        """润色节点：最终优化"""
        print(f"  [节点: polish_joke] 润色最终笑话")
        response = model.invoke(f"润色这个笑话，使其更加完美：{state['joke']}")
        return {"final_joke": response.content}

    def should_improve(state: JokeState):
        """路由函数：根据评估结果决定下一步"""
        if state["needs_improvement"]:
            print(f"  [路由] 需要改进 → 走 improve_joke 路径")
            return "improve_joke"
        else:
            print(f"  [路由] 无需改进 → 直接走 polish_joke")
            return "polish_joke"

    # 4. 构建图
    workflow = (
        StateGraph(JokeState)
        .add_node("generate_joke", generate_joke)
        .add_node("evaluate_joke", evaluate_joke)
        .add_node("improve_joke", improve_joke)
        .add_node("polish_joke", polish_joke)
        .add_edge(START, "generate_joke")
        .add_edge("generate_joke", "evaluate_joke")
        .add_conditional_edges("evaluate_joke", should_improve, {
            "improve_joke": "improve_joke",
            "polish_joke": "polish_joke",
        })
        .add_edge("improve_joke", "polish_joke")  # 改进后必须润色
        .add_edge("polish_joke", END)
        .compile()
    )

    # 5. 执行
    result = workflow.invoke({
        "topic": "程序员", "joke": "",
        "needs_improvement": False, "reason": "", "final_joke": ""
    })
    print(f"\n【最终笑话】{result['final_joke']}")
    print()


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  提示链（Prompt Chain）— 顺序执行 + 条件分支")
    print("  理解工作流（Workflow）的确定性编排")
    print("=" * 70 + "\n")

    print("【运行前检查】")
    print("  1. 已安装依赖：pip install langgraph langchain-core langchain-ollama pydantic")
    print("  2. 已安装 Ollama 并下载模型：ollama pull qwen3.5:2b")
    print("  3. 云端 API（可选）：代码改用 get_model(use_cloud=True)")
    print()

    prompt_chain_demo()

    print("=" * 70)
    print("  接下来学习：routing.py（LLM 路由）")
    print("=" * 70 + "\n")
