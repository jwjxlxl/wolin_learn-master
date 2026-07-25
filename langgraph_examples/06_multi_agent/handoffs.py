# =============================================================================
# Handoff 模式 — 控制权接力传递
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解 Handoff：控制权从一个 Agent 完全转移到下一个 Agent
#   ✅ 使用 Command(goto=...) 实现节点间的接力传递
#   ✅ 掌握与 Agent-as-Tool 的关键区别（控制权归属）
#
# 运行前检查：
# 1. 已安装依赖：pip install langgraph langchain-core langchain-ollama pydantic
# 2. 已安装 Ollama 并下载模型：ollama pull qwen3.5:2b
# 3. 示例 1 无需 LLM（纯逻辑演示，可立即运行）
# =============================================================================

import sys
import os
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 确保项目根目录可导入（用于 utils/ 共享模块）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command


# =============================================================================
# 核心概念：Handoff（交接）是什么？
# =============================================================================
"""
Handoff（控制权交接）

  场景：医院分诊系统
  患者到院后：
    分诊护士评估症状 → 转给心内科医生 → 医生看后需拍片 → 转给放射科 → 出报告 → 结束

  关键特征：
    1. 控制权完全转移：分诊护士把患者"交"给心内科后就不管了
    2. 心内科医生接手后独立决策（自己处理 or 转给放射科 or 结束）
    3. 每个 Agent 只关心"我从哪里来、我要去哪里"

  核心 API：Command(goto="目标节点", update={状态更新})
    - goto: 指定下一个要执行的节点名（或 END）
    - update: 更新 StateGraph 的状态（类似节点返回的 dict）

  与 Agent-as-Tool 的区别：
    Agent-as-Tool = 前台打电话给某部门，等回复后继续对话（前台掌控全程）
    Handoff       = 前台说"这个不归我管"，把患者直接带到下一个科室（控制权移交）

  生活化比喻：
    Agent-as-Tool = 打电话给同事问问题，挂了电话你继续处理工作
    Handoff       = 接力赛交棒，接棒的人跑你就不跑了
"""

# =============================================================================
# 示例: LLM 版 — 真正的 Handoff（LLM 决定转交方向）
# =============================================================================

def handoff_with_llm():
    """
    用 LLM 做分诊决策的 Handoff 模式。

    分诊护士用结构化输出判断去向，各科室用 LLM 做专业诊断。

    """
    print(f"\n-- 示例 2: LLM 版 — 真正的 Handoff")

    from utils.model_utils import get_model
    from pydantic import BaseModel, Field
    from typing import Literal

    model = get_model("qwen")
    if model is None:
        print("  【跳过】请安装 Ollama 并下载模型：ollama pull qwen3.5:2b")
        return

    # ===== 分诊结构化输出 =====
    class TriageDecision(BaseModel):
        department: Literal["cardiology", "neurology", "general", "end"] = Field(
            description="应该转到的科室"
        )
        reason: str = Field(description="分诊理由")

    # 定义一个带有结构化输出的模型
    triage_model = model.with_structured_output(TriageDecision)

    class HandoffState(TypedDict):
        messages: Annotated[list, add_messages]
        symptom: str
        diagnosis: str
        path: list

    def triage_nurse(state: HandoffState):
        """分诊护士：用 LLM 判断转到哪个科室。"""
        symptom = state["symptom"]
        print(f"  [分诊护士] 患者症状: {symptom}")

        decision: TriageDecision = triage_model.invoke(
            f"患者症状：{symptom}。请判断应该转到哪个科室。"
            f"可选：cardiology（心脏相关）、neurology（头痛/眩晕/神经）、"
            f"general（其他/不确定）、end（轻微症状可直接处理）。"
        )

        print(f"    判断: {decision.department} — {decision.reason}")
        return Command(goto=decision.department, update={
            "path": ["分诊"],
            "diagnosis": f"分诊理由: {decision.reason}",
        })

    def cardiology(state: HandoffState):
        """心内科：用 LLM 做初步诊断，决定是否需要拍片。"""
        symptom = state["symptom"]
        print(f"  [心内科] 接诊: {symptom}")

        response = model.invoke(
            f"你是心内科医生。患者症状：{symptom}。"
            f"请给出初步诊断，并说明是否需要转放射科拍片。"
            f"如果需要拍片，返回 'RADIOLOGY_NEEDED'，否则返回诊断结果。"
        )

        result = response.content
        if "RADIOLOGY" in result:
            print(f"    决定: 需要拍片 → 转放射科")
            return Command(goto="radiology", update={
                "path": state["path"] + ["心内科"],
                "diagnosis": f"心内科: {result}",
            })
        print(f"    诊断: {result[:80]}...")
        return Command(goto=END, update={
            "path": state["path"] + ["心内科"],
            "diagnosis": f"心内科诊断: {result}",
        })

    def neurology(state: HandoffState):
        """神经内科：用 LLM 做诊断。"""
        symptom = state["symptom"]
        print(f"  [神经内科] 接诊: {symptom}")

        response = model.invoke(
            f"你是神经内科医生。患者症状：{symptom}。请给出诊断和治疗建议。"
        )
        result = response.content
        print(f"    诊断: {result[:80]}...")
        return Command(goto=END, update={
            "path": state["path"] + ["神经内科"],
            "diagnosis": f"神经内科诊断: {result}",
        })

    def radiology(state: HandoffState):
        """放射科：用 LLM 模拟拍片报告。"""
        symptom = state["symptom"]
        print(f"  [放射科] 拍片检查")

        response = model.invoke(
            f"你是放射科医生。患者因'{symptom}'来拍片。请模拟一份 X 光/CT 检查报告。"
        )
        result = response.content
        print(f"    报告: {result[:80]}...")
        return Command(goto=END, update={
            "path": state["path"] + ["放射科"],
            "diagnosis": f"放射科报告: {result}",
        })

    def general(state: HandoffState):
        """全科医生：综合评估。"""
        symptom = state["symptom"]
        print(f"  [全科医生] 综合评估: {symptom}")

        response = model.invoke(
            f"你是全科医生。患者症状：{symptom}。请给出评估和建议。"
        )
        result = response.content
        print(f"    评估: {result[:80]}...")
        return Command(goto=END, update={
            "path": state["path"] + ["全科"],
            "diagnosis": f"全科评估: {result}",
        })

    # ===== 构建图 =====
    graph = (
        StateGraph(HandoffState)
        .add_node("triage", triage_nurse)
        .add_node("cardiology", cardiology)
        .add_node("neurology", neurology)
        .add_node("radiology", radiology)
        .add_node("general", general)
        .add_edge(START, "triage")
        .compile()
    )

    # 保存图为 PNG
    images_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images')
    os.makedirs(images_dir, exist_ok=True)
    png_path = os.path.join(images_dir, 'handoff_with_llm.png')
    with open(png_path, 'wb') as f:
        f.write(graph.get_graph().draw_mermaid_png())
    print(f"  图已保存到: {png_path}\n")

    # ===== 测试 =====
    cases = [
        "胸闷、心悸一周，偶有胸痛",
        "头痛、眩晕三天",
        "发烧 38 度，伴随咳嗽",
    ]

    for symptom in cases:
        print(f"\n【患者症状】{symptom}")
        result = graph.invoke({
            "messages": [HumanMessage(content=symptom)],
            "symptom": symptom,
            "diagnosis": "",
            "path": [],
        })
        print(f"  【诊断结果】{result['diagnosis']}")
        print(f"  【就诊路径】{' → '.join(result['path'])} → END")
        print()


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  Handoff 模式 — 控制权接力传递")
    print("  医院分诊系统：分诊护士 → 专科医生 → 放射科 → END")
    print("=" * 70 + "\n")

    handoff_with_llm()

    print("=" * 70)
    print("  回顾：Command(goto=...) 完全移交控制权，前一个 Agent 不再参与")
    print("  对比：")
    print("    Agent-as-Tool  — 主 Agent 调用子 Agent，掌控全程")
    print("    Handoff        — 控制权接力传递，交出就不管了")
    print("  接下来学习：supervisor.py（Supervisor 模式 — 主管分配+审查）")
    print("=" * 70 + "\n")
