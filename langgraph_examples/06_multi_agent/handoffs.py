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
# 示例 1: 无 LLM 版 — 关键词路由的医院分诊（理解控制流）
# =============================================================================

def hospital_triage_demo():
    """
    医院分诊：不用 LLM，用关键词匹配展示 Command(goto=...) 的接力效果。

    START → 分诊护士 →（症状判断）
                ├── 心脏相关 → 心内科 →（需拍片？）→ 放射科 → END
                ├── 神经相关 → 神经内科 → END
                └── 其他 → 分诊护士直接处理 → END

    关键点：
      - 每个节点返回 Command(goto="下一节点", update={...})
      - 不需要 add_conditional_edges — Command 自己决定去哪
      - 控制权一旦交出去就不会回来（除非显式 goto 回来）
    """
    print(f"\n-- 示例 1: 无 LLM 版 — 关键词路由的医院分诊")

    class HandoffState(TypedDict):
        messages: Annotated[list, add_messages]
        symptom: str
        diagnosis: str
        path: list  # 记录经过的科室

    def triage_nurse(state: HandoffState):
        """
        分诊护士：根据症状关键词决定转哪个科室。
        返回 Command(goto=...) 完全交出控制权。
        """
        symptom = state["symptom"]
        print(f"  [分诊护士] 患者症状: {symptom}")

        if any(k in symptom for k in ["胸闷", "心悸", "心脏", "胸痛"]):
            print(f"    判断: 心脏相关 → 转心内科")
            return Command(goto="cardiology", update={
                "path": ["分诊"],
                "diagnosis": "疑似心脏问题，请心内科接诊",
            })
        elif any(k in symptom for k in ["头痛", "眩晕", "麻木"]):
            print(f"    判断: 神经相关 → 转神经内科")
            return Command(goto="neurology", update={
                "path": ["分诊"],
                "diagnosis": "疑似神经问题，请神经内科接诊",
            })
        elif any(k in symptom for k in ["感冒", "发烧", "咳嗽"]):
            print(f"    判断: 常见症状 → 分诊护士直接处理")
            return Command(goto=END, update={
                "path": ["分诊"],
                "diagnosis": f"建议：{symptom}属于常见症状，多休息、多喝水，必要时服药。",
            })
        else:
            print(f"    判断: 未知症状 → 转全科医生")
            return Command(goto="general", update={
                "path": ["分诊"],
                "diagnosis": "未知症状，请全科医生评估",
            })

    def cardiology(state: HandoffState):
        """
        心内科：检查后决定是否需要拍片。
        需要拍片 → 转放射科；不需要 → 直接结束。
        """
        symptom = state["symptom"]
        print(f"  [心内科] 接诊患者，症状: {symptom}")

        if any(k in symptom for k in ["胸痛", "胸闷"]):
            print(f"    判断: 需要拍片确认 → 转放射科")
            return Command(goto="radiology", update={
                "path": state["path"] + ["心内科"],
                "diagnosis": "心内科初步检查：心律不齐，需拍片确认。",
            })
        else:
            print(f"    判断: 无需拍片，直接诊断")
            return Command(goto=END, update={
                "path": state["path"] + ["心内科"],
                "diagnosis": "心内科诊断：心悸，建议休息，避免咖啡因。",
            })

    def neurology(state: HandoffState):
        """神经内科：独立处理，直接结束。"""
        symptom = state["symptom"]
        print(f"  [神经内科] 接诊患者，症状: {symptom}")
        return Command(goto=END, update={
            "path": state["path"] + ["神经内科"],
            "diagnosis": f"神经内科诊断：{symptom}，建议做脑部 CT 进一步检查。",
        })

    def radiology(state: HandoffState):
        """放射科：拍片出报告，结束。"""
        symptom = state["symptom"]
        print(f"  [放射科] 为患者拍片")
        return Command(goto=END, update={
            "path": state["path"] + ["放射科"],
            "diagnosis": "放射科报告：心脏彩超显示轻度二尖瓣反流，建议心内科复诊。",
        })

    def general(state: HandoffState):
        """全科医生：综合评估。"""
        symptom = state["symptom"]
        print(f"  [全科医生] 综合评估: {symptom}")
        return Command(goto=END, update={
            "path": state["path"] + ["全科"],
            "diagnosis": f"全科评估：{symptom}，建议专科进一步检查。",
        })

    # ===== 构建图：不需要条件边，Command(goto=...) 自己决定去哪 =====
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

    # ===== 测试不同症状 =====
    cases = [
        ("胸闷、心悸一周", "预期: 分诊 → 心内科 → 放射科 → END"),
        ("头痛、眩晕", "预期: 分诊 → 神经内科 → END"),
        ("胸痛加剧", "预期: 分诊 → 心内科 → 放射科 → END"),
        ("发烧 38 度", "预期: 分诊 → END（直接处理）"),
    ]

    for symptom, expected in cases:
        print(f"\n【患者症状】{symptom}")
        print(f"  {expected}")
        result = graph.invoke({
            "messages": [HumanMessage(content=symptom)],
            "symptom": symptom,
            "diagnosis": "",
            "path": [],
        })
        print(f"  【诊断结果】{result['diagnosis']}")
        print(f"  【就诊路径】{' → '.join(result['path'])} → END")
        print()


# =============================================================================
# 示例 2: LLM 版 — 真正的 Handoff（LLM 决定转交方向）
# =============================================================================

def handoff_with_llm():
    """
    用 LLM 做分诊决策的 Handoff 模式。

    分诊护士用结构化输出判断去向，各科室用 LLM 做专业诊断。
    架构与示例 1 相同，关键词匹配换成了 LLM 判断。
    """
    print(f"\n-- 示例 2: LLM 版 — 真正的 Handoff")

    from utils.model_utils import get_model
    from pydantic import BaseModel, Field
    from typing import Literal

    model = get_model()
    if model is None:
        print("  【跳过】请安装 Ollama 并下载模型：ollama pull qwen3.5:2b")
        return

    # ===== 分诊结构化输出 =====
    class TriageDecision(BaseModel):
        department: Literal["cardiology", "neurology", "general", "end"] = Field(
            description="应该转到的科室"
        )
        reason: str = Field(description="分诊理由")

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


    hospital_triage_demo()
    handoff_with_llm()

    print("=" * 70)
    print("  回顾：Command(goto=...) 完全移交控制权，前一个 Agent 不再参与")
    print("  对比：")
    print("    Agent-as-Tool  — 主 Agent 调用子 Agent，掌控全程")
    print("    Handoff        — 控制权接力传递，交出就不管了")
    print("  接下来学习：supervisor.py（Supervisor 模式 — 主管分配+审查）")
    print("=" * 70 + "\n")
