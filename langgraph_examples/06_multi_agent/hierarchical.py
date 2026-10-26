# =============================================================================
# Hierarchical（层级）模式 — 固定流程 + 模块化子团队
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解 Hierarchical：顶层流程固定，每个环节拆分为独立子团队（子图）
#   ✅ 用「编译好的子图」作为父图节点，实现模块化拆分
#   ✅ 区分 Hierarchical 与 Supervisor（固定流程 vs 动态分配）
#
# 运行前检查：
# 1. 已安装依赖：pip install langgraph langchain-core
# 2. 本示例是确定性演示（无需 API Key），直接运行即可：
#      python hierarchical.py
# =============================================================================

import sys
import os
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 确保项目根目录可导入（用于 utils/ 共享模块）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph_examples.utils.graph_helpers import save_graph_png


# =============================================================================
# 核心概念：Hierarchical（层级）是什么？
# =============================================================================
"""
Hierarchical（层级/分层）

  场景：公司层级管理
    总经理（顶层）不直接管理每个员工，而是：
      总经理 → 部门经理（研发部 / 市场部）→ 员工
    每个部门是一个独立团队，团队内部有自己的"经理 → 员工"层级。

  本项目案例：研究报告生产流水线
    项目经理（顶层）的流程固定不变：
      规划 → 调研团队 → 撰写团队 → 审核 → END
    其中「调研团队」「撰写团队」各是一个独立子图，
    子图内部又分为"团队主管 → 工人"，形成两级层级。

  与 Supervisor 的关键区别：
    Supervisor   = 一个主管直接管理所有工人，LLM 动态决定"下一步是谁"
    Hierarchical = 流程固定（plan→A→B→C→END），每个环节是一个模块化子团队

  关键特征：
    1. 顶层流程固定：规划 → 调研 → 撰写 → 审核，顺序不会变
    2. 模块化拆分：每个环节是一个独立子图，可单独开发、测试、替换
    3. 子图内部再分层：每个子团队有自己的主管 + 工人
    4. 数据向下流动：调研团队的产出（research_result）供撰写团队使用
"""


# =============================================================================
# 子团队 1：调研团队（子图）
# =============================================================================

def build_research_team():
    """构建「调研团队」子图：研究主管 → 资料收集员 → 分析员。"""
    # 子团队状态：只声明自己需要的字段（必须是父图状态的子集）
    class ResearchTeamState(TypedDict):
        topic: str            # 调研主题（由父图传入）
        research_result: str  # 调研结论（产出，回传给父图）

    def team_supervisor(state: ResearchTeamState):
        """研究主管：拆解调研任务，明确调研重点。"""
        print(f"    [研究主管] 收到调研任务: {state['topic']}")
        return {
            "research_result": f"调研重点：围绕「{state['topic']}」梳理行业现状、发展趋势与典型案例。"
        }

    def collector(state: ResearchTeamState):
        """资料收集员：收集行业现状资料。"""
        print("    [资料收集员] 收集行业现状资料...")
        return {
            "research_result": state["research_result"] + "\n  · 行业现状：AI 已进入课堂，覆盖个性化学习、智能批改等场景。"
        }

    def analyst(state: ResearchTeamState):
        """分析员：整理资料，形成调研结论。"""
        print("    [分析员] 整理分析，形成调研结论...")
        return {
            "research_result": state["research_result"] + "\n  · 调研结论：AI 教育的价值在于因材施教与减负增效，需关注数据安全。"
        }

    team = StateGraph(ResearchTeamState)
    team.add_node("team_supervisor", team_supervisor)
    team.add_node("collector", collector)
    team.add_node("analyst", analyst)
    team.add_edge(START, "team_supervisor")
    team.add_edge("team_supervisor", "collector")
    team.add_edge("collector", "analyst")
    team.add_edge("analyst", END)
    return team.compile()


# =============================================================================
# 子团队 2：撰写团队（子图）
# =============================================================================

def build_writing_team():
    """构建「撰写团队」子图：写作主管 → 大纲编辑 → 初稿撰写。"""
    # 撰写团队额外需要调研团队的产出（research_result）作为写作素材
    class WritingTeamState(TypedDict):
        topic: str            # 报告主题
        research_result: str  # 调研团队传入的结论（只读，用作素材）
        draft: str            # 初稿（产出）

    def writing_supervisor(state: WritingTeamState):
        """写作主管：确认撰写任务与可用素材。"""
        print(f"    [写作主管] 收到撰写任务: {state['topic']}")
        # 无需改动状态，返回空 dict 表示"只做协调、不改数据"
        return {}

    def outline_editor(state: WritingTeamState):
        """大纲编辑：拟定报告大纲。"""
        print("    [大纲编辑] 拟定报告大纲...")
        return {
            "draft": f"《{state['topic']}》研究报告\n\n一、背景\n二、行业现状\n三、发展趋势与建议"
        }

    def draft_writer(state: WritingTeamState):
        """初稿撰写：基于大纲 + 调研结论撰写正文。"""
        print("    [初稿撰写] 撰写初稿正文...")
        return {
            "draft": state["draft"] + "\n\n（正文）\n" + state["research_result"]
        }

    team = StateGraph(WritingTeamState)
    team.add_node("writing_supervisor", writing_supervisor)
    team.add_node("outline_editor", outline_editor)
    team.add_node("draft_writer", draft_writer)
    team.add_edge(START, "writing_supervisor")
    team.add_edge("writing_supervisor", "outline_editor")
    team.add_edge("outline_editor", "draft_writer")
    team.add_edge("draft_writer", END)
    return team.compile()


# =============================================================================
# 示例: Hierarchical — 研究报告生产流水线（确定性演示，无需 API Key）
# =============================================================================

def hierarchical_demo():
    """
    用「固定流程 + 模块化子团队」演示 Hierarchical 模式。

    顶层项目经理的流程固定不变：
      plan（规划）→ research_team（调研团队）→ writing_team（撰写团队）→ review（审核）→ END

    关键点：
      - 顶层只有固定边（add_edge），没有 LLM 动态路由
      - 两个子团队是独立子图，内部各自有"主管 → 工人"的层级
      - 子图直接作为父图节点注册：add_node("xx_team", 编译好的子图)
    """
    print(f"\n-- 示例: Hierarchical — 固定流程 + 模块化子团队")

    # ===== 第一步：构建两个子团队（子图） =====
    research_team = build_research_team()
    writing_team = build_writing_team()

    # ===== 第二步：定义顶层状态（是各子团队状态的超集） =====
    class ProjectState(TypedDict):
        topic: str            # 报告主题
        plan: str             # 规划产出
        research_result: str  # 调研团队的产出
        draft: str            # 撰写团队的产出
        final_report: str     # 最终报告

    def plan_node(state: ProjectState):
        """规划节点：项目经理制定执行计划（流程固定，此处只记录计划）。"""
        print(f"  [项目经理] 制定计划: {state['topic']}")
        return {"plan": "固定流程：规划 → 调研团队 → 撰写团队 → 审核"}

    def review_node(state: ProjectState):
        """审核节点：项目经理审核并定稿。"""
        print("  [项目经理] 审核定稿...")
        return {"final_report": state["draft"] + "\n\n（项目经理审核通过 ✓）"}

    # ===== 第三步：构建顶层图（固定流程） =====
    main_graph = StateGraph(ProjectState)
    main_graph.add_node("plan", plan_node)
    main_graph.add_node("research_team", research_team)  # 子图直接作为节点
    main_graph.add_node("writing_team", writing_team)    # 子图直接作为节点
    main_graph.add_node("review", review_node)

    # 固定流程：规划 → 调研团队 → 撰写团队 → 审核 → END
    main_graph.add_edge(START, "plan")
    main_graph.add_edge("plan", "research_team")
    main_graph.add_edge("research_team", "writing_team")
    main_graph.add_edge("writing_team", "review")
    main_graph.add_edge("review", END)

    graph = main_graph.compile()

    # 保存图为 PNG（复用 utils/graph_helpers.py 的 save_graph_png）
    # 嵌套子图需要 graphviz，缺失时跳过不影响演示
    try:
        save_graph_png(graph, "hierarchical.png")
    except Exception as e:
        print(f"  （保存图失败，已忽略：{e}）")

    # ===== 测试 =====
    result = graph.invoke({
        "topic": "AI 在教育中的应用",
        "plan": "",
        "research_result": "",
        "draft": "",
        "final_report": "",
    })

    print("\n" + "=" * 60)
    print("【最终报告】")
    print(result["final_report"])
    print("=" * 60)


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  Hierarchical 模式 — 固定流程 + 模块化子团队")
    print("  研究报告流水线：项目经理 → 调研团队 → 撰写团队 → 审核")
    print("=" * 70 + "\n")

    hierarchical_demo()

    print("\n" + "=" * 70)
    print("  回顾：Hierarchical = 固定流程 + 模块化子团队（子图内部再分层）")
    print("  四种多 Agent 模式对比：")
    print("    Agent-as-Tool  — 主 Agent 掌控全程，子 Agent 当工具调用")
    print("    Handoff        — 控制权接力传递，交出就不管了")
    print("    Supervisor     — 主管分配+审查，LLM 动态决定下一步")
    print("    Hierarchical   — 固定流程，每个环节是一个模块化子团队")
    print("=" * 70 + "\n")
