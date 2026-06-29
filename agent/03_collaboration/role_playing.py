# =============================================================================
# 角色扮演式智能体 — Role-playing Agents（多智能体协作）
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解 Role-playing 的核心思想：拆分任务给不同角色，各司其职协作
#   ✅ 实现双角色辩论：两个角色交替发言直到达成共识
#   ✅ 实现三角色头脑风暴：三个角色围绕问题进行多轮讨论
#
# 运行前检查：
# 1. 已安装 Ollama 并下载模型：ollama pull qwen3.5:2b
# =============================================================================

import sys
import os
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 确保项目根目录和 agent/ 目录可导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from langchain_core.prompts import ChatPromptTemplate
from utils.model_utils import get_model
from helpers import format_conversation_history


# =============================================================================
# 核心概念：Role-playing Agents（角色扮演式智能体 / 多智能体协作）
# =============================================================================
"""
提出背景：源自 AutoGPT、ChatDev、CAMEL 等社区项目。

核心思想：把任务拆分给不同角色的 Agent，每个 Agent 都有专属职责，
通过对话协作完成任务。

生活化比喻：
  单 Agent = 一个人包揽产品、开发、测试 → 容易有盲区
  Role-playing = 产品经理写需求 + 程序员写代码 + 测试写用例
               = 一个团队各司其职，交叉协作

适用场景：复杂系统开发、跨职能协同、创意头脑风暴

常见架构：
  1. 双角色辩论：A 提方案 → B 反驳 → A 回应 → 达成共识
  2. 三角色头脑风暴：用户 + 设计师 + 开发者 → 多角度讨论
  3. 主管模式（Supervisor）：主管分配任务给专员 → 审查结果
"""


# =============================================================================
# 示例 1: 双角色辩论 — 交替发言直到达成共识
# =============================================================================

def two_role_debate():
    """
    双角色辩论：定义两个角色（如"产品经理"和"工程师"），
    交替发言直到达成共识或达到最大轮次。

    示例："要不要给产品加 AI 功能？"
      产品经理：支持，能提升用户体验、增加卖点
      工程师：反对，成本高、维护复杂、效果不可控
      ... → 最终共识
    """
    print(f"\n-- 示例 1: 双角色辩论 — 产品经理 vs 工程师")

    model = get_model()
    if model is None:
        print("  【跳过】请安装 Ollama 并下载模型：ollama pull qwen3.5:2b")
        return

    # 定义角色
    roles = {
        "产品经理": (
            "你是一个有远见的产品经理。你积极主张给产品加入 AI 功能，"
            "认为这能提升用户体验、增加市场竞争力和卖点。"
            "请用产品思维论证，关注用户价值和商业价值。"
        ),
        "工程师": (
            "你是一个务实的工程师。你对加 AI 功能持谨慎态度，"
            "关注技术成本、维护复杂度、效果可控性和团队能力。"
            "请用工程思维论证，关注可行性和风险。"
        ),
    }

    topic = "要不要给产品加 AI 功能？"
    max_rounds = 3

    print(f"  辩题：{topic}")
    print(f"  角色：{' | '.join(roles.keys())}")
    print(f"  轮次：最多 {max_rounds} 轮\n")

    conversation = []  # [(角色名, 发言内容), ...]

    for round_num in range(max_rounds):
        for role_name, role_system in roles.items():
            # 构建对话历史作为上下文
            if conversation:
                history_text = format_conversation_history(conversation)
            else:
                history_text = f"辩题：{topic}，请开始发表你的观点。"

            debate_prompt = ChatPromptTemplate.from_messages([
                ("system", f"{role_system}\n\n以下是当前的讨论历史：\n{history_text}\n\n请发表你的观点。注意：如果是最后一轮，请尝试提出共识方向。"),
                ("user", f"轮次 {round_num + 1}/{max_rounds}，请发表你的观点。如果这是最后一轮，请在结尾提出共识方向。"),
            ])

            response = (debate_prompt | model).invoke({})
            statement = response.content

            conversation.append((role_name, statement))

            print(f"  ── 【{role_name}】（第 {round_num + 1} 轮）──")
            for line in statement.split("\n"):
                print(f"    {line}")
            print()

    # 最终总结
    print(f"  【最终总结 — 共识方向】")
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个中立的主持人。请总结双方辩论的核心分歧，并提出一个折中的共识方案。"),
        ("user", """以下是双方辩论的内容：
{history}

请总结核心分歧，并提出一个折中的共识方案。"""),
    ])

    summary = (summary_prompt | model).invoke({
        "history": format_conversation_history(conversation),
    })
    print(f"\n  主持人总结：")
    for line in summary.content.split("\n"):
        print(f"    {line}")


# =============================================================================
# 示例 2: 三角色头脑风暴 — 多角度讨论
# =============================================================================

def three_role_brainstorm():
    """
    三角色头脑风暴：三个角色（用户、设计师、开发者）
    围绕一个问题进行多轮讨论。

    示例："设计一个学习 App 的核心功能"
      用户视角：我想要个性化学习计划
      设计师视角：界面要简洁，减少认知负担
      开发者视角：可以用 AI 生成个性化内容
    """
    print(f"\n-- 示例 2: 三角色头脑风暴 — 用户 vs 设计师 vs 开发者")

    model = get_model()
    if model is None:
        print("  【跳过】请安装 Ollama 并下载模型：ollama pull qwen3.5:2b")
        return

    # 定义角色
    roles = {
        "用户代表": (
            "你是一个普通用户，正在使用学习类 App。"
            "你关注的是：好不好用、能不能学到东西、是否有趣。"
            "请从用户体验角度发表想法。"
        ),
        "UI设计师": (
            "你是一个有 10 年经验的 UI/UX 设计师。"
            "你关注的是：界面简洁、操作流畅、视觉层次清晰、减少认知负担。"
            "请从设计角度发表想法。"
        ),
        "技术负责人": (
            "你是一个有 15 年经验的技术负责人。"
            "你关注的是：技术可行性、开发成本、维护性、可扩展性。"
            "请从技术角度发表想法。"
        ),
    }

    topic = "设计一个学习 App 的核心功能"
    max_rounds = 2

    print(f"  主题：{topic}")
    print(f"  角色：{' | '.join(roles.keys())}")
    print(f"  轮次：{max_rounds} 轮（每轮三个角色各发言一次）\n")

    conversation = []

    for round_num in range(max_rounds):
        for role_name, role_system in roles.items():
            if conversation:
                history_text = format_conversation_history(conversation[-6:])  # 只看最近两轮
            else:
                history_text = f"主题：{topic}，请开始头脑风暴。"

            brainstorm_prompt = ChatPromptTemplate.from_messages([
                ("system", f"{role_system}\n\n当前讨论：\n{history_text}\n\n请发表你的想法。关注你最擅长的领域。"),
                ("user", f"第 {round_num + 1} 轮讨论，请发表你的想法。"),
            ])

            response = (brainstorm_prompt | model).invoke({})
            statement = response.content

            conversation.append((role_name, statement))

            print(f"  ── 【{role_name}】（第 {round_num + 1} 轮）──")
            for line in statement.split("\n"):
                print(f"    {line}")
            print()

    # 汇总
    print(f"  【汇总 — 功能清单】")
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个产品负责人。请从以下讨论中提取核心功能清单。"),
        ("user", """以下是讨论内容：
{history}

请提取 3-5 个核心功能，每个功能一句话描述。格式：
1. 功能名：描述
2. 功能名：描述
..."""),
    ])

    summary = (summary_prompt | model).invoke({
        "history": format_conversation_history(conversation),
    })
    print(f"\n  核心功能清单：")
    for line in summary.content.split("\n"):
        print(f"    {line}")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  Role-playing（角色扮演式智能体 / 多智能体协作）")
    print("  不同角色各司其职，通过对话协作完成复杂任务")
    print("=" * 70 + "\n")

    two_role_debate()
    # three_role_brainstorm()

    print("\n" + "=" * 70)
    print("  协作范式学习完毕！")
    print("  Agent 开发范式模块全部完成")
    print("=" * 70 + "\n")
