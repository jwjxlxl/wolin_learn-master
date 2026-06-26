# =============================================================================
# DeepAgents 进阶 — 子代理 + Skills
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 用 Subagent 定义独立的专业子代理
#   ✅ 理解子代理的"上下文隔离"特性
#   ✅ 用 SKILL.md 文件定义可复用技能
#   ✅ 通过 skills 参数让 Agent 自动发现并加载技能
#
# 运行前检查：
# 1. 已安装依赖：pip install deepagents langgraph langchain-core
# 2. 已安装 Ollama 并下载模型：ollama pull qwen3.5:2b
# =============================================================================

import sys
import os
import io
import tempfile
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 确保项目根目录可导入（用于 utils/ 共享模块）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from langchain_core.messages import HumanMessage
from utils.model_utils import get_model


# =============================================================================
# 示例 1: 子代理基础 — 委派专业任务
# =============================================================================

def subagent_basics():
    """
    子代理 = 上下文隔离的独立 Agent。

    主 Agent 遇到专业任务时，可以委派给专门的子代理：
      - 子代理有独立的模型、工具、系统提示
      - 子代理的上下文不影响主 Agent
      - 委派完成后结果返回主 Agent

    类比：
      主 Agent = 项目经理
      子代理   = 专业外包（写完代码就交回来，不污染主对话）
    """
    print(f"\n-- 示例 1: 子代理基础 — 委派专业任务")

    try:
        from deepagents import create_deep_agent, SubAgent
        from deepagents.backends import FilesystemBackend
    except ImportError as e:
        print(f"  【跳过】请安装 deepagents：pip install deepagents{e}")
        return

    model = get_model("qwen")
    if model is None:
        print("  【跳过】请安装 Ollama 并下载模型：ollama pull qwen3.5:2b")
        return

    # 1. 定义子代理
    # 子代理是独立的 Agent，有自己的模型和指令
    # 这里用一个简化的配置（实际中可指定不同模型）
    data_analyst = SubAgent(
        name="data_analyst",
        description="分析数据并生成统计报告",
        model=model,
        system_prompt=(
            "你是一个数据分析师。当收到数据时，"
            "请分析其统计特征（平均值、最大值、最小值）并返回清晰的报告。"
        ),
    )

    # 2. 创建主 Agent，注册子代理
    agent = create_deep_agent(
        model=model,
        subagents=[data_analyst],
        system_prompt="你是项目经理。遇到数据分析任务请委派给数据分析师。",
    )

    # 3. 测试：让主 Agent 判断是否委派
    questions = [
        "请帮我分析这组数据：[12, 45, 23, 67, 34, 89, 56]",
    ]

    for q in questions:
        print(f"  【用户】{q}")
        result = agent.invoke({"messages": [HumanMessage(content=q)]})
        final_msg = result["messages"][-1]
        if hasattr(final_msg, 'content') and final_msg.content:
            print(f"  【回答】{final_msg.content[:150]}...")

        # 查看是否有子代理调用（检查工具调用中是否有子代理相关调用）
        sub_calls = [
            m for m in result["messages"]
            if hasattr(m, 'tool_calls') and m.tool_calls
            and any('sub' in tc.get('name', '').lower() or 'agent' in tc.get('name', '').lower()
                    for tc in m.tool_calls)
        ]
        if sub_calls:
            print("  【子代理委派】已委派给子代理执行")
        else:
            print("  【注】主 Agent 直接回答了问题（未委派）")
        print()


# =============================================================================
# 示例 2: Skills 加载 — 可复用行为模板
# =============================================================================

def skills_loading():
    """
    Skills = 通过 SKILL.md 文件定义的可复用行为。

    类似"插件系统"：
      1. 在目录中创建 SKILL.md 文件（定义技能说明 + 触发条件）
      2. 通过 skills=["./skills/"] 参数告诉 Agent 技能目录
      3. Agent 运行时自动发现，按需加载

    好处：
      - 技能文件化，版本控制友好
      - 多个 Agent 可共享同一技能
      - 不用重新训练，改文件即改技能
    """
    print(f"\n-- 示例 2: Skills 加载 — 可复用行为模板")

    try:
        from deepagents import create_deep_agent
        from deepagents.backends import FilesystemBackend
    except ImportError:
        print("  【跳过】请安装 deepagents：pip install deepagents")
        return

    # 1. 创建临时技能目录
    skills_dir = tempfile.mkdtemp(prefix="deepagent_skills_")
    print(f"  【技能目录】{skills_dir}")

    # 2. 创建一个简单的 SKILL.md
    # SKILL.md 遵循 Agent Skills 标准格式
    skill_content = """---
name: 代码审查
description: 审查代码并指出潜在问题、改进建议
triggers: 当用户请求审查代码、检查代码质量时
---

# 代码审查技能

你是一个资深的代码审查员。审查代码时请：
1. 指出潜在 bug 和边界情况
2. 建议代码风格和改进方案
3. 评估时间/空间复杂度
"""

    skill_file = os.path.join(skills_dir, "SKILL.md")
    with open(skill_file, "w", encoding="utf-8") as f:
        f.write(skill_content)

    print(f"  【技能文件】{skill_file}")
    print(f"  【技能内容】代码审查 — 审查代码并指出问题")

    model = get_model("qwen")
    if model is None:
        print("  【跳过】请安装 Ollama 并下载模型：ollama pull qwen3.5:2b")
        return

    # 3. 创建 Agent，加载 Skills + 文件后端
    backend = FilesystemBackend(root_dir=skills_dir, virtual_mode=False)
    agent = create_deep_agent(
        model=model,
        backend=backend,
        skills=[skills_dir],  # Agent 自动扫描 SKILL.md 文件
        system_prompt=(
            "你是一个多功能助手。请根据用户的需求自动使用合适的技能。"
            "当用户请求审查代码时，请使用代码审查技能。"
        ),
    )

    # 4. 测试：触发代码审查技能
    code_to_review = """
def calculate_average(numbers):
    return sum(numbers) / len(numbers)
"""
    print(f"  【用户】请审查以下代码：\n  {code_to_review.strip()}")
    result = agent.invoke({
        "messages": [HumanMessage(
            content=f"请审查以下代码：\n{code_to_review.strip()}"
        )]
    })

    final_msg = result["messages"][-1]
    if hasattr(final_msg, 'content') and final_msg.content:
        print(f"  【回答】{final_msg.content[:150]}...")

    print(f"  【消息流】共 {len(result['messages'])} 条")

    # 清理
    try:
        os.remove(skill_file)
        os.rmdir(skills_dir)
    except Exception:
        pass
    print()


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  DeepAgents 进阶 — 子代理 + Skills")
    print("  理解委派机制和可复用技能")
    print("=" * 70 + "\n")

    # subagent_basics()
    skills_loading()

    print("=" * 70)
    print("  总结：")
    print("    Subagent = 独立的专门 Agent，主 Agent 可委派任务")
    print("    Skills = 文件化的可复用行为，Agent 自动发现并加载")
    print("    两者结合 = 灵活、可扩展的多 Agent 架构")
    print("=" * 70 + "\n")
