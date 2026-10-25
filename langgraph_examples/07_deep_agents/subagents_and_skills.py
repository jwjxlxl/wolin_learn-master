# =============================================================================
# DeepAgents 进阶 — Skills 可复用技能
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 用 SKILL.md 文件定义可复用技能
#   ✅ 通过 skills 参数让 Agent 自动发现并加载技能
#
# 运行前检查：
# 1. 已安装依赖：pip install deepagents langgraph langchain-core
# 2. 配置 .env 中的 ALIYUN_API_KEY（使用云端 Qwen 模型）
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
from utils.model_utils import get_qwen_client


# =============================================================================
# 示例: Skills 加载 — 可复用行为模板
# =============================================================================

def skills_loading():
    """
    Skills = 通过 SKILL.md 文件定义的可复用行为。

    skills就是一组可复用的行为的指导说明书

    类似"插件系统"：
      1. 在目录中创建 SKILL.md 文件（定义技能说明 + 触发条件）
      2. 通过 skills=["./skills/"] 参数告诉 Agent 技能目录
      3. Agent 运行时自动发现，按需加载

    好处：
      - 技能文件化，版本控制友好
      - 多个 Agent 可共享同一技能
      - 不用重新训练，改文件即改技能
    """
    print(f"\n-- 示例: Skills 加载 — 可复用行为模板")

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

    model = get_qwen_client()
    if model is None:
        print("  【跳过】请先在 .env 配置 ALIYUN_API_KEY")
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


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  DeepAgents 进阶 — Skills 可复用技能")
    print("  文件化的可复用行为，Agent 自动发现并加载")
    print("=" * 70 + "\n")

    skills_loading()

    print("=" * 70)
    print("  总结：")
    print("    Skills = 文件化的可复用行为，Agent 自动发现并加载")
    print("=" * 70 + "\n")
