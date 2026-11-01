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
      1. 每个技能占一个独立子目录，子目录内放 SKILL.md（技能说明 + 触发条件）
      2. 通过 skills=["./skills/"] 参数告诉 Agent 技能源目录
      3. Agent 运行时自动扫描源目录下的【子目录】，按需加载

    ⚠️ 目录结构要求（deepagents 只扫描源目录下的子目录）：
        skills_dir/
        └── code-review/       <- 技能子目录（名称须与 frontmatter 的 name 一致）
            └── SKILL.md       <- 技能定义文件
       把 SKILL.md 直接放在 skills_dir 根目录会被【静默跳过】（不报错、无提示），
       模型将完全不知道技能存在。

    ⚠️ SKILL.md frontmatter 规范（Agent Skills 标准）：
       - name：小写字母/数字/连字符，且必须与子目录名一致
       - description：同时写清"做什么"+"何时用"（模型靠它决定是否加载技能）
       - 没有 triggers 字段！触发条件要写进 description，否则会被静默丢弃

    ⚠️ Windows 必须用 FilesystemBackend(virtual_mode=True)：
       文件工具（read_file 等）拒绝 Windows 绝对路径（C:/...），
       虚拟模式下技能路径是 /code-review/SKILL.md，模型才能读到全文。

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

    # 2. 创建技能：每个技能一个子目录，SKILL.md 放子目录内
    # 注意：直接放在 skills_dir 根目录会被静默跳过（deepagents 只扫描子目录）
    skill_subdir = os.path.join(skills_dir, "code-review")
    os.makedirs(skill_subdir, exist_ok=True)

    # SKILL.md 遵循 Agent Skills 标准格式：
    #   - name 必须小写字母/数字/连字符，且与子目录名一致
    #   - description 同时写清"做什么 + 何时用"（触发条件写在这里，没有 triggers 字段）
    #   - 正文用"强制分节模板"而非松散清单：小模型（如 qwen-plus）对松散编号清单
    #     常常只答主要几点就收尾、漏掉后面的要求；改成必填小标题（缺一不可）能显著提升遵循率
    skill_content = """---
name: code-review
description: 审查代码并指出潜在 bug、改进建议。当用户请求审查代码、检查代码质量时使用。
---

# 代码审查技能

你是一个资深的代码审查员。审查任何代码时，你的回答【必须】严格包含以下三个小节，
使用给定的 Markdown 标题，且【三节缺一不可】，即使某节内容为"无"也要写出该标题并说明。

## 1. 潜在 Bug 与边界情况
指出可能出错的场景（空输入、除零、类型错误、越界等）。

## 2. 代码风格与改进方案
给出命名、文档字符串、可读性、健壮性方面的具体建议，并附改进后的代码。

## 3. 时间/空间复杂度
分别给出时间复杂度与空间复杂度（大 O 表示），并说明是否存在优化空间。
"""

    skill_file = os.path.join(skill_subdir, "SKILL.md")
    with open(skill_file, "w", encoding="utf-8") as f:
        f.write(skill_content)

    print(f"  【技能文件】{skill_file}")
    print(f"  【技能内容】code-review — 审查代码并指出问题")

    model = get_qwen_client()
    if model is None:
        print("  【跳过】请先在 .env 配置 ALIYUN_API_KEY")
        return

    # 3. 创建 Agent，加载 Skills + 文件后端
    # ⚠️ Windows 必须用 virtual_mode=True（虚拟路径，如 /code-review/SKILL.md）：
    #    deepagents 的文件工具（read_file 等）会拒绝 Windows 绝对路径（C:/...），
    #    若用 virtual_mode=False，技能虽能被发现，但模型无法读取 SKILL.md 全文
    backend = FilesystemBackend(root_dir=skills_dir, virtual_mode=True)
    agent = create_deep_agent(
        model=model,
        backend=backend,
        skills=["/"],  # 技能源 = 虚拟根目录（root_dir 即 skills_dir），自动扫描其下子目录中的 SKILL.md
        system_prompt=(
            "你是一个多功能助手。请根据用户的需求自动使用合适的技能。"
            "当用户请求审查代码时，请使用 code-review 技能。"
            "请始终用中文回答。"
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
            content=f"请审查以下代码：\n{code_to_review.strip()}，用中文回答！"
        )]
    })

    final_msg = result["messages"][-1]
    if hasattr(final_msg, 'content') and final_msg.content:
        print(f"  【回答】{final_msg.content}...")

    # 打印工具调用轨迹：技能生效时，模型会先 read_file 读取 SKILL.md 再回答
    # （渐进式披露：系统提示词只注入技能的 name/description/路径，全文按需读取）
    print(f"  【消息流】共 {len(result['messages'])} 条")
    for msg in result["messages"][1:-1]:
        for tc in getattr(msg, "tool_calls", None) or []:
            args_str = str(tc.get("args", {}))
            if len(args_str) > 80:
                args_str = args_str[:80] + "..."
            print(f"    → 工具调用: {tc['name']}({args_str})")


if __name__ == '__main__':

    skills_loading()
