"""
自建 Agent + SKILL.md 运行机制演示

⚠️ 注意：本文件不是给 Codex/OpenClaw 配置 SKILL.md。
   而是用 Python 代码自己构建一个 Agent，
   让它能读取、解析、匹配并执行 SKILL.md 文件中定义的技能。

核心流程：
  1. 扫描 skills/ 目录下所有 SKILL.md
  2. 解析 YAML 前端（name + description）
  3. 用 LLM 判断哪个 Skill 最匹配用户问题
  4. 命中 → 加载该 Skill 的 Markdown 正文 → 注入 Agent system_prompt
  5. Agent 执行（可能调用工具，也可能直接回答）

为什么要做这个（教学价值）：
  1. 理解 SKILL.md 不只是"配置文件"，而是一种技能定义格式
  2. 任何自建 Agent 都可以通过解析这种格式来动态加载能力
  3. 与 Codex/OpenClaw 的发现机制原理相同，只是由我们的 Python 代码实现

与课程其他文件的关系：
  - skill_demo.py        → Tool vs Skill 概念（system_prompt 编排 vs 函数封装）
  - skill_md_demo.py     → 如何为 Codex/OpenClaw 编写 SKILL.md（配置外部 AI）
  - 本文件（skill_agent_demo.py） → 自己构建 Agent 来加载和执行 SKILL.md

依赖：pip install pyyaml langchain langchain-core langchain-openai python-dotenv
"""

import sys
import os
import io
import yaml

# UTF-8 编码
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.model_utils import get_qwen_client, get_model

from dotenv import load_dotenv


# =============================================================================
# 核心模块 1: SKILL.md 解析器
# =============================================================================

def load_skill_md(file_path: str) -> dict:
    """解析 SKILL.md 文件，返回 {name, description, version, body, file_path}。

    SKILL.md 格式：
      ---
      name: skill-name
      description: '触发描述'
      version: 1.0.0
      ---
      # 标题
      正文内容...
    """
    with open(file_path, encoding='utf-8') as f:
        content = f.read()

    parts = content.split('---\n')
    if len(parts) < 3:
        # 尝试用 \n---\n 分隔
        parts = content.split('\n---\n')
        if len(parts) < 3:
            return {
                'name': os.path.basename(os.path.dirname(file_path)),
                'description': '',
                'version': '',
                'body': content.strip(),
                'file_path': file_path,
                'parse_ok': False,
                'error': '缺少 YAML 前端分隔符',
            }

    try:
        metadata = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as e:
        return {
            'name': os.path.basename(os.path.dirname(file_path)),
            'description': '',
            'version': '',
            'body': parts[-1].strip(),
            'file_path': file_path,
            'parse_ok': False,
            'error': f'YAML 解析失败: {e}',
        }

    return {
        'name': metadata.get('name', os.path.basename(os.path.dirname(file_path))),
        'description': metadata.get('description', ''),
        'version': metadata.get('version', ''),
        'body': parts[-1].strip(),
        'file_path': file_path,
        'parse_ok': True,
        'error': None,
    }


# =============================================================================
# 核心模块 2: Skill 发现
# =============================================================================

def discover_skills(skills_dir: str) -> list[dict]:
    """递归扫描 {skills_dir} 下所有 {dir}/SKILL.md 文件。"""
    skills = []
    for root, dirs, files in os.walk(skills_dir):
        if 'SKILL.md' in files:
            skill_path = os.path.join(root, 'SKILL.md')
            skill = load_skill_md(skill_path)
            skills.append(skill)
    return skills


# =============================================================================
# 核心模块 3: LLM 匹配
# =============================================================================

def match_skill_by_llm(model, skills: list[dict], user_query: str) -> dict | None:
    """把所有 Skill 的 name + description 发给 LLM，让它判断哪个最匹配用户问题。

    返回匹配的 skill dict，如果没有匹配的返回 None。
    """
    if not skills:
        return None

    # 构造 Skill 列表描述
    skill_list = '\n'.join(
        f"- {s['name']}: {s['description']}"
        for s in skills if s.get('parse_ok')
    )

    prompt = (
        f"用户问题：「{user_query}」\n"
        f"\n可用技能列表：\n{skill_list}\n"
        f"\n请判断哪个技能最适合回答用户问题。"
        f"如果某个技能非常匹配，请返回该技能的 name 值（只返回 name，不要其他内容）。"
        f"如果没有合适的技能，请返回：null。"
    )

    try:
        response = model.invoke(prompt)
        result = response.content.strip()

        # 解析 LLM 返回的结果
        if result.lower() in ('null', 'none', ''):
            return None

        # 匹配 name
        for s in skills:
            if s.get('name') == result:
                return s

        # 如果 LLM 返回了不在列表中的 name，尝试模糊匹配
        for s in skills:
            if s.get('name') and s['name'].lower() in result.lower():
                return s

        return None
    except Exception as e:
        print(f"  [匹配错误] {e}")
        return None


# =============================================================================
# 核心模块 4: Agent 创建
# =============================================================================

def create_skill_agent(model, tools: list, skills_dir: str, user_query: str) -> tuple:
    """创建带 SKILL.md 加载能力的 Agent。

    返回 (agent, matched_skill)，其中 matched_skill 可能为 None。
    """
    # 1. 发现所有 Skill
    skills = discover_skills(skills_dir)

    # 2. 匹配最相关的 Skill
    matched = match_skill_by_llm(model, skills, user_query)

    # 3. 构建 system_prompt
    default_prompt = "你是一个智能助手，请使用工具来回答用户问题。"

    if matched:
        system_prompt = (
            f"你是一个专业的助手。当前已加载技能「{matched['name']}」。\n"
            f"请严格按照以下技能说明来回答用户问题：\n\n"
            f"{matched['body']}\n\n"
            f"如果用户的问题超出技能范围，你可以用自己的知识回答。"
        )
    else:
        system_prompt = default_prompt

    # 4. 创建 Agent
    from langchain.agents import create_agent
    agent = create_agent(model, tools=tools, system_prompt=system_prompt)

    return agent, matched


# =============================================================================
# 示例 1: SKILL.md 解析 — 展示格式和解析过程
# =============================================================================

def demo1_parse_skill_md():
    """展示 SKILL.md 格式、解析过程、YAML 验证。"""
    print(f"\n{'=' * 70}")
    print("  示例 1: SKILL.md 解析 — 格式与结构")
    print(f"{'=' * 70}\n")

    print("""
SKILL.md 文件格式：

  ┌──────────────────────────────────────────────────┐
  │  ---                          ← YAML 前端开始    │
  │  name: weather-reporter       ← 技能名称（必填）  │
  │  description: '生成天气报告'   ← 触发描述（必填）  │
  │  version: 1.0.0              ← 版本号（可选）    │
  │  ---                          ← YAML 前端结束    │
  │                                                  │
  │  # 天气报告助手                ← Markdown 正文    │
  │  ## 角色                      ← 角色定义          │
  │  ## 工作流程                   ← 步骤指导          │
  │  ...                          ← 其他说明          │
  └──────────────────────────────────────────────────┘

关键认知：
  - name + description 决定「什么时候被触发」
  - Markdown 正文是「被触发后怎么做」的工作手册
  - 这和 Codex/OpenClaw 的 SKILL.md 格式完全一致
""")

    # 解析 skills/ 目录下的所有 Skill
    skills_dir = os.path.join(os.path.dirname(__file__), 'skills')
    if not os.path.isdir(skills_dir):
        print(f"  [跳过] 未找到 skills/ 目录: {skills_dir}")
        return

    skills = discover_skills(skills_dir)
    print(f"  在 {skills_dir} 发现 {len(skills)} 个 Skill：\n")

    for s in skills:
        status = '✅' if s['parse_ok'] else '❌'
        print(f"  {status} {s['name']}")
        print(f"     描述: {s['description'][:60]}{'...' if len(s['description']) > 60 else ''}")
        print(f"     版本: {s['version'] or '未指定'}")
        print(f"     正文: {len(s['body'].splitlines())} 行")
        if s['error']:
            print(f"     错误: {s['error']}")
        print()


# =============================================================================
# 示例 2: Skill 发现与 LLM 匹配
# =============================================================================

def demo2_discover_and_match():
    """扫描 skills/ 目录，用 LLM 演示匹配过程。"""
    print(f"\n{'=' * 70}")
    print("  示例 2: Skill 发现与 LLM 匹配")
    print(f"{'=' * 70}\n")

    model = get_model()
    if model is None:
        print("【跳过】未配置阿里云 API Key，无法运行此示例")
        return

    # 发现 Skill
    skills_dir = os.path.join(os.path.dirname(__file__), 'skills')
    skills = discover_skills(skills_dir)

    print(f"  已发现 {len(skills)} 个 Skill：")
    for s in skills:
        print(f"    - {s['name']}: {s['description'][:50]}")
    print()

    # 测试匹配
    test_queries = [
        "今天北京天气怎么样？",
        "帮我 review 一下这段 Python 代码",
        "把这句话翻译成英文：今天天气真好",
        "1 + 1 等于几？",
    ]

    for query in test_queries:
        print(f"{'─' * 50}")
        print(f"  【用户问题】{query}")
        matched = match_skill_by_llm(model, skills, query)
        if matched:
            print(f"  【匹配结果】✅ 命中 Skill「{matched['name']}」")
        else:
            print(f"  【匹配结果】❌ 没有匹配的 Skill（使用默认 prompt）")
        print()


# =============================================================================
# 示例 3: Agent + SKILL.md 运行
# =============================================================================

def demo3_skill_agent_run():
    """创建 Agent，演示命中/未命中 Skill 两种情况。"""
    print(f"\n{'=' * 70}")
    print("  示例 3: Agent + SKILL.md 运行")
    print(f"{'=' * 70}\n")

    model = get_qwen_client()
    if model is None:
        print("【跳过】未配置阿里云 API Key，无法运行此示例")
        return

    from langchain_core.messages import HumanMessage

    skills_dir = os.path.join(os.path.dirname(__file__), 'skills')

    # ---- 场景 A：命中 Skill ----
    print("【场景 A】用户问题命中 Skill")
    print(f"{'─' * 50}")
    print("  【用户】今天深圳天气怎么样？")

    agent_a, matched_a = create_skill_agent(model, [], skills_dir, "今天深圳天气怎么样？")
    if matched_a:
        print(f"  【匹配到】{matched_a['name']}")
    else:
        print(f"  【匹配到】无（使用默认 prompt）")

    try:
        result_a = agent_a.invoke({"messages": [HumanMessage(content="今天深圳天气怎么样？")]})
        print(f"  【回答】{result_a['messages'][-1].content}")
    except Exception as e:
        print(f"  【错误】{e}")
    print()

    # ---- 场景 B：未命中 Skill ----
    print("【场景 B】用户问题未命中 Skill")
    print(f"{'─' * 50}")
    print("  【用户】1 + 1 等于几？")

    agent_b, matched_b = create_skill_agent(model, [], skills_dir, "1 + 1 等于几？")
    if matched_b:
        print(f"  【匹配到】{matched_b['name']}")
    else:
        print(f"  【匹配到】无（使用默认 prompt）")

    try:
        result_b = agent_b.invoke({"messages": [HumanMessage(content="1 + 1 等于几？")]})
        print(f"  【回答】{result_b['messages'][-1].content}")
    except Exception as e:
        print(f"  【错误】{e}")
    print()

    # ---- 场景 C：命中另一个 Skill ----
    print("【场景 C】用户问题命中翻译 Skill")
    print(f"{'─' * 50}")
    print("  【用户】把「今天天气真好」翻译成英文")

    agent_c, matched_c = create_skill_agent(model, [], skills_dir, "把「今天天气真好」翻译成英文")
    if matched_c:
        print(f"  【匹配到】{matched_c['name']}")
    else:
        print(f"  【匹配到】无（使用默认 prompt）")

    try:
        result_c = agent_c.invoke({"messages": [HumanMessage(content="把「今天天气真好」翻译成英文")]})
        print(f"  【回答】{result_c['messages'][-1].content}")
    except Exception as e:
        print(f"  【错误】{e}")
    print()

    # ---- 原理总结 ----
    print(f"  {'─' * 50}")
    print("  原理总结：")
    print()
    print("  1. Agent 收到用户问题后，先扫描 skills/ 目录加载所有 SKILL.md")
    print("  2. 用 LLM 判断哪个 Skill 的 description 最匹配用户问题")
    print("  3. 命中 → 将该 Skill 的 Markdown 正文注入到 system_prompt")
    print("  4. 未命中 → 使用默认的 system_prompt")
    print("  5. Agent 带着对应的 Skill 说明去回答用户问题")
    print()
    print("  这就是 SKILL.md 的运行机制：")
    print("  技能文件（SKILL.md）→ 解析器 → 匹配器 → Agent system_prompt 注入")


# =============================================================================
# 示例 4: 交互模式
# =============================================================================

def demo4_interactive():
    """交互模式：用户输入问题，Agent 实时加载 Skill 执行。"""
    print(f"\n{'=' * 70}")
    print("  示例 4: 交互模式")
    print(f"{'=' * 70}\n")

    model = get_qwen_client()
    if model is None:
        print("【跳过】未配置阿里云 API Key，无法运行此示例")
        return

    from langchain_core.messages import HumanMessage

    skills_dir = os.path.join(os.path.dirname(__file__), 'skills')
    skills = discover_skills(skills_dir)

    print(f"  已加载 {len(skills)} 个 Skill：")
    for s in skills:
        print(f"    - {s['name']}")
    print()
    print("  输入你的问题，Agent 会自动匹配并执行对应的 Skill。")
    print("  输入 'quit' 退出。\n")

    while True:
        try:
            user_input = input("【用户】").strip()
            if not user_input:
                continue
            if user_input.lower() in ('quit', 'exit', 'q'):
                print("  再见！")
                break

            agent, matched = create_skill_agent(model, [], skills_dir, user_input)
            if matched:
                print(f"  【匹配到】{matched['name']}")
            else:
                print(f"  【匹配到】无（使用默认 prompt）")

            try:
                result = agent.invoke({"messages": [HumanMessage(content=user_input)]})
                print(f"  【回答】{result['messages'][-1].content}")
            except Exception as e:
                print(f"  【错误】{e}")
            print()
        except (KeyboardInterrupt, EOFError):
            print("\n  再见！")
            break


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  自建 Agent + SKILL.md 运行机制")
    print("=" * 70)

    # 示例 1: SKILL.md 解析（无需 API Key）
    # demo1_parse_skill_md()

    # 示例 2: Skill 发现与匹配（需要 API Key）
    # demo2_discover_and_match()

    # 示例 3: Agent + SKILL.md 运行（需要 API Key）
    # demo3_skill_agent_run()

    # 示例 4: 交互模式（需要 API Key）
    demo4_interactive()
