# -*- coding: utf-8 -*-
"""
.md Skill 教学 — 用 Markdown 文件创建可复用的 AI 技能

学习目标：
  1. 理解 .md Skill 与 LangChain Skill 的本质区别
  2. 掌握 SKILL.md 的文件结构（YAML 前端 + Markdown 正文）
  3. 亲手创建一个能被 Codex 自动发现的 Skill
  4. 理解 Skill 的渐进式加载机制

一句话对比：
  LangChain Skill = 用 Python 代码编排工具（需要写代码、调 API）
  .md Skill       = 用 Markdown 写"使用说明书"（纯文本、零依赖、跨平台）

依赖：无（不需要任何 pip 包，不需要 API Key）
"""

import sys
import os
import io
import yaml

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# =============================================================================
# 示例 1: 三种 Skill 对比 — 一张表看懂全貌
# =============================================================================

def demo1_three_skills_overview():
    """对比课程涉及的三种 Skill，帮助学生建立全景认知。"""
    print("=" * 70)
    print("  示例 1: 三种 Skill 对比 — 一张表看懂全貌")
    print("=" * 70)
    print()

    print("""
┌──────────────────────────────────────────────────────────────────────────────┐
│  维度          │ LangChain prompt编排  │ LangChain 函数封装  │ .md Skill       │
│  实现方式      │ system_prompt 文本     │ Python @tool 函数   │ Markdown 文件    │
│  需要 API Key? │ 需要                  │ 需要                │ 不需要          │
│  需要框架?     │ LangChain             │ LangChain            │ 无框架          │
│  跨平台        │ 仅 LangChain          │ 仅 LangChain         │ Codex/Cursor/Claude │
│  学习者门槛    │ 中等                  │ 中等                │ 低（会 Markdown 即可）│
│  灵活性        │ ★★★★★                │ ★★★                 │ ★★★             │
│  确定性        │ ★★                    │ ★★★★★               │ ★★★★            │
│  适合场景      │ 多 Tool 动态编排       │ 固定流程自动化       │ 领域知识注入     │
└──────────────────────────────────────────────────────────────────────────────┘
""")

    print("一句话建议：")
    print("  想让 Agent 自己决定怎么组合工具  → 用 system_prompt 编排")
    print("  想固化一个多步骤流程确保输出一致  → 用函数封装")
    print("  想给 AI 注入领域知识/编码规范     → 用 .md Skill")
    print()



# =============================================================================
# 示例 2: SKILL.md 解剖 — 打开一个真实的 Skill 看看里面有什么
# =============================================================================

def demo2_anatomy_of_skill_md():
    """以 Codex 内置的 skill-creator 为例，讲解 SKILL.md 文件结构。"""
    print("=" * 70)
    print("  示例 2: SKILL.md 解剖 — 文件结构详解")
    print("=" * 70)
    print()

    print("""
┌──────────────────────────────────────────────────────────────┐
│  SKILL.md 文件 = YAML 前端元数据 + Markdown 正文              │
│                                                              │
│  ┌──────────────────────────────────────────┐                │
│  │  YAML 前端元数据 (必填)                   │                │
│  │  name: skill-name                        │ ← Skill 名称    │
│  │  description: ...                        │ ← 触发描述(核心!)│
│  └──────────────────────────────────────────┘                │
│                                                              │
│  ┌──────────────────────────────────────────┐                │
│  │  Markdown 正文 (必填)                     │                │
│  │  # 标题                                  │ ← 技能使用说明  │
│  │  ## 工作流                               │ ← 步骤指导      │
│  │  ## 示例                                 │ ← 正反例        │
│  │  ...                                     │                │
│  └──────────────────────────────────────────┘                │
└──────────────────────────────────────────────────────────────┘
""")

    # 尝试加载真实的 skill-creator 来展示
    skill_path = os.path.join(
        os.environ.get("CODEX_HOME", os.path.join(os.path.expanduser("~"), ".codex")),
        "skills", ".system", "skill-creator", "SKILL.md"
    )
    if os.path.exists(skill_path):
        with open(skill_path, encoding="utf-8") as f:
            raw = f.read()
        parts = raw.split("---\n")
        if len(parts) >= 3:
            try:
                metadata = yaml.safe_load(parts[1])
                print(f"  真实案例：{skill_path}")
                print(f"    name:        {metadata.get("name", "N/A")}")
                desc = metadata.get("description", "")
                print(f"    description: {desc[:100]}{"..." if len(desc) > 100 else ""}")
            except yaml.YAMLError:
                pass
        body_lines = parts[-1].strip().count("\n") + 1
        print(f"    正文行数:    {body_lines} 行")
    else:
        print("  【提示】未找到 system skill-creator，跳过真实案例展示")

    print()
    print("关键认知：")
    print("  name + description = Skill 的「身份证」，决定什么时候被触发")
    print("  Markdown 正文 = Skill 的「工作手册」，被触发后才加载")
    print("  这套设计叫「渐进式加载」（Progressive Disclosure）")
    print()



# =============================================================================
# 示例 3: 动手创建你的第一个 .md Skill
# =============================================================================

GIT_COMMIT_SKILL_MD = """---
name: git-commit-helper
description: '帮助撰写规范的 Git 提交信息。当用户要求写 commit message、提交代码、或问"怎么写 commit"时使用。支持 Conventional Commits 格式 (feat/fix/docs/style/refactor/test/chore)。'
---

# Git 提交信息助手

根据代码变更的内容，生成规范、简洁的中文 Git 提交信息。

## 格式规范

遵循 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

```
<type>(<scope>): <subject>

<body>

<footer>
```

### 类型（type）

| 类型 | 说明 | 使用场景 |
|------|------|----------|
| feat | 新功能 | 新增文件、函数、模块 |
| fix | 修复 bug | 修正错误、处理异常 |
| docs | 文档 | README、注释、教程 |
| style | 格式 | 代码格式化、缩进 |
| refactor | 重构 | 重命名、提取函数 |
| test | 测试 | 添加/修改测试 |
| chore | 杂项 | 依赖更新、配置 |

### 提交信息原则

1. 主题行（subject）不超过 50 字
2. 用祈使语气："添加" 而非 "添加了"
3. 主题行不以句号结尾
4. 正文（body）解释为什么改，而非改了什么
5. 正文每行不超过 72 字

## 示例

### 好的提交信息

```
feat(rag): 添加双集合检索设计

将文档片段和问答分别存入不同 Collection，
实现混合检索策略，提升召回率约 15%。
```

```
fix(mcp): 修复 call_tool 异步调用问题

asyncio.run() 返回的是 (list[TextContent], dict) 元组，
而不是单个对象。修改索引为 result[0][0].text。
Closes #42
```

### 不好的提交信息

- "更新代码" → 太模糊，不知道更新了什么
- "修复了一个 bug" → 没说修了什么

## 工作流程

当用户要求帮忙写 commit message 时：

1. 先用 git diff --staged 或 git status 查看变更
2. 分析变更内容，确定 type 和 scope
3. 用中文编写 subject（不超过 50 字）
4. 如果变更复杂，添加 body 解释原因
5. 输出格式：先显示完整的 commit message，再提供 git commit -m "..." 命令
"""



def demo3_create_git_commit_skill():
    """亲手创建第一个 .md Skill。"""
    print("=" * 70)
    print("  示例 3: 动手创建你的第一个 .md Skill")
    print("=" * 70)
    print()

    codex_home = os.environ.get("CODEX_HOME", os.path.join(os.path.expanduser("~"), ".codex"))
    skills_dir = os.path.join(codex_home, "skills")
    skill_name = "git-commit-helper"
    skill_dir = os.path.join(skills_dir, skill_name)
    skill_file = os.path.join(skill_dir, "SKILL.md")

    print(f"  Skill 名称: {skill_name}")
    print(f"  目标路径:   {skill_dir}")
    print()

    # 尝试安装到系统 skills 目录，无权限则回退到项目目录
    installed = False
    try:
        os.makedirs(skill_dir, exist_ok=True)
        with open(skill_file, "w", encoding="utf-8") as f:
            f.write(GIT_COMMIT_SKILL_MD)
        installed = True
        print(f"  [OK] SKILL.md 已安装到系统目录: {skill_file}")
    except PermissionError:
        # 回退到项目本地
        project_skill_dir = os.path.join(os.path.dirname(__file__), "..", ".codex", "skills", skill_name)
        project_skill_dir = os.path.abspath(project_skill_dir)
        project_skill_file = os.path.join(project_skill_dir, "SKILL.md")
        os.makedirs(project_skill_dir, exist_ok=True)
        with open(project_skill_file, "w", encoding="utf-8") as f:
            f.write(GIT_COMMIT_SKILL_MD)
        installed = True
        skill_file = project_skill_file
        skill_dir = project_skill_dir
        print(f"  [OK] 系统目录无写权限，已安装到项目目录:")
        print(f"       {skill_file}")
        print(f"  如需全局生效，请手动复制到: {os.path.join(codex_home, "skills", skill_name)}\\")
    print()

    # 验证 YAML 前端（从已写入的 SKILL.md 文件读取）
    with open(skill_file, encoding="utf-8") as f:
        md_content = f.read()
    parts = md_content.split("---\n")
    if len(parts) >= 3:
        metadata = yaml.safe_load(parts[1])
        if metadata:
            print("  YAML 前端验证通过：")
            print(f"    name:        {metadata.get("name", "N/A")}")
            desc = metadata.get("description", "")
            print(f"    description: {desc[:80]}{"..." if len(desc) > 80 else ""}")
        else:
            print("  YAML 前端为空（文件可能未正确生成）")
    else:
        print("  [FAIL] SKILL.md 格式不正确，缺少 YAML 前端")

    body = GIT_COMMIT_SKILL_MD.split("---\n", 2)[-1].strip()
    body_lines = body.count("\n") + 1
    print(f"    正文长度:    {body_lines} 行")
    print()

    # 显示已安装的 Skill 列表
    # 显示已安装的 Skill 列表
    all_skills = set()
    # 系统 skills 目录
    if os.path.isdir(skills_dir):
        try:
            for d in os.listdir(skills_dir):
                if os.path.isdir(os.path.join(skills_dir, d)) and os.path.exists(os.path.join(skills_dir, d, "SKILL.md")):
                    all_skills.add(d)
        except PermissionError:
            pass
    # 项目本地 skills 目录
    project_skills = os.path.join(os.path.dirname(__file__), "..", ".codex", "skills")
    project_skills = os.path.abspath(project_skills)
    if os.path.isdir(project_skills):
        for d in os.listdir(project_skills):
            if os.path.isdir(os.path.join(project_skills, d)) and os.path.exists(os.path.join(project_skills, d, "SKILL.md")):
                all_skills.add(d + " (项目)")
    if all_skills:
        print(f"  已安装的 Skill: {', '.join(sorted(all_skills))}")
    else:
        print("  未检测到已安装的 Skill")
    print()
    print("  试试对 Codex 说：")
    print("    \"帮我把当前的修改写一个 commit message\"")
    print("  Codex 会自动加载这个 Skill 中定义的格式规范！")
    print()


def demo4_progressive_disclosure():
    """讲解 .md Skill 的核心设计原理：渐进式加载。"""
    print("=" * 70)
    print("  示例 4: 原理进阶 — 渐进式加载与触发机制")
    print("=" * 70)
    print()

    print("""
┌──────────────────────────────────────────────────────────────────┐
│  渐进式加载（Progressive Disclosure）—— 三层加载模型              │
│                                                                  │
│  ┌──────────────────────┐                                        │
│  │ 第1层：元数据          │  name + description                   │
│  │ 始终在上下文          │  约 100 字                              │
│  └──────────────────────┘                                        │
│         ↓ 触发时加载                                              │
│  ┌──────────────────────┐                                        │
│  │ 第2层：正文            │  Markdown 使用说明                      │
│  │ 触发后加载            │  < 500 行                               │
│  └──────────────────────┘                                        │
│         ↓ 按需加载                                                │
│  ┌──────────────────────┐                                        │
│  │ 第3层：附加资源        │  scripts/  references/  assets/        │
│  │ 用到时加载            │  无限制                                  │
│  └──────────────────────┘                                        │
└──────────────────────────────────────────────────────────────────┘
""")

    print("为什么这样设计？")
    print("  上下文窗口是公共资源。所有 Skill 的元数据都在窗口中，")
    print("  但正文只有在 AI 判断「这个 Skill 可能有用」后才加载。")
    print()
    print("触发条件由 description 字段控制：")
    print("  [GOOD] \"帮助撰写规范的 Git 提交信息。当用户要求写 commit 时使用。\"")
    print("  [BAD]  \"一个帮助工具\" ← 太模糊，无法准确触发")
    print()
    print("可选子目录（按需添加）：")
    print("  scripts/    — 执行脚本（Python/Bash），重复使用的代码放这里")
    print("  references/ — 参考文档，用到时才加载（如 API 文档、Schema）")
    print("  assets/     — 输出用素材（模板、图标、字体）")
    print()



# =============================================================================
# 示例 5: 全景对比总结
# =============================================================================

def demo5_full_comparison():
    """全景对比：你到底该用哪种 Skill？"""
    print("=" * 70)
    print("  示例 5: 全景对比总结 — Skill 体系全貌")
    print("=" * 70)
    print()

    print("""
┌──────────────────────────────────────────────────────────────────┐
│  Skill 体系全景                                                  │
│                                                                  │
│  Function Calling (底层协议：让模型能调用函数)                     │
│        ↑                                                         │
│  MCP (标准化连接层：让 Tool 可跨平台复用)                          │
│        ↑                                                         │
│  ┌──────────────────────────────────────┐                        │
│  │ Skill (应用层编排)                     │                        │
│  │                                      │                        │
│  │  LangChain system_prompt  ↔  函数封装  ↔  .md Skill           │
│  │  (动态灵活)              (固定流程)    (领域知识)              │
│  └──────────────────────────────────────┘                        │
└──────────────────────────────────────────────────────────────────┘
""")

    print("三种方式的选择指南：")

    print("""
  ┌──────────────────────────────────┬──────────────────────────────────┐
  │ 场景                              │ 推荐方式                          │
  ├──────────────────────────────────┼──────────────────────────────────┤
  │ Agent 自动选择工具组合             │ LangChain system_prompt 编排     │
  │ 固定流程、输出必须一致             │ LangChain 函数封装               │
  │ 给 AI 注入领域知识/编码规范        │ .md Skill                       │
  │ 跨平台共享同一套能力               │ .md Skill + MCP                  │
  └──────────────────────────────────┴──────────────────────────────────┘
""")

    print("记住两个公式就够了：")
    print("  1. Skill = Tool + 编排规则         → LangChain 体系")
    print("  2. Skill = domain knowledge in text → .md 体系")
    print()
    print("本课程已覆盖左边（skill_demo.py / gaode_skill_test.py）。")
    print("本文件让你也掌握了右边。")
    print("两者结合，你能在任何 AI 编程助手中发挥最大效能。")
    print()


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  .md Skill 教学 — 用 Markdown 文件创建可复用的 AI 技能")
    print("=" * 70)

    # 全部示例都不需要 API Key，可直接运行
    # demo1_three_skills_overview()
    # demo2_anatomy_of_skill_md()
    demo3_create_git_commit_skill()
    # demo4_progressive_disclosure()
    # demo5_full_comparison()
