---
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
