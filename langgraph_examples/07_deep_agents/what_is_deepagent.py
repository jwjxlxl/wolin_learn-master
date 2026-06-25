# =============================================================================
# DeepAgents 是什么 — 概念认知
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解 DeepAgents 的定位：LangGraph 之上的"开箱即用" Agent 框架
#   ✅ 理解 DeepAgents 与 create_react_agent() 和手动 StateGraph 的关系
#   ✅ 知道 DeepAgents 的六大核心内置能力
#   ✅ 判断何时用 DeepAgents，何时用手动构建
#
# 安装：
#   pip install deepagents
# =============================================================================


# =============================================================================
# DeepAgents 是什么？
# =============================================================================
"""
DeepAgents = "batteries-included" Agent 框架

它是 LangChain 团队在 LangGraph 之上封装的"完整 Agent 解决方案"。

层级关系：

  手动 StateGraph        ← 最底层，完全控制（03_agent_loop 学过）
        ↑
  create_react_agent()   ← 封装了标准 ReAct 循环（一行创建 Agent）
        ↑
  create_deep_agent()    ← 再封装：内置文件系统、任务规划、子代理、Skills 等

一句话：create_deep_agent() 返回的仍然是一个标准 LangGraph StateGraph，
只是它自动帮你装了所有常用工具，不用自己一个个加了。
"""


# =============================================================================
# 六大核心内置能力
# =============================================================================
"""
1. 文件系统工具
   Agent 自动拥有 ls / read_file / write_file / edit_file / glob / grep 能力
   这是 DeepAgents 的核心特色——其它 Agent 框架默认没有文件系统操作

2. 任务规划（write_todos）
   Agent 可以把复杂任务拆解为 TODO 列表，逐步完成
   适合多步骤的复杂工作

3. 子代理（Subagents）
   主 Agent 可以把 specialized 任务委派给独立的子 Agent
   子 Agent 有独立的模型、工具、上下文——"上下文隔离"

4. Skills（技能文件）
   通过 SKILL.md 文件定义可复用行为
   Agent 运行时自动发现、按需加载
   类似"插件系统"

5. 上下文工程
   内置机制管理长期会话的上下文
   支持记忆文件（memory_paths）

6. 模型灵活性
   支持任何 LangChain 支持的模型（openai:gpt-5.5、anthropic:claude-sonnet-4 等）
   与 get_model() 获取的 Ollama 本地模型兼容
"""


# =============================================================================
# DeepAgents vs 其它构建方式
# =============================================================================
"""
| 方式 | 代码量 | 内置能力 | 自定义度 | 适合场景 |
|------|--------|---------|---------|---------|
| 手动 StateGraph | ~50 行 | 无 | ★★★★★ | 需要完全控制每个节点和边 |
| create_react_agent() | 1 行 | 工具调用 | ★★★ | 标准 ReAct Agent，不需要文件操作 |
| create_deep_agent() | 1 行 | 全部内置 | ★★★ | 需要文件系统/子代理/Skills 的完整 Agent |

生活化比喻：
  手动 StateGraph     = 自己买零件组装电脑
  create_react_agent() = 买品牌整机（标准配置）
  create_deep_agent()  = 买品牌整机 + 预装所有常用软件

何时用 DeepAgents？
  ✅ 需要 Agent 读写文件
  ✅ 需要子代理委派任务
  ✅ 需要 Skills 插件系统
  ✅ 需要任务规划（TODO 列表）
  ✅ 快速搭建生产级 Agent

何时不用？
  ❌ 只需要简单工具调用（用 create_react_agent() 就够了）
  ❌ 需要精细控制图结构（手动 StateGraph）
  ❌ 教学理解底层原理（先学手动构建）
"""


# =============================================================================
# 架构一图览
# =============================================================================
"""
create_deep_agent(model, tools, ...)
  │
  ├── 内置工具
  │     ├── write_todos    （任务规划）
  │     ├── ls             （目录浏览）
  │     ├── read_file      （文件读取）
  │     ├── write_file     （文件写入）
  │     ├── edit_file      （文件编辑）
  │     ├── glob           （文件搜索）
  │     └── grep           （内容搜索）
  │
  ├── 用户自定义工具
  │     └── 通过 tools=[...] 传入
  │
  ├── 子代理
  │     └── 通过 subagents=[Subagent(...)] 传入
  │
  ├── Skills
  │     └── 通过 skills=["./skills/"] 传入 SKILL.md 目录
  │
  └── 文件系统后端
        └── 通过 backend=FilesystemBackend(root_dir="./workspace") 配置

所有组件组合后 → 返回标准 LangGraph StateGraph
  → 可以 invoke() / stream() / 接入 LangSmith
"""


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  DeepAgents 概念认知")
    print("  理解'开箱即用'Agent 框架的定位")
    print("=" * 70 + "\n")

    print("【层级关系】")
    print("  手动 StateGraph → create_react_agent() → create_deep_agent()")
    print("  （底层控制）       （一行封装）          （全套内置能力）")
    print()

    print("【六大内置能力】")
    print("  1. 文件系统工具  2. 任务规划  3. 子代理")
    print("  4. Skills       5. 上下文工程  6. 模型灵活性")
    print()

    print("【何时用？】")
    print("  需要读写文件 / 子代理委派 / Skills 插件 → DeepAgents")
    print("  只需简单工具调用 → create_react_agent()")
    print("  需要精细控制图结构 → 手动 StateGraph")
    print()

    print("=" * 70)
    print("  接下来运行：python quickstart.py 体验最简 DeepAgent")
    print("=" * 70 + "\n")
