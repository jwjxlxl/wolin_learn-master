# =============================================================================
# MCP Resources 和 Prompts 演示
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解 MCP 的三种核心能力（Tools / Resources / Prompts）的代码区别
#   ✅ 构建提供 Resources 的 MCP Server（文件内容读取、数据库查询等）
#   ✅ 构建提供 Prompts 的 MCP Server（标准化交互模板）
#   ✅ 在 LangChain Agent 中使用 Resources 和 Prompts
#
# 运行前检查：
#   1. 已安装依赖：pip install mcp langchain-mcp-adapters
#   2. 应配合 what_is_mcp.py（概念）和 mcp_demo.py（Tool 实战）一起学习
#   3. Resources 和 Prompts 示例不需要 API Key
# =============================================================================



# =============================================================================
# 核心概念：Tool vs Resource vs Prompt
# =============================================================================
"""
MCP 的三种核心能力对比：

┌──────────┬──────────────────┬──────────────────┬──────────────────┐
│          │ Tool（工具）       │ Resource（资源）   │ Prompt（提示）    │
├──────────┼──────────────────┼──────────────────┼──────────────────┤
│ 用途     │ 让模型"做"某事     │ 让模型"读"某数据  │ 让模型"遵循"模板  │
│ 类比     │ 函数调用          │ 文件/数据库读取    │ 预设工作流        │
│ 输入     │ 参数（arguments）  │ URI（统一资源标识）│ 参数（arguments）  │
│ 输出     │ 执行结果          │ 数据内容          │ 拼装好的消息列表   │
│ 是否幂等 │ 不一定            │ 是（纯读取）       │ 是（纯查询）       │
│ 典型场景 │ 搜索、计算、API    │ 读取规章、FAQ、    │ 代码审查、摘要     │
│          │                  │ 知识库            │ 生成、客服模板     │
└──────────┴──────────────────┴──────────────────┴──────────────────┘

一句话总结：
  Tool     = "模型的手"（执行动作）
  Resource = "模型的眼睛"（读取信息）
  Prompt   = "模型的剧本"（标准化对话模式）
"""


# =============================================================================
# 示例 1：在 Agent 中直接模拟 Resources 和 Prompts（无需 MCP Server）
# =============================================================================

def demo1_without_mcp_server():
    """
    在不启动完整 MCP Server 的情况下，用 LangChain Tool 模拟
    Resources 和 Prompts 的效果，快速理解它们的区别。

    用三个 @tool 分别模拟：Tool、Resource、Prompt
    """
    from langchain.agents import create_agent
    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage
    from utils.model_utils import get_qwen_client

    print(f"\n-- 示例 1：Tool vs Resource vs Prompt 模拟对比")

    model = get_qwen_client()
    if model is None:
        print("  【跳过】请配置 ALIYUN_API_KEY")
        return

    # ---- Tool：执行动作 ----
    @tool
    def add_numbers(a: int, b: int) -> int:
        """计算两个数的和（这是 Tool：执行计算动作）"""
        return a + b

    # ---- Resource：读取数据 ----
    @tool
    def read_company_policy(topic: str) -> str:
        """读取公司规章制度（这是 Resource：纯数据读取，不修改任何内容）
        Args:
            topic: 要查询的制度主题，如 '请假'、'报销'、'加班'
        """
        policies = {
            "请假": "员工每年享有 5 天带薪年假，提前 3 天申请，紧急情况可事后补申请。",
            "报销": "差旅报销需保留发票，单次不超过 5000 元，每月 25 日前提交。",
            "加班": "加班需主管审批，工作日加班按 1.5 倍计算，周末 2 倍。",
            "考勤": "上班时间 9:00-18:00，迟到 30 分钟以内不扣薪，超过按小时扣除。",
        }
        for key, value in policies.items():
            if key in topic:
                return f"【制度查询】{key}：{value}"
        return f"未找到与 '{topic}' 相关的制度。可查询：{'、'.join(policies.keys())}"

    # ---- Prompt：标准化模板 ----
    @tool
    def get_code_review_prompt(language: str = "Python") -> str:
        """获取代码审查提示模板（这是 Prompt：提供标准化的交互模式）
        Args:
            language: 编程语言，默认 'Python'
        """
        prompt_template = f"""请对以下 {language} 代码进行审查，关注以下方面：
1. 代码正确性：逻辑是否有误？
2. 可读性：命名、注释、结构是否清晰？
3. 性能：是否有明显的性能问题？
4. 安全性：是否存在安全风险？
5. 最佳实践：是否符合 {language} 社区最佳实践？

审查结果请按以下格式输出：
- 总体评分：x/10
- 关键问题：列出 1-3 个最关键的问题
- 改进建议：给出具体的修改方案
- 优秀之处：指出代码中做得好的部分"""
        return prompt_template

    @tool
    def get_summary_prompt(content_type: str = "文章") -> str:
        """获取内容摘要提示模板（这是 Prompt：标准化输出格式）
        Args:
            content_type: 内容类型，如 '文章'、'会议记录'、'技术文档'
        """
        return f"""请对以下{content_type}进行摘要，按以下格式输出：
1. 一句话总结：用一句话概括核心内容
2. 3 个要点：列出最重要的 3 个信息点
3. 行动建议：基于内容给出 1-2 条实际建议
4. 关键词：提取 3-5 个关键词"""

    # 创建 Agent：同时拥有 Tool、Resource、Prompt 三类能力
    agent = create_agent(
        model,
        tools=[add_numbers, read_company_policy, get_code_review_prompt, get_summary_prompt],
        system_prompt=(
            "你是智能助手，拥有多种能力：\n"
            "- add_numbers：做数学计算（Tool）\n"
            "- read_company_policy：查规章制度（Resource，只读不写）\n"
            "- get_code_review_prompt：获取代码审查模板（Prompt，提供标准格式）\n"
            "- get_summary_prompt：获取摘要模板（Prompt，提供标准格式）\n"
            "请根据用户需求选择合适的能力。如果需要审查代码，先用 get_code_review_prompt 获取模板。"
        ),
    )

    # 测试：Resource 场景
    print("\n  【场景 1：Resource - 查询规章制度】")
    print("  用户：加班制度是什么？")
    r = agent.invoke({"messages": [HumanMessage(content="加班制度是什么？")]})
    print(f"  回答：{r['messages'][-1].content[:120]}...")

    # 测试：Prompt 场景
    print("\n  【场景 2：Prompt - 获取代码审查模板】")
    print("  用户：我想审查一段 Python 代码，给我一个审查清单")
    r = agent.invoke({"messages": [HumanMessage(content="我想审查一段 Python 代码，给我一个审查清单")]})
    print(f"  回答：{r['messages'][-1].content[:200]}...")

    # 测试：Resource + Tool 组合
    print("\n  【场景 3：Resource + Tool 组合】")
    print("  用户：请假制度和加班制度有什么区别？用 add 工具把年假天数加 2")
    r = agent.invoke({"messages": [HumanMessage(content="请假制度和加班制度有什么区别？用 add 工具把 5 加 2")]})
    print(f"  回答：{r['messages'][-1].content[:150]}...")

    print("\n  总结：以上分别演示了 Tool（计算）、Resource（读取制度）、Prompt（审查模板）三种能力。\n")


# =============================================================================
# 示例 2：构建提供 Resources 的 MCP Server
# =============================================================================

def demo2_build_resource_server():
    """
    用 mcp 库构建一个真正的 MCP Server，提供 Resources。

    Resources 适合：知识库查询、配置文件读取、FAQ 检索等只读数据场景。
    与 Tool 的关键区别：Resource 是"被动数据源"，不修改系统状态。
    """
    print(f"\n-- 示例 2：构建提供 Resources 的 MCP Server")
    print("  以下代码展示了如何定义一个带 Resources 的 MCP Server：")
    print()
    print("  ```python")
    print("  import asyncio")
    print("  from mcp.server.fastmcp import FastMCP")
    print()
    print("  mcp = FastMCP('KnowledgeBase')")
    print()
    print("  # 定义 Resource：公司制度（通过装饰器）")
    print("  @mcp.resource('policies://leave')")
    print("  def get_leave_policy() -> str:")
    print('      return "员工每年享有5天带薪年假..."')
    print()
    print("  @mcp.resource('policies://reimbursement')  ")
    print("  def get_reimbursement_policy() -> str:")
    print('      return "差旅报销需保留发票..."')
    print()
    print("  # 定义动态 Resource：FAQ（带参数）")
    print("  @mcp.resource('faq://{topic}')")
    print("  def get_faq(topic: str) -> str:")
    print("      faq_db = {...}")
    print("      return faq_db.get(topic, '未找到')")
    print()
    print("  if __name__ == '__main__':")
    print("      mcp.run(transport='stdio')")
    print("  ```")
    print()
    print("  然后在 LangChain Agent 中使用：")
    print("  ```python")
    print("  from langchain_mcp_adapters.client import MultiServerMCPClient")
    print()
    print("  client = MultiServerMCPClient({")
    print("      'kb': {")
    print("          'command': 'python',")
    print("          'args': ['kb_server.py'],")
    print("          'transport': 'stdio',")
    print("      }")
    print("  })")
    print()
    print("  tools = await client.get_tools()")
    print("  resources = await client.get_resources()  # 获取所有 Resource")
    print()
    print("  # 读取特定 Resource")
    print("  content = await client.read_resource('policies://leave')")
    print("  print(content)  # 输出：员工每年享有5天带薪年假...")
    print("  ```")
    print()


# =============================================================================
# 示例 3：构建提供 Prompts 的 MCP Server
# =============================================================================

def demo3_build_prompt_server():
    """
    用 mcp 库构建一个 MCP Server，提供 Prompts。

    Prompts 适合：代码审查模板、文档生成模板、标准化客服话术。
    与直接写 system_prompt 的区别：
      - MCP Prompt 是可发现的（Client 能自动列出所有模板）
      - MCP Prompt 可以带参数，自动拼装上下文
      - MCP Prompt 跨模型复用，不依赖特定模型格式
    """
    print(f"\n-- 示例 3：构建提供 Prompts 的 MCP Server")
    print("  以下代码展示了如何定义一个带 Prompts 的 MCP Server：")
    print()
    print("  ```python")
    print("  import asyncio")
    print("  from mcp.server.fastmcp import FastMCP")
    print("  from mcp.types import TextContent")
    print()
    print("  mcp = FastMCP('CodeReview')")
    print()
    print("  @mcp.prompt()")
    print("  def code_review(language: str = 'Python') -> list:")
    print('      """代码审查模板"""')
    print("      return [")
    print("          TextContent(type='text', text=f'请对以下{language}代码审查：'),")
    print("          TextContent(type='text', text='关注：正确性、可读性、性能、安全'),")
    print("          TextContent(type='text', text='输出格式：评分+问题+建议'),")
    print("      ]")
    print()
    print("  @mcp.prompt()")
    print("  def meeting_summary(participants: str = '') -> list:")
    print('      """会议纪要模板"""')
    print("      return [")
    print("          TextContent(type='text', text='请根据以下内容生成会议纪要：'),")
    print("          TextContent(type='text', text='1. 会议主题和日期'),")
    print("          TextContent(type='text', text='2. 关键讨论和决策'),")
    print("          TextContent(type='text', text='3. 后续行动项和责任人'),")
    print(f"          TextContent(type='text', text=f'参会人：{participants}'),")
    print("      ]")
    print()
    print("  if __name__ == '__main__':")
    print("      mcp.run(transport='stdio')")
    print("  ```")
    print()
    print("  在 LangChain Agent 中使用 Prompt：")
    print("  ```python")
    print("  # 获取所有可用 Prompt 模板")
    print("  prompts = await client.get_prompts()")
    print("  for p in prompts:")
    print("      print(p.name, p.description)")
    print()
    print("  # 获取特定 Prompt 的内容（已拼装好参数）")
    print("  prompt_content = await client.get_prompt('code_review', {'language': 'Python'})")
    print("  # 将 prompt_content 传入 system_prompt")
    print("  ```")
    print()


# =============================================================================
# 示例 4：完整对比 — Tool / Resource / Prompt 在同一个 MCP Server 中
# =============================================================================

def demo4_full_comparison():
    """
    在同一段代码逻辑中展示 Tool、Resource、Prompt 三者的区别，
    帮助学生建立完整的"MCP 能力地图"。
    """
    print(f"\n-- 示例 4：Tool / Resource / Prompt 完整对比")
    print("""
┌──────────────────────────────────────────────────────────────────┐
│                    同一个 Server 的三种能力                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  @mcp.tool()                                                     │
│  def search_code(query: str) -> str:                             │
│      '''搜索代码库（Tool：执行动作，有副作用？不一定）'''               │
│      return search_result                                        │
│                                                                  │
│  @mcp.resource("docs://api/{endpoint}")                          │
│  def read_api_doc(endpoint: str) -> str:                         │
│      '''读 API 文档（Resource：只读数据，无副作用）'''               │
│      return api_docs[endpoint]                                   │
│                                                                  │
│  @mcp.prompt()                                                   │
│  def api_review(endpoint: str) -> list:                          │
│      '''API 审查模板（Prompt：标准化交互模式）'''                    │
│      return [TextContent(...), TextContent(...)]                  │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│  选择规则：                                                       │
│  - 需要模型"做"某事         → Tool                               │
│  - 需要模型"读"某数据       → Resource（更轻量，无副作用）           │
│  - 需要模型"遵循"某模板     → Prompt（可发现、可组合、跨模型）       │
└──────────────────────────────────────────────────────────────────┘
""")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  MCP Resources 和 Prompts 演示")
    print("  理解 Tool / Resource / Prompt 三种能力")
    print("=" * 70 + "\n")

    print("【运行前检查】")
    print("  1. pip install langchain langchain-openai python-dotenv")
    print("  2. 配置 .env 中的 ALIYUN_API_KEY（示例 1 需要）")
    print()

    # 示例 1 需要 API Key
    try:
        demo1_without_mcp_server()
    except Exception as e:
        print(f"  【跳过示例 1】{e}")

    # 示例 2-4 不需要 API Key，始终可运行
    demo2_build_resource_server()
    demo3_build_prompt_server()
    demo4_full_comparison()

    print("=" * 70)
    print("  继续学习：mcp_demo.py（MCP Tool 实战）")
    print("=" * 70 + "\n")
