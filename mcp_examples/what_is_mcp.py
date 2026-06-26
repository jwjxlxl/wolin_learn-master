# =============================================================================
# MCP (Model Context Protocol) — 核心概念
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解 MCP 是什么、解决什么问题
#   ✅ 掌握 MCP 的三种核心能力（Tools / Resources / Prompts）
#   ✅ 理解两种传输方式（Stdio / SSE）
#   ✅ 理解 MCP 与 Function Calling、Skill 的关系
#   ✅ 知道常见的 MCP 服务器有哪些
#
# 本文件为纯概念文件，不涉及代码实现，无需模型或 API Key。
# =============================================================================


# =============================================================================
# 一、MCP 是什么？
# =============================================================================
"""
MCP = Model Context Protocol（模型上下文协议）

🔲 USB-C 类比
   以前：每个设备用不同接口（Mini-USB、Micro-USB、Lightning...）
   现在：统一用 USB-C，一个接口连所有设备

   AI 领域以前：
   - OpenAI 用 Function Calling
   - Anthropic 用 Tool Use
   - 每个框架用自己的方式连接工具

   现在有了 MCP：
   - 一个标准协议，任何 AI 模型都能连接任何工具服务器
   - 工具提供者只需写一次 MCP Server，所有 MCP 兼容的 AI 都能用

💻 架构组成

   ┌──────────┐        MCP 协议           ┌──────────────┐
   │ AI 模型   │←────── Transport (传输) ───→│ MCP 服务器    │
   │ (Client) │        Tools (工具)         │ 文件系统      │
   │ Claude   │←────── Resources (资源) ───→│ GitHub       │
   │ Qwen     │        Prompts (提示)       │ 高德地图      │
   └──────────┘                            └──────────────┘

📦 三种核心能力

   1. Tools（工具）    — 模型可调用的函数（搜索、计算、API 调用）
      类似编程中的函数：有输入参数，返回结果

   2. Resources（资源）— 模型可读取的数据源（文件内容、数据库记录）
      类似只读的数据源：提供上下文信息

   3. Prompts（提示）  — 预定义的交互模板（代码审查、摘要生成）
      类似预设的工作流：给模型提供标准化的交互模式

🔧 两种传输方式

   1. Stdio（标准输入输出）
      - 启动本地进程，通过标准输入/输出通信
      - 适合：本地工具（文件系统、天气服务等）
      - 示例：python my_mcp_server.py

   2. SSE / HTTP（远程传输）
      - 通过 HTTP 流式传输连接远程服务
      - 适合：云端服务（高德地图、GitHub 等）
      - 示例：https://mcp.amap.com/sse?key=AMAP_KEY
"""


# =============================================================================
# 二、MCP 与 Function Calling、Skill 的关系
# =============================================================================
"""
自底向上的三层架构：

┌──────────────────────────────────────────────────────────────┐
│ Skill（应用层编排）                                           │
│ 组合多个 Tool 完成复杂任务                                      │
│ 例：旅行规划 = 查天气 + 查汇率 + 算预算                        │
├──────────────────────────────────────────────────────────────┤
│ MCP（标准化连接层）                                           │
│ 让 Tool 可跨平台复用，一个 Server 适配所有模型                  │
│ 例：高德 MCP Server 一次编写，Claude/Qwen 都能用               │
├──────────────────────────────────────────────────────────────┤
│ Function Calling（底层协议）                                  │
│ 让模型能调用外部函数的基础能力                                  │
│ 例：OpenAI bind_tools、Anthropic tool_use                    │
└──────────────────────────────────────────────────────────────┘

生活化比喻：
   Function Calling = 你"会打电话"的能力（基础技能）
   MCP              = 统一的电话网络标准（打给任何人都一样）
   Skill            = 你知道什么时候打给谁、打完后接着做什么（业务能力）
"""


# =============================================================================
# 三、常见的 MCP 服务器
# =============================================================================
"""
官方/热门 MCP 服务器一览：

| 服务器   | 启动方式 | 用途 | 需要认证 |
|--------|---------|------|---------|
| 文件系统 | npx @modelcontextprotocol/server-filesystem /path | 读写本地文件 | 否 |
| GitHub  | npx @modelcontextprotocol/server-github | 仓库操作（搜索/Issue/PR）| 需要 Token |
| 高德地图 | https://mcp.amap.com/sse?key=KEY | 地图/天气/路径规划 | 需要 AMAP_KEY |
| 浏览器   | npx @anthropic/mcp-server-browser | 网页截图/交互 | 否 |
| 数据库   | 各厂商 MCP Server | 查询数据库 | 需要连接配置 |
| 搜索引擎 | Brave Search / Google | 互联网搜索 | 需要 API Key |

在 LangChain 中使用 MCP 服务器：

  1. 安装依赖：pip install mcp langchain-mcp-adapters

  2. 连接本地 stdio 服务器：
     from langchain_mcp_adapters.client import MultiServerMCPClient

     client = MultiServerMCPClient({
         "weather": {
             "command": "python",
             "args": ["local_weather_server.py"],
             "transport": "stdio",
         }
     })
     tools = await client.get_tools()

  3. 连接远程 SSE 服务器：
     client = MultiServerMCPClient({
         "amap": {
             "url": "https://mcp.amap.com/sse?key=AMAP_KEY",
             "transport": "sse",
         }
     })
     tools = await client.get_tools()

  4. 同时连接多个服务器（工具自动合并）：
     client = MultiServerMCPClient({
         "weather": {"command": "python", "args": [...], "transport": "stdio"},
         "amap": {"url": "https://...", "transport": "sse"},
     })
     tools = await client.get_tools()  # 所有服务器的工具合并到一个列表

  5. 用于 Agent：
     agent = create_agent(model, tools=tools, system_prompt="...")
     result = await agent.ainvoke({"messages": [...]})
"""


# =============================================================================
# 四、MCP 协议内部原理
# =============================================================================
"""
MCP 是基于 JSON-RPC 2.0 的应用层协议。理解协议内部工作原理，
有助于调试连接问题、性能优化和自定义实现。

1. JSON-RPC 2.0 基础

   MCP 的消息格式遵循 JSON-RPC 2.0 规范：
   - 请求（Request）：
     {
       "jsonrpc": "2.0",
       "id": 1,
       "method": "tools/call",
       "params": {"name": "get_weather", "arguments": {"city": "北京"}}
     }

   - 响应（Response）：
     {
       "jsonrpc": "2.0",
       "id": 1,
       "result": {"content": [{"type": "text", "text": "晴，25℃"}]}
     }

   - 通知（Notification，无需回复）：
     {
       "jsonrpc": "2.0",
       "method": "notifications/initialized"
     }

2. MCP 生命周期（分三个阶段）

   阶段一：初始化（Initialization）
   ┌──────────┐                        ┌──────────┐
   │  Client  │                        │  Server  │
   └────┬─────┘                        └────┬─────┘
        │  → initialize (协议版本+客户端信息) │
        │  ← server_info (名称+版本+能力)     │
        │  → initialized (通知：初始化完成)   │

   阶段二：协商（Negotiation）
   - Client 发送 tools/list 获取可用工具列表
   - Client 发送 resources/list 获取可用资源列表
   - Client 发送 prompts/list 获取可用提示模板
   - Server 返回对应的能力清单

   阶段三：执行（Execution）
   - tools/call → 调用指定工具并返回结果
   - resources/read → 读取指定资源内容
   - prompts/get → 获取指定提示模板
   - 任意消息均可携带 _meta 进行能力扩展或传递认证信息

3. 核心消息类型一览

   | 消息方法             | 方向      | 说明                          |
   |---------------------|-----------|-------------------------------|
   | initialize          | C → S    | 握手，协商协议版本和客户端能力  |
   | initialized         | C → S    | 通知 Server 初始化完成         |
   | tools/list          | C → S    | 获取 Server 提供的所有工具      |
   | tools/call          | C → S    | 调用指定工具                  |
   | resources/list      | C → S    | 获取 Server 提供的所有资源      |
   | resources/read      | C → S    | 读取指定资源内容               |
   | resources/subscribe | C → S    | 订阅资源变更通知               |
   | prompts/list        | C → S    | 获取 Server 提供的所有提示模板  |
   | prompts/get         | C → S    | 获取指定提示模板内容           |
   | notifications/...   | S → C    | Server 主动推送（如资源变更）   |

4. 传输层差异

   - stdio（标准输入输出）：
     * 通过子进程的 stdin/stdout 传输 JSON-RPC 消息
     * 每条消息以换行符分隔，适合本地进程间通信
     * 优点：零网络开销，无需认证，适合本地工具
     * 缺点：不支持远程调用，需自行管理子进程生命周期

   - SSE（Server-Sent Events）：
     * Client 通过 HTTP GET 建立 SSE 长连接接收 Server 推送
     * Client 通过 HTTP POST 向 Server 发送 JSON-RPC 请求
     * 优点：支持远程调用，可穿透防火墙
     * 缺点：单向推送，需双通道（GET 收 + POST 发）

   - Streamable HTTP（MCP 2024-11-05 新增）：
     * 单一 HTTP 端点，支持双向流式传输
     * 优点：简化部署，无需维护双通道
     * 缺点：要求 HTTP/2 或更高版本支持

5. langchain-mcp-adapters 的桥接原理

   MultiServerMCPClient 内部做了三件事：
   (1) 为每个 MCP Server 建立连接（启动子进程或建立 SSE 会话）
   (2) 在 initialize 阶段发送 tools/list 获取工具定义
   (3) 将 MCP Tool 的 JSON Schema 自动转换为 LangChain BaseTool
       - MCP Tool.name      → BaseTool.name
       - MCP Tool.description → BaseTool.description
       - MCP inputSchema     → BaseTool.args_schema (Pydantic 动态生成)
   这样 Agent 就能像调用普通 LangChain Tool 一样调用 MCP 工具。

6. 常见调试方法

   - 查看原始 JSON-RPC 报文：设置环境变量 MCP_DEBUG=1
   - 测试 Server 是否正常：用 mcp dev 命令启动本地调试
   - 验证工具定义：在 initialize 后打印 tools/list 的返回结果
   - 检查连接状态：MultiServerMCPClient 在初始化失败时会抛出异常
"""


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  MCP (Model Context Protocol) — 核心概念")
    print("=" * 70 + "\n")

    print("MCP 是连接 AI 模型与外部工具的标准协议，类似于 USB-C 接口。")
    print("自底向上三层结构：")
    print("  Function Calling（底层）→ MCP（连接层）→ Skill（应用层）")
    print()
    print("详见上方注释文件中的完整概念说明。")
    print()
    print("=" * 70 + "\n")
