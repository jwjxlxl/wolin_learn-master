from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import model_untils


# =============================================================================
# MCP (Model Context Protocol) 演示
# =============================================================================
#
# 用途：教学演示 - 理解 MCP 协议及其在 LangChain 中的使用
#
# 核心概念：
#   - MCP = Model Context Protocol（模型上下文协议）
#   - 统一的标准协议，让 AI 模型能够安全地连接外部工具和数据源
#   - 类似 "USB-C" 接口，统一了 AI 与外部世界的连接方式
# =============================================================================

# -----------------------------------------------------------------------------
# 运行前检查
# -----------------------------------------------------------------------------
# 1. 已安装 mcp 包：pip install mcp
# 2. 已安装 langchain-mcp-adapters：pip install langchain-mcp-adapters
# -----------------------------------------------------------------------------

# 检查依赖
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("提示：运行此示例需要安装 mcp 和 langchain-mcp-adapters 包")
    print("  pip install mcp langchain-mcp-adapters")


# =============================================================================
# 第一部分：理解 MCP
# =============================================================================
"""
什么是 MCP（Model Context Protocol）？

🤖 定义
   MCP = Model Context Protocol（模型上下文协议）
   由 Anthropic 提出的一种开放标准协议，用于连接 AI 模型与外部数据源和工具

🔌 类比理解
   MCP 就像 "USB-C 接口"：
   - 以前：每个设备用不同的接口（Mini-USB、Micro-USB、Lightning...）
   - 现在：统一用 USB-C，一个接口连所有设备
   - AI 领域：以前每个模型用不同方式连接工具，现在用 MCP 统一

📐 架构组成
   ┌──────────┐         MCP 协议          ┌──────────┐
   │  AI 模型  │ ←─── Transport (传输) ───→ │ MCP 服务器 │
   │ (Client) │         Tools (工具)        │ (Server) │
   │  Claude  │ ←─── Resources (资源) ────→ │  文件系统  │
   │  Qwen    │         Prompts (提示)       │  数据库    │
   └──────────┘                            └──────────┘

🎯 三种核心能力
   1. Tools（工具）：模型可调用的函数（如搜索、计算、API调用）
   2. Resources（资源）：模型可读取的数据（如文件内容、数据库记录）
   3. Prompts（提示）：预定义的交互模板（如代码审查、摘要生成）

💡 传输方式
   - Stdio（标准输入输出）：本地进程通信，适合本地工具
   - SSE（Server-Sent Events）：HTTP 流式传输，适合远程服务
   - Streamable HTTP：双向 HTTP 流，适合云端服务
"""


# =============================================================================
# 示例 1: 使用本地 MCP 服务器（stdio 方式）
# =============================================================================

def local_mcp_demo():
    """
    连接本地 MCP 服务器，将 MCP 工具转换为 LangChain 工具并用于 Agent

    这是最常见的 MCP 使用场景：通过 stdio 启动本地 MCP 服务器，
    将其中的工具加载到 LangChain Agent 中使用

    API 用法（langchain-mcp-adapters >= 0.1.0）：
      client = MultiServerMCPClient({"server_name": {"command": ..., "args": ...}})
      tools = await client.get_tools()
    """

    import asyncio

    # 实例化大模型的客户端
    model = model_untils.get_qwen_client()
    if model is None:
        print("【跳过】未配置阿里云 API Key，无法运行此示例")
        return


    async def run_with_mcp():
        try:
            # 通过 stdio 启动本地 MCP 服务器
            # command 指定启动命令，args 指定参数
            client = MultiServerMCPClient(
                {
                    "weather": {
                        "command": "python",
                        "args": [
                            os.path.join(os.path.dirname(__file__), "local_weather_server.py")
                        ],
                        "transport": "stdio",
                    },
                }
            )

            # 获取 MCP 工具（新版 API，无需 async with）
            tools = await client.get_tools()
            print(f"  从 MCP 服务器加载了 {len(tools)} 个工具")
            for t in tools:
                print(f"    - {t.name}: {t.description}")
            print()

            # 创建 Agent 并使用 MCP 工具
            agent = create_agent(
                model,
                tools=tools,
                system_prompt="你是一个智能助手，请使用工具来回答用户问题。"
            )

            question = "东京的天气怎么样？"
            print(f"【用户提问】{question}")

            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": question}]}
            )
            last_msg = result["messages"][-1]
            print(f"【Agent 回答】{last_msg.content}")

        except Exception as e:
            print(f"  注意：本地 MCP 服务连接失败: {e}")
            print("  提示：实际使用时需要确保 MCP 服务正在运行")

    asyncio.run(run_with_mcp())
    print()


# =============================================================================
# 示例 2: 使用远程 MCP 服务器（高德地图 MCP - SSE 方式）
# =============================================================================

def remote_mcp_demo():
    """
    连接高德地图官方 MCP 服务器（SSE 远程传输）

    高德地图 MCP Server 提供的工具：
    - amap_maps_geo: 地理编码（地址转经纬度）
    - amap_maps_regeo: 逆地理编码（经纬度转地址）
    - amap_maps_ip_location: IP 定位
    - amap_maps_weather: 天气查询
    - amap_maps_direction_bicycling: 骑行路径规划
    - amap_maps_direction_walking: 步行路径规划
    - amap_maps_direction_driving: 驾车路径规划
    - amap_maps_direction_transit_integrated: 公交路径规划
    - amap_maps_distance: 距离测量
    - amap_maps_search: POI 关键词搜索
    - amap_maps_around_search: POI 周边搜索

    运行前准备：
    1. 高德开放平台注册账号：https://lbs.amap.com/
    2. 创建应用 → 添加 Key，服务平台选择 "Web服务"
    3. 将 Key 设置到 .env 文件的 AMAP_KEY 变量中

    SSE 端点：https://mcp.amap.com/sse?key=AMAP_KEY
    """

    if not MCP_AVAILABLE:
        print("【跳过】缺少依赖包，请先安装：pip install mcp langchain-mcp-adapters")
        return

    import asyncio
    from dotenv import load_dotenv

    load_dotenv()
    amap_key = os.getenv("AMAP_KEY")
    print(amap_key)
    if not amap_key:
        print("【跳过】未配置 AMAP_KEY")
        print("  请在 .env 文件中添加高德地图 API Key：")
        return

    model = model_untils.get_qwen_client()
    if model is None:
        print("【跳过】未配置阿里云 API Key，无法运行此示例")
        return


    async def run_with_amap_mcp():
        try:
            # 通过 SSE 连接高德地图官方 MCP 服务器
            client = MultiServerMCPClient(
                {
                    "amap": {
                        "url": f"https://mcp.amap.com/sse?key={amap_key}",
                        "transport": "sse",
                    },
                }
            )

            # 获取高德 MCP 工具
            tools = await client.get_tools()
            print(f"  从高德 MCP 服务器加载了 {len(tools)} 个工具")
            for t in tools:
                print(f"    - {t.name}: {t.description}")
            print()

            # 创建 Agent 并使用高德地图 MCP 工具
            agent = create_agent(
                model,
                tools=tools,
                system_prompt=(
                    "你是一个地图助手，使用高德地图工具来回答用户问题。"
                    "当用户询问某地的位置时，使用地理编码工具获取经纬度；"
                    "当用户需要查天气、路线规划、地点搜索等问题时，请调用相应的高德地图工具。"
                )
            )

            questions = [
                "深圳市龙岗区富通海智科技园的经纬度是多少？",
                "今天深圳的天气怎么样？",
            ]

            for q in questions:
                print(f"{'─' * 50}")
                print(f"【用户】{q}")
                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": q}]}
                )
                last_msg = result["messages"][-1]
                print(f"【Agent 回答】{last_msg.content}")
                print()

        except Exception as e:
            print(f"  注意：高德 MCP 服务连接失败: {e}")
            print("  提示：请确保 AMAP_KEY 配置正确且网络可访问高德 MCP 服务")

    asyncio.run(run_with_amap_mcp())
    print()


# =============================================================================
# 示例 3: 同时连接多个 MCP 服务器（本地 + 远程混合）
# =============================================================================

def multi_server_mcp_demo():
    """
    同时连接本地和远程 MCP 服务器

    MultiServerMCPClient 的核心优势：
    - 一个客户端管理多个 MCP 服务器连接
    - 本地 stdio + 远程 SSE 可以混合使用
    - 所有服务器的工具自动合并到一个列表中
    """

    if not MCP_AVAILABLE:
        print("【跳过】缺少依赖包，请先安装：pip install mcp langchain-mcp-adapters")
        return

    import asyncio
    from dotenv import load_dotenv

    load_dotenv()
    amap_key = os.getenv("AMAP_KEY")
    if not amap_key:
        print("【跳过】未配置 AMAP_KEY，请先在 .env 中设置高德地图 Key")
        return

    model = model_untils.get_qwen_client()
    if model is None:
        print("【跳过】未配置阿里云 API Key，无法运行此示例")
        return


    async def run_with_multi_server():
        try:
            # 同时连接多个 MCP 服务器
            client = MultiServerMCPClient(
                {
                    # 本地服务（stdio 方式）
                    "weather": {
                        "command": "python",
                        "args": [
                            os.path.join(os.path.dirname(__file__), "local_weather_server.py")
                        ],
                        "transport": "stdio",
                    },
                    # 远程服务（SSE 方式 - 高德地图）
                    "amap": {
                        "url": f"https://mcp.amap.com/sse?key={amap_key}",
                        "transport": "sse",
                    },
                }
            )

            # 获取所有服务器的工具（自动合并）
            tools = await client.get_tools()
            print(f"  共加载 {len(tools)} 个工具")
            for t in tools:
                print(f"    - {t.name}: {t.description}")
            print()

            agent = create_agent(
                model,
                tools=tools,
                system_prompt="你是一个智能助手，可以使用可用工具来回答问题。"
            )

            question = "今天北京的天气怎么样？"
            print(f"【用户提问】{question}")

            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": question}]}
            )
            last_msg = result["messages"][-1]
            print(f"【Agent 回答】{last_msg.content}")

        except Exception as e:
            print(f"  注意：MCP 服务连接失败: {e}")
            print()
            print("  【多服务器配置格式】")
            print("  client = MultiServerMCPClient({")
            print('      "服务名A": {"command": "xxx", "args": [...], "transport": "stdio"},   # 本地')
            print('      "服务名B": {"url": "http://...", "transport": "sse"},                # 远程')
            print("  })")
            print("  tools = await client.get_tools()")

    asyncio.run(run_with_multi_server())
    print()





def github_mcp_demo():
    """
    连接 GitHub 官方 MCP 服务器，实现对 GitHub 仓库的操作

    GitHub MCP Server 是 GitHub 官方提供的 MCP 服务，支持：
    - 搜索代码和仓库
    - 创建 Issue、Pull Request
    - 查看仓库信息
    - 提交代码等

    运行前准备：
    1. 已安装 Node.js（npx 命令）
    2. 已创建 GitHub Personal Access Token
       - 访问 https://github.com/settings/tokens
       - 创建 token，勾选 repo 权限
    3. 将 token 设置到 .env 文件的 GITHUB_TOKEN 变量中

    连接方式：通过 npx 启动 @modelcontextprotocol/server-github
    """

    if not MCP_AVAILABLE:
        print("【跳过】缺少依赖包，请先安装：pip install mcp langchain-mcp-adapters")
        return

    import asyncio

    model = model_untils.get_qwen_client()
    if model is None:
        print("【跳过】未配置阿里云 API Key，无法运行此示例")
        return

    # 检查 GITHUB_TOKEN
    from dotenv import load_dotenv
    load_dotenv()
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        print("【跳过】未配置 GITHUB_TOKEN")
        print("  请在 .env 文件中添加你的 GitHub Personal Access Token：")
        print("  GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        print()
        print("  获取 Token：https://github.com/settings/tokens")
        print("  需要权限：repo (Repository)")
        return

    print("【GitHub MCP 连接演示】")
    print()

    async def run_with_github_mcp():
        try:
            # 通过 npx 启动 GitHub 官方 MCP 服务器
            # env 字段传入 GITHUB_TOKEN 认证
            client = MultiServerMCPClient(
                {
                    "github": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-github"],
                        "transport": "stdio",
                        "env": {
                            "GITHUB_TOKEN": github_token,
                        },
                    },
                }
            )

            # 获取 GitHub MCP 工具
            tools = await client.get_tools()
            print(f"  从 GitHub MCP 服务器加载了 {len(tools)} 个工具")
            for t in tools[:5]:  # 只展示前 5 个
                print(f"    - {t.name}")
            if len(tools) > 5:
                print(f"    ... 还有 {len(tools) - 5} 个工具")
            print()

            # 创建 Agent 并使用 GitHub MCP 工具
            agent = create_agent(
                model,
                tools=tools,
                system_prompt="你是一个 GitHub 助手，可以使用 GitHub 工具来回答问题。"
            )

            # 示例问题 - 查询自己的仓库
            question = "搜索一下 langchain 相关的公开仓库"
            print(f"【用户提问】{question}")

            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": question}]}
            )
            last_msg = result["messages"][-1]
            print(f"【Agent 回答】{last_msg.content}")

        except Exception as e:
            print(f"  注意：GitHub MCP 服务连接失败: {e}")
            print("  提示：确保已安装 Node.js，且 GITHUB_TOKEN 配置正确")

    asyncio.run(run_with_github_mcp())
    print()


# =============================================================================
# MCP 使用指南
# =============================================================================
"""
快速上手：

1. 安装依赖
   pip install mcp langchain-mcp-adapters

2. 启动 MCP 服务器
   # 方式一：运行已有的 MCP 服务器脚本
   python my_mcp_server.py

   # 方式二：使用 npx 运行 npm 包提供的 MCP 服务
   npx -y @modelcontextprotocol/server-filesystem /path/to/files

3. 在 LangChain 中连接 MCP 服务器

   本地 stdio 方式（langchain-mcp-adapters >= 0.1.0）：
   ┌──────────────────────────────────────────────────┐
   │ from langchain_mcp_adapters.client import        │
   │     MultiServerMCPClient                         │
   │                                                  │
   │ client = MultiServerMCPClient({                  │
   │     "my_server": {                               │
   │         "command": "python",                     │
   │         "args": ["my_mcp_server.py"],            │
   │         "transport": "stdio"                     │
   │     }                                            │
   │ })                                               │
   │                                                  │
   │ tools = await client.get_tools()                 │
   │ agent = create_agent(model, tools=tools)         │
   │ result = await agent.ainvoke({"messages": [...]})│
   └──────────────────────────────────────────────────┘

   远程 SSE 方式：
   ┌──────────────────────────────────────────────────┐
   │ client = MultiServerMCPClient({                  │
   │     "amap": {                                    │
   │         "url": "https://mcp.amap.com/sse?key=YOUR_KEY", │
   │         "transport": "sse"                       │
   │     }                                            │
   │ })                                               │
   │ tools = await client.get_tools()                 │
   └──────────────────────────────────────────────────┘

4. 常见的公开 MCP 服务器
   - 高德地图：https://mcp.amap.com/sse?key=AMAP_KEY（SSE，需要 AMAP_KEY）
   - GitHub 仓库：npx -y @modelcontextprotocol/server-github（需要 GITHUB_TOKEN）
   - 文件系统：npx -y @modelcontextprotocol/server-filesystem /path/to/files
   - 数据库：查询数据库数据
   - 浏览器：网页截图和交互
   - 搜索引擎：互联网信息检索
"""


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':

    # local_mcp_demo()
    # remote_mcp_demo()
    # multi_server_mcp_demo()
    github_mcp_demo()