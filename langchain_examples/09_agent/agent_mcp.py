# =============================================================================
# MCP Agent — 通过配置文件接入高德地图 MCP 工具
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解 MCP（Model Context Protocol）= 让 Agent 接入外部工具的标准协议
#   ✅ 用 JSON 配置文件声明 MCP Server（而不是把连接信息硬编码进代码）
#   ✅ 用 MultiServerMCPClient 加载配置，把远程工具变成 LangChain 工具
#
# 配置文件：同目录下的 mcp_config.json
#   {
#     "mcpServers": {
#       "amap": {
#         "transport": "sse",                              ← 传输方式（不是 "type"）
#         "url": "https://mcp.amap.com/sse?key=${AMAP_KEY}" ← Key 用环境变量占位符
#       }
#     }
#   }
#
# 运行前提：
#   1. pip install langchain-mcp-adapters
#   2. 在项目根目录 .env 中配置 AMAP_KEY（高德开放平台 → Web服务 类型的 Key）
# =============================================================================

import sys
import os
import json
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from utils.model_utils import get_model, setup_encoding
from utils.agent_tools import calculator

setup_encoding()

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

# 配置文件路径：与本文件同目录
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'mcp_config.json')


# =============================================================================
# 工具函数：从配置文件加载 MCP 工具
# =============================================================================

async def load_mcp_tools(config_path: str = CONFIG_PATH):
    """
    读取 mcp_config.json，连接 MCP Server，返回 LangChain 工具列表。

    三个关键步骤：
      1. json.load 读取配置文本
      2. os.path.expandvars 把 ${AMAP_KEY} 替换成 .env 里的真实 Key
         （Key 不进 git，配置文件的结构可以进 git）
      3. MultiServerMCPClient.get_tools() 连接 Server 并拉取工具清单
         （注意：这是异步方法，所以本函数和调用方都要用 async/await）
    """
    load_dotenv()

    if not os.getenv("AMAP_KEY"):
        print("提示：未配置 AMAP_KEY，跳过 MCP 示例")
        print("  获取 Key：https://lbs.amap.com/ → 创建应用 → 添加 Key → 选 Web服务")
        return []

    with open(config_path, encoding='utf-8') as f:
        raw = f.read()

    # 展开 ${AMAP_KEY} 等环境变量占位符
    config = json.loads(os.path.expandvars(raw))

    client = MultiServerMCPClient(config["mcpServers"])
    tools = await client.get_tools()

    return tools


# =============================================================================
# 示例 1: 查看 MCP Server 提供了哪些工具
# =============================================================================

async def list_mcp_tools_demo():
    """
    连接高德 MCP Server，打印它提供的工具清单。

    MCP 的价值在这里体现：我们一行高德 API 代码都没写，
    却拿到了地理编码、天气查询、路径规划等十几个现成工具。
    """
    print(f"\n-- 示例 1: 从 mcp_config.json 加载高德 MCP 工具")

    tools = await load_mcp_tools()
    if not tools:
        return []

    print(f"共加载 {len(tools)} 个工具:")
    for t in tools:
        print(f"  - {t.name}: {(t.description or '').splitlines()[0][:40]}")

    return tools


# =============================================================================
# 示例 2: MCP 工具 + 本地工具混合的 Agent
# =============================================================================

async def mcp_agent_demo(tools):
    """
    把 MCP 工具直接塞进 create_agent()——它们和本地 @tool 函数没有区别。

    这里同时混入了本地的 calculator 工具，说明：
    MCP 远程工具与本地工具可以自由组合，Agent 自己决定调哪个。
    """
    print(f"\n-- 示例 2: 高德 MCP Agent 回答天气 + 算术问题")

    model = get_model("qwen")
    agent = create_agent(
        model=model,
        tools=[*tools, calculator],
        system_prompt="你是一个有用的助手，必须根据工具的返回结果回答问题，不允许自己回答")

    question = "北京今天天气怎么样？23 加 20 的结果再乘以 10 等于多少？"
    print(f"  问: {question}")

    result = await agent.ainvoke({"messages": [HumanMessage(content=question)]})

    # 打印工具调用轨迹（否则 Agent 内部调了哪些工具对学员完全不透明）
    print("  工具调用轨迹:")
    for msg in result["messages"]:
        for call in getattr(msg, "tool_calls", None) or []:
            print(f"    → {call['name']}({call['args']})")

    print(f"  答: {result['messages'][-1].content}\n")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n>>> 09_agent/agent_mcp — 配置文件接入高德 MCP\n")

    async def main():
        mcp_tools = await list_mcp_tools_demo()
        if mcp_tools:
            await mcp_agent_demo(mcp_tools)

    # get_tools() / ainvoke() 都是异步的，入口用 asyncio.run 驱动
    asyncio.run(main())
