# MCP 教学模块

Model Context Protocol（模型上下文协议）—— 让 AI 模型连接任何工具的标准方式。

---

## 学习路线图

```
概念理解 ──→ Tool 实战 ──→ Resources & Prompts ──→ 企业实战 ──→ Skill 编排 ──→ 多 Agent ──→ A2A 通信
   │              │                  │                  │              │             │            │
what_is_mcp   mcp_demo         mcp_resources_     enterprise_     skill_demo    multiple_     a2a_demo
.py           .py              prompts_demo.py      api_mcp_       .py           agent.py      .py
                                                    demo.py
                                                            │
                                                  客户端配置指南
                                                  mcp_client_config
                                                  _guide.py
```

**建议学习顺序**（标注 `*` 为核心文件）：

| 阶段 | 文件 | 内容 | 需要 API | 预计时间 |
|:----:|------|------|:--------:|:-------:|
| 1 | `what_is_mcp.py` | MCP 核心概念、三层架构、协议原理 | 否 | 15 min |
| 2* | `mcp_demo.py` | 本地 stdio / 远程 SSE / 多服务混合 / GitHub | 是 | 20 min |
| 3 | `mcp_resources_prompts_demo.py` | Tool vs Resource vs Prompt 对比 | 是 | 15 min |
| 4 | `mcp_client_config_guide.py` | Claude Desktop / Cursor / Codex 配置指南 | 否 | 15 min |
| 5 | `skill_demo.py` | Skill 是什么、两种实现方式、对比演示 | 是 | 15 min |
| 6* | `enterprise_api_mcp_demo.py` | 企业 REST API → FastMCP 封装 → Agent 接入 | 是 | 20 min |
| 7 | `gaode_skill_test.py` | 高德地图 6 大场景 + 智能旅游规划 | 是 | 20 min |
| 8 | `multiple_agent.py` | Supervisor 多 Agent 调度模式 | 是 | 15 min |
| 9* | `a2a_demo.py` | A2A 协议：Agent 之间通过 HTTP/JSON-RPC 通信 | 否 | 20 min |

---

## 核心概念

### MCP 三层架构

```
┌────────────────────────────────────────────────┐
│ Skill（应用层编排）                               │
│ 组合多个 Tool 完成复杂任务                        │
│ 例：旅行规划 = 查天气 + 查汇率 + 算预算            │
├────────────────────────────────────────────────┤
│ MCP（标准化连接层）                               │
│ 一个 Server 适配所有模型                          │
│ 例：高德 MCP Server 一次编写，Claude/Qwen 都能用 │
├────────────────────────────────────────────────┤
│ Function Calling（底层协议）                      │
│ 让模型能调用外部函数的基础能力                     │
│ 例：OpenAI bind_tools、Anthropic tool_use       │
└────────────────────────────────────────────────┘
```

### 三种核心能力

| 能力 | 类比 | 用途 | 典型场景 |
|------|------|------|----------|
| **Tool**（工具） | 模型的手 | 让模型"做"某事 | 搜索、计算、API 调用 |
| **Resource**（资源） | 模型的眼睛 | 让模型"读"某数据 | 读取规章、FAQ、知识库 |
| **Prompt**（提示） | 模型的剧本 | 让模型"遵循"模板 | 代码审查、摘要生成 |

### 两种传输方式

| 方式 | 通信机制 | 适用场景 | 示例 |
|------|---------|---------|------|
| **Stdio** | 本地进程 stdin/stdout | 本地工具 | `python local_weather_server.py` |
| **SSE** | HTTP Server-Sent Events | 云端服务 | `https://mcp.amap.com/sse?key=KEY` |

---

## 文件说明

### 📖 概念层

#### `what_is_mcp.py` — MCP 核心概念
纯概念文件，通过 USB-C 类比理解 MCP 解决的问题。涵盖：
- MCP 架构组成（Client ↔ Server）
- 三种核心能力详解（Tool / Resource / Prompt）
- 与 Function Calling、Skill 的关系
- 常见 MCP 服务器列表（文件系统、GitHub、高德、浏览器、数据库、搜索）
- MCP 协议内部原理（JSON-RPC 2.0、生命周期、消息类型、传输层差异）
- `langchain-mcp-adapters` 桥接原理

#### `mcp_client_config_guide.py` — 客户端配置指南
在三种流行客户端中配置 MCP Server 的完整指南：
- **Claude Desktop**：全局 `claude_desktop_config.json` 配置
- **Cursor**：项目级 `.cursor/mcp.json` 配置
- **Codex**：插件系统 + `settings.json` 配置
- 三种客户端对比表
- 5 个常见问题排查方案

### 🔧 实战层

#### `local_weather_server.py` — 最简 MCP Server
仅 50 行的教学用 MCP Server，提供两个 Tool：
- `get_weather(city)` — 天气查询
- `get_air_quality(city)` — 空气质量查询

#### `mcp_demo.py` — MCP Tool 实战 ⭐
四个阶段递进学习：
1. **入门**：本地 stdio 连接天气 Server
2. **进阶**：远程 SSE 连接高德地图官方 Server
3. **实战**：本地 + 远程多服务混合编排
4. **扩展**：GitHub 官方 MCP Server 连接

#### `mcp_resources_prompts_demo.py` — Resource 与 Prompt 演示
- 用 LangChain Tool 模拟 Resource 和 Prompt 效果（无需完整 MCP Server）
- 构建提供 Resources 的 MCP Server（`@mcp.resource` 装饰器）
- 构建提供 Prompts 的 MCP Server（`@mcp.prompt` 装饰器）
- Tool / Resource / Prompt 完整对比表

#### `enterprise_api_mcp_demo.py` — 企业 API → MCP Server 实战 ⭐
完整演示如何把企业内部 REST API 封装为 MCP Server：
- **第一步**：`OrderAPI` 类模拟企业内部订单系统（订单查询/取消、库存检查、客户信息）
- **第二步**：用 `FastMCP` 定义 6 个 Tool（最小权限 + Pydantic 输入校验）
- **第三步**：两种方式接入 LangChain Agent（直接调用 vs AI 自动决策）
- **第四步**：企业部署建议（认证鉴权、权限控制、日志审计、速率限制、stdio/SSE 部署）

### 🎯 进阶层

#### `skill_demo.py` — Skill 技能演示
理解 Skill 是什么、与 Tool 的区别：
- **对比演示**：Tool（螺丝刀）vs Skill（维修技能）
- **方式 A（推荐）**：通过 `system_prompt` 编排多个 Tool 形成 Skill
- **方式 B**：封装函数作为 Skill，内部调用多个子 Tool
- **价值对比**：没有 Skill 编排的 Agent vs 有 Skill 编排的 Agent

#### `gaode_skill_test.py` — 高德地图综合实战
覆盖高德官方 Skill 6 大场景 + 1 个组合 Skill：

| 场景 | Tool | 说明 |
|------|------|------|
| 地理编码 | `amap_geo` | 地址 → 经纬度 |
| 逆地理编码 | `amap_regeo` | 经纬度 → 地址 |
| POI 关键词搜索 | `amap_poi_search` | 搜索地点/商家/景点 |
| POI 周边搜索 | `amap_around_search` | 指定坐标周边搜索 |
| 天气查询 | `amap_weather` | 城市实时天气 |
| 路径规划 | `amap_route` | 步行/驾车/骑行/公交 |
| 地图链接 | `amap_search_url` | 生成高德地图搜索链接 |
| 智能旅游规划 | `amap_travel_planner` | 组合多个 API 的旅游规划 Skill |

#### `multiple_agent.py` — 多 Agent Supervisor 模式
一个主 Agent（Supervisor）负责规划分发，三个子 Agent 各负责细分领域：
- 计算专家（Calculator Agent）
- 翻译专家（Translator Agent）
- 写作专家（Writer Agent）

使用 `langgraph-supervisor` 的 `create_supervisor` 实现。

#### `a2a_demo.py` — A2A 协议实战 ⭐
Google 提出的 Agent 间通信协议（HTTP/JSON-RPC），通过"旅游规划三剑客"场景演示：
- **天气 Agent**（模拟 HTTP 8001 端口）：提供天气查询
- **酒店 Agent**（模拟 HTTP 8002 端口）：提供酒店搜索
- **旅行协调员**：通过 A2A 协议发现并调用上述两个 Agent，合成旅行计划

核心教学内容：
- **Agent Card 发现机制**：每个 Agent 通过名片描述自己的能力
- **Task 生命周期**：SUBMITTED → WORKING → COMPLETED
- **JSON-RPC 2.0 消息格式**：真实 A2A 协议的通信格式
- **MCP vs A2A 对比**：MCP 是模型连工具（USB-C），A2A 是 Agent 连 Agent（打电话）
- **7 个递进示例**：概念讲解 → Agent Card 发现 → 单 Agent 调用 → 多 Agent 协作 → LLM 增强 → 交互模式

**无需 API Key 即可运行核心示例（1-5）**，适合课堂演示。

---

## 快速开始

### 安装依赖

```bash
pip install mcp langchain-mcp-adapters langchain langchain-core langchain-openai python-dotenv pydantic
```

### 配置环境变量

创建 `.env` 文件：

```env
# 阿里云 Qwen 模型（大部分示例需要）
ALIYUN_API_KEY=your_dashscope_api_key

# 高德地图 API Key（高德相关示例需要）
AMAP_KEY=your_amap_web_service_key

# GitHub Token（GitHub MCP 示例需要）
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 运行示例

```bash
# 1. 概念学习（无需 API Key）
python what_is_mcp.py

# 2. MCP Tool 实战
python mcp_demo.py

# 3. 企业 API 封装
python enterprise_api_mcp_demo.py

# 4. Skill 编排
python skill_demo.py

# 5. 高德地图综合实战
python gaode_skill_test.py

# 6. 多 Agent 调度
python multiple_agent.py

# 7. A2A 协议实战（无需 API Key）
python a2a_demo.py
```

---

## 知识体系全景

```
                    ┌─────────────────────────────────────┐
                    │          应用层：Skill                │
                    │   skill_demo.py, gaode_skill_test.py │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────┴───────────────────┐
                    │          连接层：MCP                  │
                    │   mcp_demo.py                        │
                    │   mcp_resources_prompts_demo.py      │
                    │   enterprise_api_mcp_demo.py         │
                    │   local_weather_server.py            │
                    │   mcp_client_config_guide.py         │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────┴───────────────────┐
                    │        底层协议：Function Calling     │
                    │   what_is_mcp.py（概念）              │
                    └─────────────────────────────────────┘

    横向能力：
    ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │  单 Agent     │  │  多 Agent    │  │  客户端集成       │  │  A2A 协议        │
    │  mcp_demo.py  │→ │  multiple_   │→ │  mcp_client_    │→ │  a2a_demo.py    │
    │              │  │  agent.py    │  │  config_guide.py │  │                 │
    └──────────────┘  └──────────────┘  └──────────────────┘  └──────────────────┘
```

---

## 常见 MCP 服务器速查

| 服务器 | 启动方式 | 用途 | 需要认证 |
|--------|---------|------|---------|
| 本地天气 | `python local_weather_server.py` | 天气/空气质量教学演示 | 否 |
| 文件系统 | `npx @modelcontextprotocol/server-filesystem /path` | 读写本地文件 | 否 |
| GitHub | `npx -y @modelcontextprotocol/server-github` | 仓库操作 | GITHUB_TOKEN |
| 高德地图 | `https://mcp.amap.com/sse?key=KEY` | 地图/天气/路径规划 | AMAP_KEY |
| 浏览器 | `npx @anthropic/mcp-server-browser` | 网页截图/交互 | 否 |
| 数据库 | 各厂商 MCP Server | 查询数据库 | 连接配置 |
| 搜索引擎 | Brave Search / Google | 互联网搜索 | API Key |

---

## 与 LangGraph 的关系

本目录与 `langgraph_examples/` 的关系：

- **MCP** 解决的是"AI 模型如何连接外部工具"的问题（标准化接口层）
- **LangGraph** 解决的是"AI 工具之间如何编排流程"的问题（图结构执行层）
- 两者互补：MCP 提供工具，LangGraph 编排工具的执行流程

在 `langgraph_examples/07_deep_agents/` 中创建的 DeepAgent 也可以通过 MCP 工具来增强能力。
