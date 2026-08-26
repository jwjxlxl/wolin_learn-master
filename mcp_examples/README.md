# MCP 教学模块

Model Context Protocol（模型上下文协议）—— 让 AI 模型连接任何工具的标准方式。

---

## 学习路线图

```
概念理解 ──→ Tool 实战 ──→ 企业实战 ──→ 手动封装 ──→ Skill ──→ 多 Agent ──→ A2A
   │              │              │            │          │         │         │
what_is_mcp   mcp_demo      student_       gaode_     skill_    multiple_  a2a_demo
.py           .py           management_    api_tool_  demo.py   agent.py   .py
                            mcp_demo.py    demo.py
```

**建议学习顺序**（标注 `*` 为核心文件）：

| 阶段 | 文件 | 内容 | 需要 API | 预计时间 |
|:----:|------|------|:--------:|:-------:|
| 1 | `what_is_mcp.py` | MCP 核心概念、三层架构、协议原理 | 否 | 15 min |
| 2* | `mcp_demo.py` | 本地 stdio / 远程 SSE / 多服务混合 / GitHub | 是 | 20 min |
| 3 | `local_weather_server.py` | 最简 MCP Server（50 行，2 个 Tool） | 否 | 10 min |
| 4* | `student_management_mcp_demo.py` | 企业 REST API → FastMCP → Agent 接入 | 是 | 20 min |
| 5 | `gaode_api_tool_demo.py` | 手动用 requests 封装高德 REST API 为 Agent Tool | 是 | 20 min |
| 6 | `skill_demo.py` | Skill 是什么、与 Tool 的区别、多 Tool 编排 | 是 | 15 min |
| 7 | `multiple_agent.py` | Supervisor 多 Agent 调度模式 | 是 | 15 min |
| 8* | `a2a_demo.py` | A2A 协议：Agent 之间通过 HTTP/JSON-RPC 通信 | 否 | 20 min |

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
| **Resource**（资源） | 模型的眼睛 | 让模型"读"数据 | 读取规章、FAQ、知识库 |
| **Prompt**（提示） | 模型的剧本 | 让模型"遵循"模板 | 代码审查、摘要生成 |

### 两种传输方式

| 方式 | 通信机制 | 适用场景 | 示例 |
|------|---------|---------|------|
| **Stdio** | 本地进程 stdin/stdout | 本地工具 | `python local_weather_server.py` |
| **SSE** | HTTP Server-Sent Events | 云端服务 | `https://mcp.amap.com/sse?key=KEY` |

---

## 面试必问：为什么要封装成 MCP？直接调 API 不行吗？

> 这是面试官常问的问题。核心答案：**能直接调，但要看"谁"在调。**

### 关键区分

| 调用方 | 直接调 API | 通过 MCP |
|--------|-----------|----------|
| **人类写的代码** | ✅ 最合适 | ❌ 没必要，多一层反而增加复杂度 |
| **AI Agent / LLM** | ❌ 问题很多 | ✅ MCP 就是为此设计的 |

### 直接调 API 给 LLM 用的 5 个坑

**1. 参数理解成本高**

REST API 的 JSON schema 对 LLM 来说缺乏语义约束。10+ 个字段里哪些必填？数据类型是什么？LLM 容易拼错字段名（`classId` vs `class_id`）。

MCP 的 Tool 定义自带类型签名 + 文档字符串，LLM 一看就懂：
```python
def add_student(name: str, class_id: int, gender: str = "") -> str
```

**2. 响应格式不统一**

不同 API 返回格式各异，LLM 每次都要自己解析。MCP 的 Tool 返回统一的自然语言文本，LLM 拿到就能用。

**3. 没有"能力描述"**

REST API 是一堆端点，LLM 不知道哪个该用。MCP 的每个 Tool 有清晰的 `description`，Agent 的规划器能据此自动选择合适工具。

**4. 安全管控难**

直接暴露 API = 给 LLM 全量访问能力。MCP 按最小权限原则，只暴露需要的 Tool，写操作可在 Tool 内加二次确认和数据校验。

**5. 模型适配成本高**

换模型 = 换 Agent 框架 = 重新适配 API。MCP 是标准协议，一次封装，所有兼容模型都能用。

### 一句话总结

> **MCP 是给 AI Agent 用的 API 适配层，不是给人类代码用的。**
> 
> 就像预制菜：原材料（REST API）直接给厨师（人类代码）没问题，但做成预制菜（MCP）才能让不会做饭的人（LLM）轻松吃上饭。

### MCP 的缺陷与局限

面试官追问：**MCP 有什么缺点？** 能主动说出缺陷，说明你不是盲目追热点。

| 缺陷 | 说明 | 应对方式 |
|------|------|---------|
| **额外延迟** | 每次 Tool 调用都要经过 MCP 协议转换（序列化 → 传输 → 反序列化），比直接 HTTP 调用多一层开销 | 对延迟敏感的调用，Agent 代码内直接 HTTP 请求 |
| **调试成本高** | 错误可能在 3 层传递中丢失或变形：Agent → MCP Client → MCP Server → 原始 API，排查链路长 | 开启 `MCP_DEBUG=1` 查看原始 JSON-RPC 报文 |
| **生态碎片化** | MCP 2024 年才提出，标准仍在演进，不同实现（FastMCP / MCP SDK）之间存在兼容性问题 | 锁定版本，不要追最新版 |
| **异步复杂度** | MCP 基于 async 通信，在同步代码中需要 `asyncio.run()` 包装，容易出事件循环冲突 | 统一项目内的 async/sync 风格 |
| **安全边界模糊** | stdio 方式下 MCP Server 以子进程运行，拥有当前用户的完整文件系统权限 | 生产环境用 SSE + 认证，限制 Server 运行权限 |
| **不适合复杂工作流** | MCP 是"单次调用 → 返回结果"模型，不适合需要状态保持、多轮交互的复杂流程 | 复杂流程用 LangGraph 或 A2A 协议 |
| **版本协商缺失** | MCP Server 升级后 Tool 签名可能变化，Client 端难以感知，导致 Agent 调用失败 | Server 端做向后兼容，或在 Tool description 中标注版本 |

### 什么时候不应该用 MCP

- **前后端通信**：前端调后端 API，直接 HTTP 就行
- **微服务间调用**：gRPC / REST 更高效，MCP 反而增加延迟
- **批处理 / 定时任务**：脚本直接调 API，不需要经过 LLM
- **高并发场景**：MCP 协议有额外开销，不适合 QPS 要求高的场景

---

## 文件说明

### 📖 概念层

**`what_is_mcp.py`** — MCP 核心概念（纯概念，无需 API）
- USB-C 类比理解 MCP 解决的问题
- 三种核心能力详解（Tool / Resource / Prompt）
- 与 Function Calling、Skill 的关系
- MCP 协议内部原理（JSON-RPC 2.0、生命周期、传输层差异）
- `langchain-mcp-adapters` 桥接原理

### 🔧 实战层

**`local_weather_server.py`** — 最简 MCP Server（50 行）
- `get_weather(city)` — 天气查询
- `get_air_quality(city)` — 空气质量查询

**`mcp_demo.py`** — MCP Tool 实战 ⭐
四个阶段递进学习：
1. **入门**：本地 stdio 连接天气 Server
2. **进阶**：远程 SSE 连接高德地图官方 Server
3. **实战**：本地 + 远程多服务混合编排
4. **扩展**：GitHub 官方 MCP Server 连接

**`student_management_mcp_demo.py`** — 企业 API → MCP Server 实战 ⭐

完整演示如何把企业内部 REST API 封装为 MCP Server：
- **第一步**：`StudentManagementAPI` 类封装 HTTP 客户端（登录鉴权 + 查询 + 新增）
- **第二步**：用 `FastMCP` 定义 2 个 Tool（最小权限 + 自然语言返回）
- **第三步**：两种方式接入 LangChain Agent（直接调用 vs AI 自动决策）
- **第四步**：企业部署建议（认证鉴权、权限控制、日志审计、速率限制、stdio/SSE 部署）

**`gaode_api_tool_demo.py`** — 手动封装高德 REST API 为 Agent Tool

⚠️ 不是连接高德官方 MCP Server，而是手动用 `requests` 调用高德 Web Service API，将每个接口封装为 LangChain `@tool`。覆盖 6 大场景 + 1 个组合 Skill（智能旅游规划）。

**教学价值**：理解如何手动将任意 REST API 封装为 Agent Tool，不依赖 mcp 包。

### 🎯 进阶层

**`skill_demo.py`** — Skill 技能演示
- Tool（螺丝刀）vs Skill（维修技能）对比
- 通过 `system_prompt` 编排多个 Tool 形成 Skill

**`multiple_agent.py`** — 多 Agent Supervisor 模式
- 主 Agent（Supervisor）负责规划分发
- 三个子 Agent 各负责细分领域（计算/翻译/写作）
- 使用 `langgraph-supervisor` 实现

**`a2a_demo.py`** — A2A 协议实战 ⭐（无需 API Key）

Google 提出的 Agent 间通信协议（HTTP/JSON-RPC），通过"旅游规划三剑客"场景演示：
- **天气 Agent**（模拟 HTTP 8001） + **酒店 Agent**（模拟 HTTP 8002）
- **旅行协调员**：通过 A2A 协议发现并调用上述两个 Agent
- Agent Card 发现机制 / Task 生命周期 / JSON-RPC 消息格式 / MCP vs A2A 对比

---

## 快速开始

### 安装依赖

```bash
pip install mcp langchain-mcp-adapters langchain langchain-core langchain-openai python-dotenv pydantic httpx
```

### 配置环境变量

```env
# 阿里云 Qwen 模型
ALIYUN_API_KEY=your_dashscope_api_key

# 高德地图 API Key
AMAP_KEY=your_amap_web_service_key

# GitHub Token（可选）
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 运行示例

```bash
# 1. 概念学习（无需 API Key）
python what_is_mcp.py

# 2. MCP Tool 实战
python mcp_demo.py

# 3. 企业 API 封装（需启动学生管理系统，见文件内路径）
python student_management_mcp_demo.py

# 4. 高德 API 手动封装
python gaode_api_tool_demo.py

# 5. A2A 协议实战（无需 API Key）
python a2a_demo.py
```

---

## 知识体系全景

```
                    ┌─────────────────────────────────────┐
                    │          应用层：Skill                │
                    │   skill_demo.py                      │
                    │   gaode_api_tool_demo.py             │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────┴───────────────────┐
                    │          连接层：MCP                  │
                    │   mcp_demo.py                        │
                    │   student_management_mcp_demo.py     │
                    │   local_weather_server.py            │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────┴───────────────────┐
                    │        底层协议：Function Calling     │
                    │   what_is_mcp.py（概念）              │
                    └─────────────────────────────────────┘

    横向能力：
    ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
    │  单 Agent     │  │  多 Agent    │  │  A2A 协议        │
    │  mcp_demo.py  │→ │  multiple_   │→ │  a2a_demo.py    │
    │              │  │  agent.py    │  │                 │
    └──────────────┘  └──────────────┘  └──────────────────┘
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

- **MCP** 解决的是"AI 模型如何连接外部工具"的问题（标准化接口层）
- **LangGraph** 解决的是"AI 工具之间如何编排流程"的问题（图结构执行层）
- 两者互补：MCP 提供工具，LangGraph 编排工具的执行流程
