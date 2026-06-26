# =============================================================================
# A2A (Agent-to-Agent Protocol) 旅游规划实战演示
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解 A2A 是什么、与 MCP 的区别
#   ✅ 掌握 Agent Card 的结构和发现机制
#   ✅ 理解 A2A 任务生命周期（SUBMITTED → WORKING → COMPLETED）
#   ✅ 掌握 JSON-RPC 2.0 消息格式在 A2A 中的应用
#   ✅ 实现多 Agent 通过 A2A 协作完成旅行规划
#
# 场景：旅游规划 -- 三个独立的 HTTP 服务通过 A2A 协议协作
#   - 天气 Agent (weather_agent)    : 提供天气查询服务（模拟 HTTP 8001 端口）
#   - 酒店 Agent (hotel_agent)      : 提供酒店搜索服务（模拟 HTTP 8002 端口）
#   - 旅行协调员 (travel_coordinator): 发现并调用上述 Agent，合成旅行计划
#
# 运行前准备：
#   1. 核心演示无需额外依赖，内置模拟 HTTP 通信（示例 1-5 可直接运行）
#   2. 可选：配置 ALIYUN_API_KEY 用于 LLM 增强回复（示例 6）
#   3. 可选：pip install fastapi uvicorn httpx（用于运行真实 HTTP 服务，见文件末尾）
#
# 建议阅读顺序：
#   1. what_is_mcp.py          - 理解 MCP 概念
#   2. mcp_demo.py             - 理解 MCP 实战
#   3. multiple_agent.py       - 理解多 Agent Supervisor 模式
#   4. 本文件（a2a_demo.py）   - 理解 Agent 之间的通信协议 A2A
# =============================================================================

import sys
import os
import io
import json
import uuid
import time
from datetime import datetime
from typing import Optional
from enum import Enum

# 设置标准输出编码为 UTF-8，避免 Windows GBK 编码错误
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.model_utils import get_model


# =============================================================================
# 核心概念：A2A 是什么？和 MCP 有什么区别？
# =============================================================================
"""
什么是 A2A（Agent-to-Agent Protocol）？

📻 定义
   A2A = Agent-to-Agent Protocol（Agent 间通信协议）
   由 Google 提出的开放标准，用于 AI Agent 之间通过 HTTP/JSON-RPC 进行通信。

🔲 类比理解
   MCP = USB-C 接口（模型连工具，像插头连插座）
   A2A = 打电话（Agent 连 Agent，像同事之间打电话协作）

   你是项目经理（旅行协调员 Agent），需要完成一个旅行规划任务：
   - 你打电话给气象局同事（天气 Agent）：请问北京明天天气？
   - 你打电话给酒店预订部（酒店 Agent）：请帮我找北京三里屯附近的酒店
   - 你综合两边的信息，给客户出一份完整的旅行计划
   每个同事都是独立的 HTTP 服务，通过标准协议（A2A）沟通。

📊 架构对比

   MCP 架构（模型 ↔ 工具）：
   ┌──────────┐     MCP 协议      ┌──────────────┐
   │ AI 模型   │←──── USB-C ─────→│ MCP 服务器    │
   │ (Client) │     Tools         │ 文件系统      │
   └──────────┘                   └──────────────┘

   A2A 架构（Agent ↔ Agent）：
   ┌──────────────┐    A2A 协议     ┌──────────────┐
   │ 旅行协调员    │←── HTTP/JSON ──→│ 天气 Agent    │
   │ (Client Agent)│    -RPC        │ (Server Agent)│
   │              │←── HTTP/JSON ──→│ 酒店 Agent    │
   └──────────────┘                 └──────────────┘

📦 A2A 核心概念

   1. Agent Card（Agent 名片）
      每个 Agent 都有一个 JSON 格式的名片，描述：
      - name: Agent 名称
      - description: Agent 描述
      - skills: 提供的技能列表
      - url: 通信端点
      类似于企业的"服务目录"，告诉别人"我能做什么"
      真实协议中通过 /.well-known/agent.json 端点暴露

   2. Task（任务）
      A2A 中的核心工作单元，有完整的生命周期：
      SUBMITTED（已提交）→ WORKING（处理中）→ COMPLETED（完成）
                                             → FAILED（失败）
      类似于"工单系统"，可以追踪进度

   3. Message（消息）
      通信的基本单位，使用 JSON-RPC 2.0 格式：
      {"jsonrpc": "2.0", "method": "tasks/send", "params": {...}, "id": 1}

   4. Artifact（产出物）
      任务完成后返回的结果，包含文本、文件、结构化数据等

📋 MCP vs A2A 对比表

   | 维度       | MCP                    | A2A                    |
   |-----------|------------------------|------------------------|
   | 关系       | 客户端-服务器           | 点对点（Peer-to-Peer） |
   | 类比       | USB-C 接口             | 打电话                |
   | 通信       | 工具调用 → 直接返回结果  | 任务提交 → 生命周期管理 |
   | 时长       | 短调用（秒级）          | 可长运行（分钟/小时级）  |
   | 发现       | 启动时获取 tools/list   | Agent Card 发现机制    |
   | 流式       | 通过 SSE 流式传输       | 支持任务状态更新推送    |
   | 典型场景   | 模型调用外部工具        | Agent 之间协作完成复杂任务|

🔗 LangGraph 多 Agent vs A2A 多 Agent

   | 维度       | LangGraph 多 Agent          | A2A 多 Agent              |
   |-----------|----------------------------|--------------------------|
   | 进程       | 同一个进程内               | 不同进程/不同机器         |
   | 通信       | 内存共享/函数调用           | HTTP/JSON-RPC            |
   | 部署       | 单体应用                   | 微服务架构                |
   | 发现       | 代码中硬编码                | Agent Card 动态发现       |
   | 优势       | 低延迟、简单                | 跨团队、跨语言、可独立部署 |
"""


# =============================================================================
# A2A 协议核心数据结构
# =============================================================================

class TaskStatus(str, Enum):
    """A2A 任务状态枚举"""
    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class A2AAgentCard:
    """
    A2A Agent Card — Agent 的"名片"，描述 Agent 的身份和能力。

    在真实 A2A 协议中，Agent Card 通过 /.well-known/agent.json 端点暴露，
    其他 Agent 可以通过 HTTP GET 读取这张名片来了解对方能做什么。
    """

    def __init__(self, name: str, description: str, skills: list[dict],
                 url: str, version: str = "1.0.0"):
        self.name = name
        self.description = description
        self.skills = skills
        self.url = url
        self.version = version

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "url": self.url,
            "skills": self.skills,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class A2ATask:
    """
    A2A 任务 — 通信的核心工作单元。

    真实 A2A 协议中，任务通过 JSON-RPC 2.0 消息在 Agent 之间传递，
    每个任务都有完整的生命周期：SUBMITTED → WORKING → COMPLETED/FAILED
    """

    def __init__(self, task_id: str, context_id: str, status: TaskStatus):
        self.id = task_id
        self.context_id = context_id
        self.status = status
        self.history: list[dict] = []
        self.artifacts: list[dict] = []
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.updated_at = self.created_at

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "contextId": self.context_id,
            "status": {"state": self.status, "timestamp": self.updated_at},
            "history": self.history,
            "artifacts": self.artifacts,
            "metadata": {"createdAt": self.created_at},
        }


def create_jsonrpc_request(method: str, params: dict, request_id: int = 1) -> dict:
    """创建 JSON-RPC 2.0 请求消息"""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    }


def create_jsonrpc_response(result: dict, request_id: int = 1) -> dict:
    """创建 JSON-RPC 2.0 响应消息"""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result,
    }


# =============================================================================
# 模拟数据 — 天气和酒店的"内部数据库"
# =============================================================================

WEATHER_DB = {
    "北京": {"condition": "晴", "temp": "25°C", "air": "良好", "humidity": "40%"},
    "上海": {"condition": "多云", "temp": "28°C", "air": "轻微雾霾", "humidity": "65%"},
    "广州": {"condition": "小雨", "temp": "30°C", "air": "良好", "humidity": "80%"},
    "深圳": {"condition": "晴", "temp": "29°C", "air": "良好", "humidity": "70%"},
    "杭州": {"condition": "多云", "temp": "26°C", "air": "良好", "humidity": "55%"},
    "成都": {"condition": "阴", "temp": "22°C", "air": "湿度较大", "humidity": "75%"},
    "三亚": {"condition": "晴", "temp": "32°C", "air": "优", "humidity": "60%"},
    "西安": {"condition": "晴", "temp": "27°C", "air": "良好", "humidity": "35%"},
}

HOTEL_DB = {
    "北京": [
        {"name": "北京王府井希尔顿酒店", "area": "王府井", "price": 899, "rating": 4.8},
        {"name": "北京三里屯 CHAO 酒店", "area": "三里屯", "price": 680, "rating": 4.6},
        {"name": "北京颐和安缦酒店", "area": "颐和园", "price": 2800, "rating": 4.9},
    ],
    "上海": [
        {"name": "上海外滩华尔道夫酒店", "area": "外滩", "price": 1580, "rating": 4.8},
        {"name": "上海陆家嘴丽思卡尔顿", "area": "陆家嘴", "price": 2200, "rating": 4.9},
        {"name": "上海静安香格里拉", "area": "静安寺", "price": 1200, "rating": 4.7},
    ],
    "广州": [
        {"name": "广州四季酒店", "area": "珠江新城", "price": 1380, "rating": 4.8},
        {"name": "广州白天鹅宾馆", "area": "沙面", "price": 680, "rating": 4.5},
    ],
    "深圳": [
        {"name": "深圳瑞吉酒店", "area": "罗湖", "price": 1280, "rating": 4.7},
        {"name": "深圳华侨城洲际酒店", "area": "南山", "price": 980, "rating": 4.6},
    ],
    "杭州": [
        {"name": "杭州西湖国宾馆", "area": "西湖", "price": 1800, "rating": 4.9},
        {"name": "杭州西溪悦榕庄", "area": "西溪", "price": 2200, "rating": 4.8},
    ],
    "三亚": [
        {"name": "三亚亚特兰蒂斯酒店", "area": "海棠湾", "price": 2680, "rating": 4.8},
        {"name": "三亚太阳湾柏悦酒店", "area": "太阳湾", "price": 3200, "rating": 4.9},
    ],
    "成都": [
        {"name": "成都博舍酒店", "area": "太古里", "price": 1580, "rating": 4.7},
        {"name": "成都万达瑞华酒店", "area": "锦江", "price": 780, "rating": 4.5},
    ],
    "西安": [
        {"name": "西安 W 酒店", "area": "曲江", "price": 980, "rating": 4.6},
        {"name": "西安索菲特传奇酒店", "area": "钟楼", "price": 680, "rating": 4.5},
    ],
}


def _get_weather_tip(condition: str, temp: str) -> str:
    """根据天气给出出行建议"""
    num_temp = int("".join(filter(str.isdigit, temp)))
    tips = []
    if "雨" in condition:
        tips.append("请携带雨具")
    if num_temp > 30:
        tips.append("注意防晒")
    elif num_temp < 15:
        tips.append("注意保暖")
    if "霾" in condition:
        tips.append("建议佩戴口罩")
    return "；".join(tips) if tips else "适宜出行"


def simulate_weather_query(city: str) -> dict:
    """天气 Agent 的核心逻辑 — 模拟天气查询"""
    data = WEATHER_DB.get(city)
    if not data:
        return {"error": f"暂无 {city} 的天气数据"}
    return {
        "city": city,
        "condition": data["condition"],
        "temperature": data["temp"],
        "air_quality": data["air"],
        "humidity": data["humidity"],
        "tip": _get_weather_tip(data["condition"], data["temp"]),
    }


def simulate_hotel_query(city: str, budget: Optional[int] = None,
                         area: Optional[str] = None) -> list[dict]:
    """酒店 Agent 的核心逻辑 — 模拟酒店查询"""
    hotels = HOTEL_DB.get(city, [])
    if budget:
        hotels = [h for h in hotels if h["price"] <= budget]
    if area:
        hotels = [h for h in hotels if area.lower() in h["area"].lower()]
    return hotels


# =============================================================================
# A2A 客户端实现
# =============================================================================

class A2AClient:
    """
    A2A 协议客户端 — 用于与远程 Agent 通信。

    在真实 A2A 协议中，此类通过 HTTP 发送 JSON-RPC 2.0 请求：
      - 获取 Agent Card: GET  http://{host}:{port}/.well-known/agent.json
      - 提交任务:        POST http://{host}:{port}/a2a (JSON-RPC 消息)

    本演示中使用内存模拟 HTTP 通信，展示完全相同的消息格式。
    """

    def __init__(self, agent_url: str, agent_card: A2AAgentCard,
                 agent_type: str):
        self.agent_url = agent_url
        self.agent_card = agent_card
        self.agent_type = agent_type  # "weather" | "hotel"
        self._request_id = 0

    def discover_agent_card(self) -> dict:
        """
        发现 Agent Card — A2A 协议的第一步。

        真实实现：
          response = httpx.get(f"{self.agent_url}/.well-known/agent.json")
          return response.json()
        """
        card = self.agent_card.to_dict()
        return card

    def submit_task(self, message: str, context_id: str = None) -> A2ATask:
        """
        提交任务到远程 Agent。

        真实实现（JSON-RPC 2.0）：
          request = {
              "jsonrpc": "2.0",
              "id": 1,
              "method": "tasks/send",
              "params": {
                  "message": {
                      "role": "user",
                      "parts": [{"type": "text", "text": message}]
                  },
                  "contextId": context_id or str(uuid.uuid4())
              }
          }
          response = httpx.post(f"{self.agent_url}/a2a", json=request)
        """
        self._request_id += 1
        task = A2ATask(
            task_id=f"task-{uuid.uuid4().hex[:8]}",
            context_id=context_id or str(uuid.uuid4()),
            status=TaskStatus.SUBMITTED,
        )

        # 打印 JSON-RPC 请求
        jsonrpc_req = create_jsonrpc_request(
            method="tasks/send",
            params={
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": message}],
                },
                "contextId": task.context_id,
            },
            request_id=self._request_id,
        )
        print(f"  [HTTP] POST {self.agent_url}/a2a")
        print(f"  [JSON-RPC 请求]")
        for line in json.dumps(jsonrpc_req, ensure_ascii=False).replace(",", ",\n    ").replace("{", "{\n    ").replace("}", "\n  }").split("\n"):
            print(f"    {line}")

        # 模拟 Agent 处理
        time.sleep(0.3)
        task.status = TaskStatus.WORKING
        task.history.append({"role": "user", "content": message})
        task.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"  [Task 状态] SUBMITTED → WORKING")

        return task

    def get_task_result(self, task: A2ATask,
                        query_params: dict) -> A2ATask:
        """
        获取任务结果（模拟 Agent 处理完成并返回）。

        真实实现：
          response = httpx.post(f"{self.agent_url}/a2a", json={
              "jsonrpc": "2.0",
              "method": "tasks/get",
              "params": {"taskId": task.id}
          })
        """
        # 模拟 Agent 处理逻辑
        if self.agent_type == "weather":
            result = simulate_weather_query(query_params.get("city", ""))
        elif self.agent_type == "hotel":
            result = simulate_hotel_query(
                query_params.get("city", ""),
                query_params.get("budget"),
                query_params.get("area"),
            )
        else:
            result = {"error": "未知的 Agent 类型"}

        task.status = TaskStatus.COMPLETED
        task.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        task.artifacts.append({
            "name": f"{self.agent_type}_result",
            "parts": [{"type": "data", "data": result}],
        })
        task.history.append({
            "role": "agent",
            "content": json.dumps(result, ensure_ascii=False),
        })

        # 打印 JSON-RPC 响应
        jsonrpc_resp = create_jsonrpc_response(
            result={"taskId": task.id, "status": task.status.value},
            request_id=self._request_id,
        )
        print(f"  [HTTP] 200 OK")
        print(f"  [JSON-RPC 响应]")
        for line in json.dumps(jsonrpc_resp, ensure_ascii=False).replace(",", ",\n    ").replace("{", "{\n    ").replace("}", "\n  }").split("\n"):
            print(f"    {line}")

        print(f"  [Task 状态] WORKING → COMPLETED")

        return task


# =============================================================================
# 示例 1: 概念讲解 — 什么是 A2A
# =============================================================================

def example1_what_is_a2a():
    """
    示例 1: 理解 A2A 协议 — 纯概念讲解，无需 API Key。

    通过生活化类比和架构图，建立对 A2A 的直观理解。
    """
    print("=" * 70)
    print("  示例 1: 什么是 A2A (Agent-to-Agent Protocol)")
    print("=" * 70)
    print()

    print("""
💡 一句话理解 A2A：
   如果 MCP 是"模型连接工具的 USB-C 接口"，
   那 A2A 就是"Agent 之间打电话的通信线路"。

📞 生活化场景 — 旅行规划：

   你（旅行协调员 Agent）接到一个任务："帮我规划北京 3 天旅行"

   你一个人搞不定所有信息，需要找同事帮忙：

   ┌─────────────────────────────────────────────────────┐
   │  你：打电话给气象局 → "北京明天天气怎么样？"         │
   │       ← "晴，25°C，空气质量良好"                    │
   │                                                     │
   │  你：打电话给酒店预订部 → "北京三里屯附近有什么酒店？"│
   │       ← "找到 3 家，价格 680-2800 元"               │
   │                                                     │
   │  你：综合信息 → 给客户出一份完整的旅行计划            │
   └─────────────────────────────────────────────────────┘

   在这个场景中：
   - 每个"同事"都是独立的 Agent（HTTP 服务）
   - "打电话"就是 A2A 协议（HTTP/JSON-RPC 通信）
   - 每个同事的"名片"就是 Agent Card（描述能做什么）
   - 每个"工单"就是 Task（有提交→处理→完成的生命周期）

🔗 与之前学过的内容对比：

   | 概念         | MCP                | LangGraph 多 Agent  | A2A               |
   |-------------|--------------------|---------------------|--------------------|
   | 连接对象     | 模型 → 工具        | 进程内 Agent ↔ Agent| 跨进程 Agent ↔ Agent|
   | 通信方式     | stdio / SSE        | 内存共享            | HTTP/JSON-RPC     |
   | 类比        | USB-C 接口         | 同一个办公室喊话     | 打电话            |
   | 典型场景     | 调外部 API、读文件  | 多专家协作分析       | 跨服务/跨团队协作  |
""")


# =============================================================================
# 示例 2: Agent Card 发现机制
# =============================================================================

def example2_agent_card_discovery():
    """
    示例 2: Agent Card 发现 — 展示三个 Agent 的名片。

    在 A2A 协议中，Agent 必须先通过 Agent Card 了解对方的能力，
    才能正确地调用它。这就像商务交往中先看名片再谈合作。

    无需 API Key，纯本地运行。
    """
    print("\n" + "=" * 70)
    print("  示例 2: Agent Card 发现机制")
    print("=" * 70)
    print()

    # 定义三个 Agent 的名片
    weather_card = A2AAgentCard(
        name="天气查询 Agent",
        description="提供城市天气查询服务，包括温度、天气状况、空气质量、出行建议",
        skills=[{
            "name": "check_weather",
            "description": "查询指定城市的实时天气信息，返回温度、天气状况、空气质量和出行建议",
            "tags": ["weather", "temperature", "air-quality", "travel-tip"],
        }],
        url="http://localhost:8001",
    )

    hotel_card = A2AAgentCard(
        name="酒店预订 Agent",
        description="提供酒店搜索和推荐服务，支持按城市、预算、区域筛选",
        skills=[{
            "name": "find_hotels",
            "description": "根据城市名称、预算上限、目标区域等条件搜索酒店，返回酒店名称、位置、价格和评分",
            "tags": ["hotel", "accommodation", "travel", "search"],
        }],
        url="http://localhost:8002",
    )

    travel_card = A2AAgentCard(
        name="旅行协调员 Agent",
        description="旅行规划协调员，综合天气和酒店信息，为您制定完整的旅行计划",
        skills=[{
            "name": "plan_trip",
            "description": "接收旅行需求，自动调用天气和酒店 Agent，综合生成旅行计划",
            "tags": ["travel", "planning", "coordination", "multi-agent"],
        }],
        url="http://localhost:8003",
    )

    print("【步骤 1】A2A 客户端发现可用的 Agent 服务\n")

    for i, card in enumerate([weather_card, hotel_card, travel_card], 1):
        print(f"  ┌─ Agent {i}: {card.name}")
        print(f"  │ URL: {card.url}")
        print(f"  │ 描述: {card.description}")
        print(f"  │ 技能列表:")
        for skill in card.skills:
            print(f"  │   • {skill['name']} — {skill['description']}")
        print(f"  │ 标签: {', '.join(tag for skill in card.skills for tag in skill['tags'])}")
        print(f"  │ Agent Card JSON:")
        card_json = card.to_json(indent=4)
        for line in card_json.split("\n"):
            print(f"  │   {line}")
        print(f"  └─{'─' * 50}")
        print()

    print("💡 Agent Card 的价值：")
    print("   就像看名片了解对方能力一样，Agent Card 让 Agent 在调用前就知道：")
    print("   - 对方叫什么名字、能做什么")
    print("   - 需要传什么参数")
    print("   - 在哪个 URL 通信")
    print("   这样就不需要提前硬编码每个 Agent 的接口！")
    print()


# =============================================================================
# 示例 3: 客户端 → 天气 Agent 直接 A2A 调用
# =============================================================================

def example3_direct_weather_call():
    """
    示例 3: 通过 A2A 协议调用天气 Agent。

    展示完整的 A2A 调用流程：
      1. 发现 Agent Card
      2. 提交任务（JSON-RPC 请求）
      3. 获取任务结果（JSON-RPC 响应）
      4. 查看任务状态和产出物

    无需 API Key，纯本地运行。
    """
    print("\n" + "=" * 70)
    print("  示例 3: 客户端 → 天气 Agent 直接 A2A 调用")
    print("=" * 70)
    print()

    # 定义天气 Agent 名片
    weather_card = A2AAgentCard(
        name="天气查询 Agent",
        description="提供城市天气查询服务",
        skills=[{"name": "check_weather", "description": "查询城市天气", "tags": ["weather"]}],
        url="http://localhost:8001",
    )

    # 创建 A2A 客户端
    client = A2AClient("http://localhost:8001", weather_card, "weather")

    # Step 1: 发现 Agent Card
    print("[步骤 1] 发现天气 Agent...")
    card = client.discover_agent_card()
    print(f"  已获取 Agent Card: {card['name']} — {card['description']}")
    print()

    # Step 2: 提交任务
    print("[步骤 2] 提交天气查询任务...")
    task = client.submit_task("查询北京的天气")
    print(f"  Task ID: {task.id}")
    print(f"  Context ID: {task.context_id}")
    print()

    # Step 3: 获取任务结果
    print("[步骤 3] 获取任务结果...")
    task = client.get_task_result(task, {"city": "北京"})
    print()

    # Step 4: 查看结果
    print("[步骤 4] 任务结果:")
    weather_data = task.artifacts[0]["parts"][0]["data"]
    print(f"  城市: {weather_data['city']}")
    print(f"  天气: {weather_data['condition']}")
    print(f"  温度: {weather_data['temperature']}")
    print(f"  空气质量: {weather_data['air_quality']}")
    print(f"  出行建议: {weather_data['tip']}")
    print()

    print("💡 注意看上面的 JSON-RPC 消息格式：")
    print("   - 请求包含 method='tasks/send'，消息内容是用户问题")
    print("   - 响应包含 task 的 ID、状态和结果")
    print("   - 这就是 A2A 协议的真实通信格式！")
    print()


# =============================================================================
# 示例 4: 客户端 → 酒店 Agent 直接 A2A 调用
# =============================================================================

def example4_direct_hotel_call():
    """
    示例 4: 通过 A2A 协议调用酒店 Agent。

    展示带参数过滤的酒店查询（预算、区域筛选）。

    无需 API Key，纯本地运行。
    """
    print("\n" + "=" * 70)
    print("  示例 4: 客户端 → 酒店 Agent 直接 A2A 调用")
    print("=" * 70)
    print()

    hotel_card = A2AAgentCard(
        name="酒店预订 Agent",
        description="提供酒店搜索和推荐服务",
        skills=[{"name": "find_hotels", "description": "搜索酒店", "tags": ["hotel"]}],
        url="http://localhost:8002",
    )

    # 场景 A: 按城市搜索（无过滤）
    print("─" * 50)
    print("  场景 A: 搜索北京所有酒店")
    print("─" * 50)
    print()

    client = A2AClient("http://localhost:8002", hotel_card, "hotel")

    task = client.submit_task("查找北京的酒店")
    task = client.get_task_result(task, {"city": "北京"})
    print()

    hotels = task.artifacts[0]["parts"][0]["data"]
    print(f"  找到 {len(hotels)} 家酒店:")
    for i, h in enumerate(hotels, 1):
        print(f"    {i}. {h['name']} | {h['area']} | ¥{h['price']}/晚 | 评分 {h['rating']}")
    print()

    # 场景 B: 带预算过滤
    print("─" * 50)
    print("  场景 B: 搜索北京 1000 元以下的酒店")
    print("─" * 50)
    print()

    client2 = A2AClient("http://localhost:8002", hotel_card, "hotel")

    task2 = client2.submit_task("查找北京1000元以下的酒店")
    task2 = client2.get_task_result(task2, {"city": "北京", "budget": 1000})
    print()

    hotels2 = task2.artifacts[0]["parts"][0]["data"]
    print(f"  找到 {len(hotels2)} 家符合条件的酒店:")
    for i, h in enumerate(hotels2, 1):
        print(f"    {i}. {h['name']} | {h['area']} | ¥{h['price']}/晚 | 评分 {h['rating']}")
    print()


# =============================================================================
# 示例 5: 旅行协调员 — 多 Agent A2A 协作（核心示例）
# =============================================================================

def example5_travel_coordinator():
    """
    示例 5: 旅行协调员通过 A2A 协议调用天气 + 酒店 Agent，合成旅行计划。

    这是 A2A 最核心的应用场景 — 一个 Agent 发现并调用其他 Agent，
    通过协作完成单个 Agent 无法独立完成的复杂任务。

    架构：
      用户 → 旅行协调员 → (A2A) → 天气 Agent
                        → (A2A) → 酒店 Agent
                        → 综合结果 → 用户

    无需 API Key，纯本地运行。
    """
    print("\n" + "=" * 70)
    print("  示例 5: 旅行协调员 — 多 Agent A2A 协作规划旅行（核心）")
    print("=" * 70)
    print()

    # 定义 Agent 名片
    weather_card = A2AAgentCard(
        name="天气查询 Agent",
        description="提供城市天气查询服务",
        skills=[{"name": "check_weather", "description": "查询城市天气", "tags": ["weather"]}],
        url="http://localhost:8001",
    )

    hotel_card = A2AAgentCard(
        name="酒店预订 Agent",
        description="提供酒店搜索和推荐服务",
        skills=[{"name": "find_hotels", "description": "搜索酒店", "tags": ["hotel"]}],
        url="http://localhost:8002",
    )

    def travel_coordinator_plan(city: str, days: int = 2,
                                budget: Optional[int] = None):
        """旅行协调员的核心逻辑 — 通过 A2A 调用多个 Agent。"""

        print(f"  【旅行协调员】收到任务: 规划 {city} {days} 天旅行")
        if budget:
            print(f"  【旅行协调员】预算限制: ¥{budget}/晚")
        print()

        # 步骤 1: 通过 A2A 调用天气 Agent
        print("  ┌─ 步骤 1: 通过 A2A 调用天气 Agent ────────────────────")
        weather_client = A2AClient("http://localhost:8001", weather_card, "weather")
        weather_task = weather_client.submit_task(f"查询{city}的天气")
        weather_task = weather_client.get_task_result(weather_task, {"city": city})
        weather_data = weather_task.artifacts[0]["parts"][0]["data"]
        print(f"  └─ 天气 Agent 返回: {weather_data['condition']}, {weather_data['temperature']}")
        print()

        # 步骤 2: 通过 A2A 调用酒店 Agent
        print("  ┌─ 步骤 2: 通过 A2A 调用酒店 Agent ────────────────────")
        hotel_client = A2AClient("http://localhost:8002", hotel_card, "hotel")
        hotel_params: dict = {"city": city}
        if budget:
            hotel_params["budget"] = budget
        hotel_task = hotel_client.submit_task(f"查找{city}的酒店")
        hotel_task = hotel_client.get_task_result(hotel_task, hotel_params)
        hotels = hotel_task.artifacts[0]["parts"][0]["data"]
        print(f"  └─ 酒店 Agent 返回: 找到 {len(hotels)} 家酒店")
        print()

        # 步骤 3: 综合信息生成旅行计划
        print("  ┌─ 步骤 3: 综合信息生成旅行计划 ────────────────────")
        _synthesize_plan(city, days, weather_data, hotels, budget)
        print()

    # 测试场景
    scenarios = [
        {"city": "北京", "days": 3, "budget": 1000},
        {"city": "三亚", "days": 5, "budget": None},
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"{'═' * 60}")
        print(f"  【旅行请求 {i}】{scenario['city']} {scenario['days']} 天旅行")
        print(f"{'═' * 60}")
        print()
        travel_coordinator_plan(**scenario)

    print("💡 这个示例展示了 A2A 的核心价值：")
    print("   旅行协调员不需要内置天气和酒店功能，")
    print("   而是通过 A2A 协议发现和调用专门的 Agent 服务。")
    print("   就像公司里不同部门协作一样，每个 Agent 只专注自己的领域！")
    print()


def _synthesize_plan(city: str, days: int, weather_data: dict,
                     hotels: list[dict], budget: Optional[int] = None):
    """综合天气和酒店信息生成旅行计划。"""

    if "error" in weather_data:
        print(f"  ⚠️  天气数据获取失败: {weather_data['error']}")
        return

    plan_lines = [
        f"  📋 【{city}】{days}天旅行计划",
        f"  {'─' * 40}",
        f"",
        f"  🌤️  天气信息：",
        f"    天气: {weather_data['condition']} | 温度: {weather_data['temperature']}",
        f"    空气质量: {weather_data['air_quality']} | 湿度: {weather_data['humidity']}",
        f"    出行提示: {weather_data['tip']}",
        f"",
    ]

    if not hotels:
        plan_lines.append("  🏨 酒店信息：未找到符合条件的酒店")
    else:
        plan_lines.append(f"  🏨 酒店推荐（找到 {len(hotels)} 家）：")
        # 按评分排序
        hotels_sorted = sorted(hotels, key=lambda h: h["rating"], reverse=True)
        for i, h in enumerate(hotels_sorted, 1):
            price_tag = "💰 性价比高" if h["price"] <= 800 else ""
            plan_lines.append(
                f"    {i}. {h['name']} | {h['area']} | "
                f"¥{h['price']}/晚 | 评分 {h['rating']} {price_tag}"
            )

    plan_lines.extend([
        f"",
        f"  📝 行程建议：",
        f"    Day 1: 到达{city}，入住酒店，周边熟悉",
        f"    Day 2: 游览{city}核心景点（根据天气调整室内外活动）",
    ])
    if days >= 3:
        plan_lines.append(f"    Day 3: 深度游或返程")
    if days >= 5:
        plan_lines.extend([
            f"    Day 4: 特色体验（当地美食/文化活动）",
            f"    Day 5: 购物和返程",
        ])

    print("\n".join(plan_lines))


# =============================================================================
# 示例 6: LLM 增强回复（可选，需要 API Key）
# =============================================================================

def example6_llm_enhanced_response():
    """
    示例 6: 使用 LLM 生成更自然的旅行计划。

    先用 A2A 协议从天气和酒店 Agent 获取结构化数据，
    再用 LLM 将这些数据转换为自然语言旅行计划。

    需要配置 API Key（阿里云 Qwen 或 Ollama 本地模型）。
    """
    print("\n" + "=" * 70)
    print("  示例 6: LLM 增强 — 生成自然语言旅行计划")
    print("=" * 70)
    print()

    model = get_model()
    if model is None:
        print("  【跳过】未配置 API Key，此示例需要模型支持")
        print("  提示：配置 ALIYUN_API_KEY 或使用 Ollama 本地模型")
        return

    from langchain_core.messages import HumanMessage

    # 先用 A2A 协议获取数据
    weather_data = simulate_weather_query("北京")
    hotels = simulate_hotel_query("北京", budget=1000)

    print("  【用户请求】请帮我规划北京 3 天的旅行\n")
    print("  【Agent 数据收集】")
    print(f"    天气: {weather_data['condition']}, {weather_data['temperature']}")
    print(f"    空气质量: {weather_data['air_quality']}")
    print(f"    酒店: 找到 {len(hotels)} 家 1000 元以下的酒店")
    for h in hotels:
        print(f"      - {h['name']} | ¥{h['price']}/晚 | 评分 {h['rating']}")
    print()
    print("  【LLM 生成旅行计划】")
    print("  " + "─" * 50)

    prompt = (
        f"你是一个旅行规划助手。请根据以下信息，为用户生成一份详细的北京 3 天旅行计划：\n\n"
        f"天气信息：{json.dumps(weather_data, ensure_ascii=False)}\n"
        f"酒店选项：{json.dumps(hotels, ensure_ascii=False)}\n\n"
        f"要求：\n"
        f"1. 根据天气给出穿衣建议和出行提示\n"
        f"2. 推荐性价比最高的酒店并说明理由\n"
        f"3. 给出 3 天的行程建议（包括景点和活动）\n"
        f"4. 语气友好，像专业的旅行顾问"
    )

    try:
        response = model.invoke([HumanMessage(content=prompt)])
        for line in response.content.split("\n"):
            print(f"  {line}")
    except Exception as e:
        print(f"  【错误】LLM 调用失败: {e}")

    print("  " + "─" * 50)
    print()


# =============================================================================
# 示例 7: 交互式旅行规划
# =============================================================================

def interactive_mode():
    """
    交互式旅行规划 — 用户可以输入城市和预算，
    旅行协调员会通过 A2A 协议调用天气和酒店 Agent。

    无需 API Key，纯本地运行。
    """
    print("\n" + "=" * 70)
    print("  示例 7: 交互式旅行规划")
    print("=" * 70)
    print()

    # 定义 Agent 名片
    weather_card = A2AAgentCard(
        name="天气查询 Agent",
        description="提供城市天气查询服务",
        skills=[{"name": "check_weather", "description": "查询城市天气", "tags": ["weather"]}],
        url="http://localhost:8001",
    )

    hotel_card = A2AAgentCard(
        name="酒店预订 Agent",
        description="提供酒店搜索和推荐服务",
        skills=[{"name": "find_hotels", "description": "搜索酒店", "tags": ["hotel"]}],
        url="http://localhost:8002",
    )

    print("  输入格式：城市 [天数] [预算]")
    print("  示例：北京 3 1000")
    print("  输入 '退出' 或 'quit' 结束")
    print()

    while True:
        try:
            user_input = input("  【请输入】> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  再见！")
            break

        if not user_input or user_input.lower() in ("退出", "quit", "exit", "q"):
            print("  再见！")
            break

        parts = user_input.split()
        if not parts:
            continue

        city = parts[0]
        days = int(parts[1]) if len(parts) > 1 else 2
        budget = int(parts[2]) if len(parts) > 2 else None

        print()
        travel_coordinator_interactive(city, days, budget, weather_card, hotel_card)
        print()


def travel_coordinator_interactive(city: str, days: int,
                                   budget: Optional[int],
                                   weather_card: A2AAgentCard,
                                   hotel_card: A2AAgentCard):
    """交互式旅行协调员逻辑。"""
    print(f"  【旅行协调员】收到任务: 规划 {city} {days} 天旅行")
    if budget:
        print(f"  【旅行协调员】预算限制: ¥{budget}/晚")
    print()

    print("  ┌─ 调用天气 Agent ────────────────────")
    wc = A2AClient("http://localhost:8001", weather_card, "weather")
    wt = wc.submit_task(f"查询{city}的天气")
    wt = wc.get_task_result(wt, {"city": city})
    wd = wt.artifacts[0]["parts"][0]["data"]
    print(f"  └─ 返回: {wd.get('condition', wd.get('error', '未知'))}, {wd.get('temperature', '')}")
    print()

    print("  ┌─ 调用酒店 Agent ────────────────────")
    hc = A2AClient("http://localhost:8002", hotel_card, "hotel")
    hp: dict = {"city": city}
    if budget:
        hp["budget"] = budget
    ht = hc.submit_task(f"查找{city}的酒店")
    ht = hc.get_task_result(ht, hp)
    hotels = ht.artifacts[0]["parts"][0]["data"]
    print(f"  └─ 返回: 找到 {len(hotels)} 家酒店")
    print()

    print("  ┌─ 旅行计划 ────────────────────")
    _synthesize_plan(city, days, wd, hotels, budget)
    print()


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  A2A (Agent-to-Agent Protocol) 旅游规划实战演示")
    print("=" * 70)
    print()
    print("  学习阶段总览：")
    print("    示例 1: 概念讲解 — 什么是 A2A（无需 API Key）")
    print("    示例 2: Agent Card 发现机制（无需 API Key）")
    print("    示例 3: 客户端 → 天气 Agent 直接调用（无需 API Key）")
    print("    示例 4: 客户端 → 酒店 Agent 直接调用（无需 API Key）")
    print("    示例 5: 旅行协调员 — 多 Agent A2A 协作 ★ 核心（无需 API Key）")
    print("    示例 6: LLM 增强回复（需要 API Key）")
    print("    示例 7: 交互式旅行规划（无需 API Key）")
    print()
    print("  建议按顺序学习：1 → 2 → 3 → 4 → 5")
    print()
    print("=" * 70 + "\n")

    # 示例 1: 概念讲解（无需 API Key）
    example1_what_is_a2a()

    # 示例 2: Agent Card 发现（无需 API Key）
    example2_agent_card_discovery()

    # 示例 3: 直接调用天气 Agent（无需 API Key）
    example3_direct_weather_call()

    # 示例 4: 直接调用酒店 Agent（无需 API Key）
    example4_direct_hotel_call()

    # 示例 5: 旅行协调员多 Agent 协作（无需 API Key，核心示例）
    example5_travel_coordinator()

    # 示例 6: LLM 增强（需要 API Key，取消注释后运行）
    # example6_llm_enhanced_response()

    # 示例 7: 交互式模式（需要手动输入）
    # interactive_mode()
