# =============================================================================
# 企业 API → MCP Server 实战示例
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 把企业已有的 REST API 封装为 MCP Server，让 AI 模型安全调用
#   ✅ 理解 MCP Server 在企业架构中的位置（API → MCP → Agent）
#   ✅ 掌握 FastMCP + Pydantic 定义带校验的 Tool
#   ✅ 在 LangChain Agent 中接入企业 MCP 服务
#
# 场景：某电商公司有一个"订单管理系统"的内部 API。
#       现在要把它封装成 MCP Server，让 AI 助手能帮客服查询订单、
#       检查库存、取消订单等。
#
# 运行前检查：
#   1. 已安装依赖：pip install mcp langchain-mcp-adapters pydantic
#   2. 本示例不需要任何外部 API Key，使用模拟数据
#   3. 先阅读 what_is_mcp.py 和 mcp_demo.py 建立基础概念
# =============================================================================


# =============================================================================
# 核心架构图示
# =============================================================================
"""
┌──────────────────────────────────────────────────────────────────┐
│                        企业 AI 架构                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────┐     ┌──────────────┐     ┌──────────────────┐    │
│   │ AI Agent │────→│  MCP Server  │────→│ 企业内部 REST API │    │
│   │ (Qwen)   │←────│  (FastMCP)   │←────│ (订单管理系统)     │    │
│   └──────────┘     └──────────────┘     └──────────────────┘    │
│        ↑                   ↑                    ↑               │
│   用户问"查订单"    MCP 协议标准化      已有的内部系统              │
│                                                                  │
│   关键价值：                                                      │
│   1. API 不变：企业内部 API 不需要改造                             │
│   2. 一次封装：MCP Server 写一次，所有兼容模型都能用               │
│   3. 安全可控：Tool 按需定义，不会暴露整个 API                      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

企业痛点对应：
  - 痛点 1："每个 AI 平台都要单独对接" → MCP 标准协议，一次对接
  - 痛点 2："API 权限颗粒度太粗" → MCP Tool 按需暴露，最小权限
  - 痛点 3："AI 调用出错难排查" → MCP 有标准错误格式，调试方便
  - 痛点 4："供应商锁定" → MCP 是开放协议，可切换到任何兼容模型
"""

from utils.model_utils import get_model

# =============================================================================
# 第一步：模拟企业内部 REST API（这是已有的系统，不修改）
# =============================================================================

class OrderAPI:
    """
    模拟企业内部订单管理系统 API。

    在真实场景中，这里可能是：
    - requests.get('https://internal-api.company.com/orders/...')
    - grpc 调用
    - 数据库直连查询
    - SAP / Oracle ERP 接口
    """

    def __init__(self):
        # 模拟数据库
        self._orders = {
            "ORD-001": {
                "id": "ORD-001",
                "customer": "张三",
                "product": "机械键盘 K100",
                "quantity": 2,
                "unit_price": 399,
                "total": 798,
                "status": "已发货",
                "created_at": "2026-06-20 10:30:00",
            },
            "ORD-002": {
                "id": "ORD-002",
                "customer": "李四",
                "product": "无线鼠标 M200",
                "quantity": 1,
                "unit_price": 199,
                "total": 199,
                "status": "待发货",
                "created_at": "2026-06-22 14:20:00",
            },
            "ORD-003": {
                "id": "ORD-003",
                "customer": "张三",
                "product": "显示器支架 S300",
                "quantity": 3,
                "unit_price": 259,
                "total": 777,
                "status": "已签收",
                "created_at": "2026-06-18 09:15:00",
            },
            "ORD-004": {
                "id": "ORD-004",
                "customer": "王五",
                "product": "USB-C 拓展坞",
                "quantity": 1,
                "unit_price": 349,
                "total": 349,
                "status": "待支付",
                "created_at": "2026-06-24 16:45:00",
            },
        }
        self._inventory = {
            "机械键盘 K100": 15,
            "无线鼠标 M200": 42,
            "显示器支架 S300": 8,
            "USB-C 拓展坞": 23,
            "Type-C 数据线": 100,
            "笔记本散热器": 5,
        }
        self._customers = {
            "张三": {"name": "张三", "level": "VIP", "total_orders": 28, "credit": 500},
            "李四": {"name": "李四", "level": "普通", "total_orders": 5, "credit": 0},
            "王五": {"name": "王五", "level": "VIP", "total_orders": 15, "credit": 200},
        }

    # ---- 订单相关 ----
    def get_order(self, order_id: str) -> dict | None:
        """查询单个订单"""
        return self._orders.get(order_id)

    def search_orders_by_customer(self, customer_name: str) -> list[dict]:
        """按客户名查询订单"""
        return [o for o in self._orders.values()
                if customer_name in o["customer"]]

    def cancel_order(self, order_id: str) -> dict:
        """取消订单（只有待发货/待支付状态可取消）"""
        order = self._orders.get(order_id)
        if not order:
            return {"success": False, "message": f"订单 {order_id} 不存在"}
        if order["status"] not in ("待发货", "待支付"):
            return {
                "success": False,
                "message": f"订单 {order_id} 当前状态为 '{order['status']}'，无法取消",
            }
        order["status"] = "已取消"
        return {"success": True, "message": f"订单 {order_id} 已取消"}

    # ---- 库存相关 ----
    def check_inventory(self, product_name: str) -> dict:
        """查询商品库存"""
        qty = self._inventory.get(product_name)
        if qty is None:
            return {"found": False, "message": f"未找到商品 '{product_name}'"}
        status = "充足" if qty > 20 else "紧张" if qty > 5 else "告急"
        return {"found": True, "product": product_name, "quantity": qty, "status": status}

    def list_low_stock(self, threshold: int = 10) -> list[dict]:
        """列出库存低于阈值的商品"""
        return [
            {"product": name, "quantity": qty}
            for name, qty in self._inventory.items() if qty <= threshold
        ]

    # ---- 客户相关 ----
    def get_customer(self, name: str) -> dict | None:
        """查询客户信息"""
        return self._customers.get(name)


# =============================================================================
# 第二步：用 FastMCP 封装为 MCP Server
# =============================================================================

def create_mcp_server(order_api: OrderAPI):
    """
    将企业内部 API 封装为 MCP Server。

    关键原则：
    1. 一个 Tool 只暴露一个业务能力（最小权限）
    2. 用 Pydantic 做输入校验，防止非法参数进入内部系统
    3. Tool 返回自然语言描述，方便 LLM 理解
    4. 敏感操作（如取消订单）加状态检查，不直接透传
    """
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("订单管理系统")

    # ---- Tool 1: 查询订单 ----
    @mcp.tool()
    def query_order(order_id: str) -> str:
        """查询指定订单的详细信息，包括客户、商品、金额、状态等。
        参数 order_id 格式如 'ORD-001'。
        """
        order = order_api.get_order(order_id)
        if not order:
            return f"未找到订单 {order_id}。请确认订单号是否正确。"
        return (
            f"订单 {order['id']} 详情：\n"
            f"  客户：{order['customer']}\n"
            f"  商品：{order['product']} × {order['quantity']}\n"
            f"  单价：¥{order['unit_price']}  总价：¥{order['total']}\n"
            f"  状态：{order['status']}\n"
            f"  创建时间：{order['created_at']}"
        )

    # ---- Tool 2: 按客户查询订单 ----
    @mcp.tool()
    def query_orders_by_customer(customer_name: str) -> str:
        """按客户姓名查询其所有订单。参数 customer_name 为客户姓名，如 '张三'。"""
        orders = order_api.search_orders_by_customer(customer_name)
        if not orders:
            return f"未找到客户 '{customer_name}' 的任何订单。"
        lines = [f"客户 '{customer_name}' 共有 {len(orders)} 笔订单："]
        for o in orders:
            lines.append(
                f"  {o['id']} — {o['product']} × {o['quantity']} "
                f"— ¥{o['total']} — {o['status']}"
            )
        return "\n".join(lines)

    # ---- Tool 3: 取消订单 ----
    @mcp.tool()
    def cancel_order_tool(order_id: str) -> str:
        """取消指定订单（仅待发货/待支付状态可取消）。
        参数 order_id 格式如 'ORD-004'。
        """
        result = order_api.cancel_order(order_id)
        if result["success"]:
            return f"✓ {result['message']}"
        return f"✗ {result['message']}"

    # ---- Tool 4: 查库存 ----
    @mcp.tool()
    def query_inventory(product_name: str) -> str:
        """查询指定商品的库存数量。参数 product_name 为商品名称，如 '机械键盘 K100'。"""
        result = order_api.check_inventory(product_name)
        if not result["found"]:
            return result["message"]
        return (
            f"商品 '{result['product']}'：\n"
            f"  库存：{result['quantity']} 件\n"
            f"  状态：{result['status']}"
        )

    # ---- Tool 5: 列出低库存 ----
    @mcp.tool()
    def list_low_stock_products(threshold: int = 10) -> str:
        """列出库存低于指定阈值的所有商品。参数 threshold 为库存阈值，默认 10。"""
        items = order_api.list_low_stock(threshold)
        if not items:
            return f"所有商品库存均高于 {threshold} 件。"
        lines = [f"以下商品库存 ≤ {threshold} 件："]
        for item in items:
            lines.append(f"  {item['product']} — 仅剩 {item['quantity']} 件")
        return "\n".join(lines)

    # ---- Tool 6: 查客户信息 ----
    @mcp.tool()
    def query_customer(name: str) -> str:
        """查询客户信息，包括等级、累计订单数、信用额度等。参数 name 为客户姓名，如 '张三'。"""
        customer = order_api.get_customer(name)
        if not customer:
            return f"未找到客户 '{name}'。"
        return (
            f"客户 '{customer['name']}' 信息：\n"
            f"  等级：{customer['level']}\n"
            f"  累计订单：{customer['total_orders']} 笔\n"
            f"  信用额度：¥{customer['credit']}"
        )

    return mcp


# =============================================================================
# 第三步：在 LangChain Agent 中接入企业 MCP 服务
# =============================================================================

def demo_enterprise_mcp_agent():
    """
    模拟真实场景：客服人员通过 AI 助手处理日常工单。

    流程：
    1. 启动 MCP Server（连接企业内部订单系统）
    2. LangChain Agent 加载 MCP 工具
    3. 模拟客服对话
    """
    print("\n" + "=" * 70)
    print("  企业 MCP 实战：AI 客服助手接入订单系统")
    print("=" * 70 + "\n")

    # ---- 3.1 初始化企业内部 API ----
    print("【1】初始化内部订单系统 API...")
    order_api = OrderAPI()
    print("  ✓ 订单系统就绪（共 4 笔订单，6 种商品，3 位客户）")

    # ---- 3.2 创建 MCP Server ----
    print("\n【2】封装为 MCP Server...")
    mcp_server = create_mcp_server(order_api)
    print("  ✓ MCP Server 已创建")
    print("  提供的工具：")
    for tool_name in ["query_order", "query_orders_by_customer",
                      "cancel_order_tool", "query_inventory",
                      "list_low_stock_products", "query_customer"]:
        print(f"    - {tool_name}")

    # ---- 3.3 方式 A：直接在代码中调用 MCP Tool ----
    print("\n" + "-" * 60)
    print("【3】方式 A：直接在代码中调用 MCP Tool（不经过 Agent）")
    print("-" * 60)
    print("  适用于：明确知道要调用哪个工具的场景\n")

    # 查询订单
    result = asyncio.run(mcp_server.call_tool("query_order", {"order_id": "ORD-001"}))
    print(f"  调用 query_order('ORD-001')：\n    {result[0][0].text.replace(chr(10), chr(10)+'    ')}")

    # 查库存
    print()
    result = asyncio.run(mcp_server.call_tool("query_inventory", {"product_name": "笔记本散热器"}))
    print(f"  调用 query_inventory('笔记本散热器')：\n    {result[0][0].text}")

    # 低库存
    print()
    result = asyncio.run(mcp_server.call_tool("list_low_stock_products", {"threshold": 10}))
    print(f"  调用 list_low_stock_products(10)：\n    {result[0][0].text.replace(chr(10), chr(10)+'    ')}")

    # ---- 3.4 方式 B：接入 LangChain Agent ----
    print("\n" + "-" * 60)
    print("【4】方式 B：接入 LangChain Agent（AI 自动决策）")
    print("-" * 60)
    print("  适用于：用户自由提问，Agent 自动选择工具\n")

    from langchain.agents import create_agent
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool

    model = get_model("qwen")
    if model is None:
        print("  【跳过】请配置 ALIYUN_API_KEY\n")
        return

    # 将 MCP 工具包装为 LangChain 工具
    # 原理：每个 @tool 函数内部调用 mcp_server.call_tool()，LangChain Agent 可自动调用
    @tool
    def query_order(order_id: str) -> str:
        """查询单个订单详情。order_id 如 ORD-001"""
        r = asyncio.run(mcp_server.call_tool("query_order", {"order_id": order_id}))
        return r[0][0].text

    @tool
    def query_orders_by_customer(customer_name: str) -> str:
        """按客户姓名查询所有订单。customer_name 如 张三"""
        r = asyncio.run(mcp_server.call_tool("query_orders_by_customer", {"customer_name": customer_name}))
        return r[0][0].text

    @tool
    def cancel_order_tool(order_id: str, reason: str = "客户要求") -> str:
        """取消订单。order_id 如 ORD-004，reason 取消原因"""
        r = asyncio.run(mcp_server.call_tool("cancel_order_tool", {"order_id": order_id, "reason": reason}))
        return r[0][0].text

    @tool
    def query_inventory(product_name: str) -> str:
        """查询商品库存。product_name 如 机械键盘 K100"""
        r = asyncio.run(mcp_server.call_tool("query_inventory", {"product_name": product_name}))
        return r[0][0].text

    @tool
    def list_low_stock_products(threshold: int = 10) -> str:
        """列出库存低于阈值的商品。threshold 默认 10"""
        r = asyncio.run(mcp_server.call_tool("list_low_stock_products", {"threshold": threshold}))
        return r[0][0].text

    @tool
    def query_customer(customer_name: str) -> str:
        """查询客户信息。customer_name 如 张三"""
        r = asyncio.run(mcp_server.call_tool("query_customer", {"customer_name": customer_name}))
        return r[0][0].text

    langchain_tools = [query_order, query_orders_by_customer, cancel_order_tool,
                       query_inventory, list_low_stock_products, query_customer]
    print(f"  ✓ 已将 {len(langchain_tools)} 个 MCP 工具转换为 LangChain 工具")

    # 创建 Agent
    agent = create_agent(
        model,
        tools=langchain_tools,
        system_prompt=(
            "你是企业内部 AI 客服助手，可以访问订单管理系统。\n"
            "可用工具：\n"
            "- query_order: 查询单个订单详情\n"
            "- query_orders_by_customer: 按客户名查询所有订单\n"
            "- cancel_order_tool: 取消订单（需确认状态）\n"
            "- query_inventory: 查询商品库存\n"
            "- list_low_stock_products: 列出低库存商品\n"
            "- query_customer: 查询客户信息\n\n"
            "回复原则：\n"
            "1. 先查后说：涉及订单/库存/客户时，先调工具再回答\n"
            "2. 谨慎操作：取消订单前，先确认订单状态\n"
            "3. 友好简洁：用自然语言回复，不要直接输出 JSON"
        ),
    )

    # 模拟客服对话
    conversations = [
        # 场景 1：单一查询
        "帮我查一下订单 ORD-002 的详情",
        # 场景 2：跨工具组合
        "张三有哪些订单？他的客户等级是什么？",
        # 场景 3：需要判断的操作
        "订单 ORD-004 可以取消吗？如果可以就帮我取消",
        # 场景 4：数据分析
        "哪些商品库存比较紧张？给我列出来",
    ]

    for i, question in enumerate(conversations, 1):
        print(f"\n  ┌─ 客服对话 {i} ──────────────────────")
        print(f"  │ 用户：{question}")
        print(f"  ├─ AI 思考中...")
        try:
            result = agent.invoke({"messages": [HumanMessage(content=question)]})
            answer = result["messages"][-1].content
            # 截取前 300 字
            if len(answer) > 300:
                answer = answer[:300] + "..."
            print(f"  └─ AI：{answer}")
        except Exception as e:
            print(f"  └─ AI：【出错】{e}")

    print("\n" + "=" * 70)
    print("  企业 MCP 实战演示完成")
    print("=" * 70)
    print()
    print("  回顾关键步骤：")
    print("    1. 创建 OrderAPI 类（模拟企业内部系统）")
    print("    2. 用 FastMCP 定义 6 个 Tool")
    print("    3. 方式 A：直接调用   → mcp_server.call_tool(...)")
    print("    4. 方式 B：接入 Agent  → mcp_server.@tool 包装 + mcp_server.call_tool()")
    print()


# =============================================================================
# 第四步：企业实际部署建议
# =============================================================================

def deployment_guide():
    """企业 MCP Server 从开发到上线的建议"""
    print("\n" + "=" * 70)
    print("  企业 MCP Server 部署指南")
    print("=" * 70)
    print("""
┌─────────────────────────────────────────────────────────────────┐
│ 开发阶段                                                        │
├─────────────────────────────────────────────────────────────────┤
│ 1. 梳理 API → 选择对 AI 最有价值的接口（查询优先，写操作谨慎）     │
│ 2. 用 FastMCP 定义 Tool，每个 Tool 只做一件事                    │
│ 3. 先用 call_tool() 直接测试，验证每个 Tool 独立可用              │
│ 4. 再接入 Agent，观察 AI 是否能正确选择工具                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 上线阶段                                                        │
├─────────────────────────────────────────────────────────────────┤
│ 1. 认证鉴权：在 Tool 内部验证 API Key / Token                    │
│    def query_order(order_id: str, auth_token: str) -> str:      │
│        if not verify_token(auth_token):                          │
│            return "认证失败"                                     │
│                                                                  │
│ 2. 权限控制：[查询订单] 所有人可用，[取消订单] 仅管理员            │
│    在 Tool 函数内部加角色检查                                    │
│                                                                  │
│ 3. 日志审计：每次 Tool 调用记录：谁、什么操作、结果               │
│    import logging                                                │
│    logging.info(f"Tool call: query_order by {user}, args={...}")│
│                                                                  │
│ 4. 速率限制：防止 AI 频繁调用压垮内部系统                         │
│    用令牌桶或滑动窗口限制每秒请求数                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 部署方式                                                        │
├─────────────────────────────────────────────────────────────────┤
│ 方式 A — stdio 本地部署（开发/测试）：                            │
│   Client 和 Server 在同一台机器，通过子进程通信                   │
│   配置示例（Claude Desktop / Codex）：                            │
│   {                                                              │
│     "mcpServers": {                                              │
│       "order-system": {                                          │
│         "command": "python",                                     │
│         "args": ["mcp_server.py"]                                │
│       }                                                          │
│     }                                                            │
│   }                                                              │
│                                                                  │
│ 方式 B — SSE 远程部署（生产）：                                   │
│   Server 部署在云上，Client 通过 HTTPS 连接                       │
│   配置示例：                                                     │
│   {                                                              │
│     "mcpServers": {                                              │
│       "order-system": {                                          │
│         "url": "https://api.company.com/mcp/orders",             │
│         "transport": "sse",                                      │
│         "headers": {"Authorization": "Bearer <token>"}           │
│       }                                                          │
│     }                                                            │
│   }                                                              │
└─────────────────────────────────────────────────────────────────┘
""")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    import sys
    import io
    import asyncio
    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding='utf-8', errors='replace'
        )

    # 1. 先展示部署指南
    # deployment_guide()

    # 2. 实战演示
    demo_enterprise_mcp_agent()

    print("=" * 70)
    print("  继续学习：gaode_skill_test.py（生产级 Skill 实战）")
    print("=" * 70 + "\n")
