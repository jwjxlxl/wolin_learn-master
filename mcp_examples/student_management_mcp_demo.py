# =============================================================================
# 学生管理 API → MCP Server 实战
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 把企业已有的 REST API 封装为 MCP Server，让 AI 模型调用
#   ✅ 理解 MCP Server 在企业架构中的位置（API → MCP → Agent）
#   ✅ 掌握 httpx 同步调用 + FastMCP 定义 Tool 的方式
#   ✅ 在 LangChain Agent 中接入企业 MCP 服务
#
# 场景：某学校有一个"学生管理系统"（FastAPI，端口 8000）。
#       现在要把它封装成 MCP Server，让 AI 助手能帮教务助理
#       查询学生信息、录入新学生。
#
# 运行前检查：
#   1. 已安装依赖：pip install mcp langchain-mcp-adapters pydantic httpx
#   2. 学生管理系统已启动：http://127.0.0.1:8000（见 C:\python_project\AI409TEst\student_management）
#   3. 不需要 API Key，使用本地学生管理系统
#   4. 先阅读 enterprise_api_mcp_demo.py 建立基础概念
# =============================================================================


# =============================================================================
# 核心架构图示
# =============================================================================
"""
┌──────────┐     ┌──────────────┐     ┌──────────────────┐
│ AI Agent │────→│  MCP Server  │────→│ 学生管理 REST API │
│ (Qwen)   │←────│  (FastMCP)   │←────│ (FastAPI :8000)  │
└──────────┘     └──────────────┘     └──────────────────┘
     ↑                   ↑                    ↑
用户问"查学生"    MCP 协议标准化      已有的内部系统

关键价值：
1. API 不变：学生管理系统不需要改造
2. 一次封装：MCP Server 写一次，所有兼容模型都能用
3. 安全可控：Tool 按需暴露，最小权限
"""

# =============================================================================
# 运行前安装（如尚未安装）：
#   pip install httpx
# =============================================================================

import os
import sys
import logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 静默 langchain-openai 的代理检测日志
logging.getLogger("langchain_openai").setLevel(logging.WARNING)

from utils.model_utils import get_model


# =============================================================================
# 第一步：学生管理 REST API 客户端
# =============================================================================

class StudentManagementAPI:
    """
    学生管理系统的 REST API 客户端。

    真实场景中，这里调用的是已有的内部系统 API：
    - POST /auth/login       — 登录获取 JWT Token
    - GET  /students/{id}    — 查询单个学生详情（需 user 角色）
    - POST /students/        — 新增学生（需 operator 角色）

    所有接口统一响应格式：
    {"status_code": 1, "message": "成功", "data": {...}}
    """

    BASE_URL = "http://127.0.0.1:8000"

    def __init__(self):
        self._access_token: str | None = None
        self._refresh_token: str | None = None

    def login(self, username: str, password: str) -> bool:
        """
        登录学生管理系统，获取 JWT Token。

        Args:
            username: 用户名
            password: 密码

        Returns:
            True 表示登录成功，False 表示失败
        """
        import httpx
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(
                    f"{self.BASE_URL}/auth/login",
                    json={"username": username, "password": password},
                )
                result = resp.json()
                if result.get("status_code") == 1:
                    data = result.get("data", {})
                    self._access_token = data.get("access_token")
                    self._refresh_token = data.get("refresh_token")
                    return True
                print(f"  ✗ 登录失败：{result.get('message', '未知错误')}")
                return False
        except Exception as e:
            print(f"  ✗ 登录请求失败：{e}")
            return False

    def _get_headers(self) -> dict:
        """返回包含 Bearer Token 的请求头。"""
        if not self._access_token:
            raise RuntimeError("未登录，请先调用 login() 方法")
        return {"Authorization": f"Bearer {self._access_token}"}

    def get_student(self, student_id: int) -> dict | None:
        """
        查询单个学生的详细信息。

        Args:
            student_id: 学生 ID（整数），如 1

        Returns:
            学生信息 dict，如果不存在返回 None
        """
        import httpx
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(
                    f"{self.BASE_URL}/students/{student_id}",
                    headers=self._get_headers(),
                )
                if resp.status_code == 401:
                    print(f"  ✗ 鉴权失败，Token 可能已过期")
                    return None
                result = resp.json()
                if result.get("status_code") == 1 and result.get("data"):
                    return result["data"]
                return None
        except Exception:
            return None

    def add_student(self, student_data: dict) -> dict:
        """
        新增学生。

        Args:
            student_data: 学生数据 dict，必填 class_id + name
                可选：gender, age, native_place, graduate_school,
                major, education, enrollment_time, graduation_time, consultant_id

        Returns:
            API 原始响应 dict
        """
        import httpx
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(
                    f"{self.BASE_URL}/students/",
                    json=student_data,
                    headers=self._get_headers(),
                )
                if resp.status_code == 401:
                    return {"status_code": 0, "message": "鉴权失败，Token 可能已过期", "data": None}
                if resp.status_code == 403:
                    return {"status_code": 0, "message": "权限不足，需要 operator 角色", "data": None}
                return resp.json()
        except Exception as e:
            return {"status_code": 0, "message": f"连接失败：{e}", "data": None}

    def check_server_health(self) -> bool:
        """
        检查学生管理系统是否可达（不鉴权的轻量检查）。

        Returns:
            True 表示服务正常运行
        """
        import httpx
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self.BASE_URL}/docs")
                return resp.status_code == 200
        except Exception:
            return False


# =============================================================================
# 第二步：用 FastMCP 封装为 MCP Server
# =============================================================================

def create_mcp_server(student_api: StudentManagementAPI):
    """
    将学生管理 REST API 封装为 MCP Server。

    关键原则：
    1. 一个 Tool 只暴露一个业务能力（最小权限）
    2. 输入参数有清晰文档，方便 LLM 理解
    3. Tool 返回自然语言描述，方便 LLM 理解
    4. 写操作（如新增学生）在 Tool 内做基本校验
    """
    from mcp.server.fastmcp import FastMCP

    # 实例化mcp服务
    mcp = FastMCP("学生管理系统")

    # ---- Tool 1: 查询学生 ----
    @mcp.tool()
    def query_student(student_id: int) -> str:
        """查询指定学生的详细信息，包括姓名、班级、专业、入学时间等。
        参数 student_id 为学生 ID（整数），如 1。
        """
        student = student_api.get_student(student_id)
        if not student:
            return f"未找到学号 {student_id} 的记录。请确认学号是否正确。"
        return (
            f"学生 {student.get('name', '未知')}（ID: {student.get('id', student_id)}）详情：\n"
            f"  班级：{student.get('class_id', '未分配')}\n"
            f"  性别：{student.get('gender', '未登记')}\n"
            f"  年龄：{student.get('age', '未登记')}\n"
            f"  籍贯：{student.get('native_place', '未登记')}\n"
            f"  毕业院校：{student.get('graduate_school', '未登记')}\n"
            f"  专业：{student.get('major', '未登记')}\n"
            f"  学历：{student.get('education', '未登记')}\n"
            f"  入学时间：{student.get('enrollment_time', '未登记')}\n"
            f"  毕业时间：{student.get('graduation_time', '未登记')}\n"
            f"  顾问 ID：{student.get('consultant_id', '未分配')}"
        )

    # ---- Tool 2: 新增学生 ----
    @mcp.tool()
    def add_student(
        name: str,
        class_id: int,
        gender: str = "",
        age: int = 0,
        native_place: str = "",
        graduate_school: str = "",
        major: str = "",
        education: str = "",
        enrollment_time: str = "",
        graduation_time: str = "",
        consultant_id: int = 0,
    ) -> str:
        """新增一名学生记录。必填参数：name（姓名）、class_id（班级 ID，整数）。
        可选参数：gender（性别）、age（年龄）、native_place（籍贯）、
        graduate_school（毕业院校）、major（专业）、education（学历）、
        enrollment_time（入学时间）、graduation_time（毕业时间）、
        consultant_id（顾问 ID，整数）。
        """
        payload = {
            "name": name,
            "class_id": class_id,
        }
        # 只添加非空/非默认值的可选字段
        optional = {
            "gender": gender,
            "age": age,
            "native_place": native_place,
            "graduate_school": graduate_school,
            "major": major,
            "education": education,
            "enrollment_time": enrollment_time,
            "graduation_time": graduation_time,
            "consultant_id": consultant_id,
        }
        for key, value in optional.items():
            if value not in ("", 0):
                payload[key] = value

        result = student_api.add_student(payload)
        if result.get("status_code") == 1:
            new_id = result.get("data", {}).get("id", "未知")
            return f"✓ 成功添加学生 '{name}'，系统分配 ID: {new_id}。"
        return f"✗ 添加学生失败：{result.get('message', '未知错误')}"

    return mcp


# =============================================================================
# 到这里为止，MCP的服务已经创建完成，其实就是把学生管理系统的API 接口包装成了MCP服务
# 第三步：在 LangChain Agent 中接入学生管理 MCP 服务
# =============================================================================

def demo_student_mcp_agent():
    """
    模拟真实场景：教务助理通过 AI 助手处理日常学生管理工作。

    流程：
    1. 连接学生管理系统 API
    2. 创建 MCP Server（封装 2 个工具）
    3. 方式 A：直接调用 MCP Tool
    4. 方式 B：LangChain Agent 自动选工具
    """
    import asyncio

    print("\n" + "=" * 70)
    print("  企业 MCP 实战：AI 教务助理接入学生管理系统")
    print("=" * 70 + "\n")

    # ---- 3.1 连接学生管理系统 ----
    print("【1】连接学生管理系统 API（http://127.0.0.1:8000）...")
    student_api = StudentManagementAPI()

    healthy = student_api.check_server_health()
    if not healthy:
        print("  ✗ 无法连接到学生管理系统")
        print("  请确认 FastAPI 服务已启动：")
        print("    cd C:\\python_project\\AI409TEst\\student_management")
        print("    uvicorn main:app --reload\n")
        return
    print("  ✓ 学生管理系统可达")

    # ---- 3.1.5 登录鉴权 ----
    print("\n【2】登录获取 JWT Token...")
    username = input("  请输入用户名: ").strip()
    password = input("  请输入密码: ").strip()
    if not username or not password:
        print("  ✗ 用户名或密码为空，跳过演示\n")
        return
    logged_in = student_api.login(username, password)
    if not logged_in:
        print("  ✗ 登录失败，无法继续演示\n")
        return
    print("  ✓ 登录成功，Token 已获取")

    # ---- 3.2 创建 MCP Server ----
    print("\n【3】封装为 MCP Server...")
    mcp_server = create_mcp_server(student_api)
    print("  ✓ MCP Server 已创建")
    print("  提供的工具：")
    print("    - query_student  — 查询学生详情")
    print("    - add_student    — 新增学生")


    # ---- 3.4 方式 B：接入 LangChain Agent ----
    print("\n" + "-" * 60)
    print("【5】方式 B：接入 LangChain Agent（AI 自动选工具）")
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
    @tool
    def query_student(student_id: int) -> str:
        """查询学生详情。student_id 为学号（整数），如 1"""
        r = asyncio.run(mcp_server.call_tool("query_student", {"student_id": student_id}))
        return r[0][0].text

    @tool
    def add_student(name: str, class_id: int, **kwargs) -> str:
        """新增学生。name 姓名，class_id 班级 ID（整数），其他参数可选"""
        params = {"name": name, "class_id": class_id}
        params.update(kwargs)
        r = asyncio.run(mcp_server.call_tool("add_student", params))
        return r[0][0].text

    langchain_tools = [query_student, add_student]
    print(f"  ✓ 已将 {len(langchain_tools)} 个 MCP 工具转换为 LangChain 工具")

    # 创建 Agent
    agent = create_agent(
        model,
        tools=langchain_tools,
        system_prompt=(
            "你是教务 AI 助手，可以访问学生管理系统。\n"
            "可用工具：\n"
            "- query_student: 按学号查询学生详细信息\n"
            "- add_student: 录入新学生信息\n\n"
            "回复原则：\n"
            "1. 先查后说：涉及学生信息时，先调工具再回答\n"
            "2. 录入确认：新增学生时，确认关键信息（姓名、班级）后再调用\n"
            "3. 友好简洁：用自然语言回复，不要直接输出 JSON"
        ),
    )

    # 模拟教务对话
    conversations = [
        # 场景 1：单一查询
        "帮我查一下学号为 1 的学生信息",
        # 场景 2：查询不存在的记录（测试 Agent 错误处理）
        "学号 99999 的学生是谁？",
        # 场景 3：新增学生
        "请帮我录入一个新学生：姓名李小红，班级 ID 为 390，专业软件工程，学历本科",
    ]

    for i, question in enumerate(conversations, 1):
        print(f"\n  ┌─ 教务对话 {i} ──────────────────────")
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
    print("  学生管理 MCP 实战演示完成")
    print("=" * 70)
    print()
    print("  回顾关键步骤：")
    print("    1. 创建 StudentManagementAPI（HTTP 客户端，调用真实 REST API）")
    print("    2. 登录获取 JWT Token（user 角色可查询，operator 角色可新增）")
    print("    3. 用 FastMCP 定义 2 个 Tool（query_student + add_student）")
    print("    4. 方式 A：直接调用   → asyncio.run(mcp_server.call_tool(...))")
    print("    5. 方式 B：接入 Agent  → @tool 包装 + asyncio.run(mcp_server.call_tool(...))")
    print()


# =============================================================================
# 第四步：企业实际部署建议
# 做的事情，就是把原本的已经存在的业务系统，不管是什么语言开发的，对它进行解耦，把已经存在的API封装成mcp服务，让不同的大模型都可以调用
# =============================================================================

def student_mcp_deployment_guide():
    """学生管理 MCP Server 从开发到上线的建议"""
    print("\n" + "=" * 70)
    print("  学生管理 MCP Server 部署指南")
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
│    def query_student(student_id: int, auth_token: str) -> str:  │
│        if not verify_token(auth_token):                          │
│            return "认证失败"                                     │
│                                                                  │
│ 2. 权限控制：[查询学生] 所有教务可用，[新增学生] 仅管理员          │
│    在 Tool 函数内部加角色检查                                    │
│                                                                  │
│ 3. 日志审计：每次 Tool 调用记录：谁、什么操作、结果               │
│    import logging                                                │
│    logging.info(f"Tool call: query_student by {user}, ...")     │
│                                                                  │
│ 4. 数据校验：新增学生时，对敏感字段做校验                         │
│    - 姓名：非空、长度限制                                        │
│    - 班级 ID：是否存在                                           │
│    - 日期格式：YYYY-MM-DD                                        │
│                                                                  │
│ 5. 速率限制：防止 AI 频繁调用压垮学生管理系统                     │
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
│       "student-system": {                                        │
│         "command": "python",                                     │
│         "args": ["student_mcp_server.py"]                        │
│       }                                                          │
│     }                                                            │
│   }                                                              │
│                                                                  │
│ 方式 B — SSE 远程部署（生产）：                                   │
│   Server 部署在云上，Client 通过 HTTPS 连接                       │
│   配置示例：                                                     │
│   {                                                              │
│     "mcpServers": {                                              │
│       "student-system": {                                        │
│         "url": "https://api.school.com/mcp/students",            │
│         "transport": "sse",                                      │
│         "headers": {"Authorization": "Bearer <token>"}           │
│       }  │
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
    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding='utf-8', errors='replace'
        )

    # 1. 部署指南（可选，取消注释查看）
    # student_mcp_deployment_guide()

    # 2. 实战演示
    demo_student_mcp_agent()

    print("=" * 70)
    print("  对比学习：enterprise_api_mcp_demo.py（订单管理，6 个工具）")
    print("=" * 70 + "\n")
