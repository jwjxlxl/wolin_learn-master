# =============================================================================
# MCP 客户端配置指南 — Claude / Cursor / Codex
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 在 Claude Desktop 中配置 MCP Server
#   ✅ 在 Cursor 编辑器中配置 MCP Server
#   ✅ 在 Codex 中配置 MCP Server
#   ✅ 理解三种客户端配置的区别和选型
#
# 使用场景：
#   你已经在 Python 代码里用 langchain-mcp-adapters 连接过 MCP Server 了
#   （见 mcp_demo.py），但你的同事／用户更想在 IDE 或桌面 AI 助手里直接使用
#   这些 MCP 工具——本文件就是给他们看的配置指南。
#
# 本文件为配置指南，不涉及 Python 执行，可直接阅读或用作教学讲义。
# =============================================================================


# =============================================================================
# 一、通用配置原理
# =============================================================================
"""
所有 MCP 客户端（Claude Desktop / Cursor / Codex / Windsurf 等）配置
都遵循相同的逻辑：

┌──────────────┐                    ┌──────────────┐
│  MCP 客户端   │  ── stdio/SSE ──→  │  MCP Server  │
│ (配置文件)    │                    │  (你的工具)   │
└──────────────┘                    └──────────────┘

核心配置参数（所有客户端通用）：

| 参数        | 说明                                      | stdio 必填 | SSE 必填 |
|------------|-------------------------------------------|:--------:|:------:|
| command    | 启动 Server 的命令（如 python / npx）        |   ✅     |   ❌    |
| args       | 命令参数列表                                |   ✅     |   ❌    |
| url        | 远程 MCP Server 的 SSE 端点                 |   ❌     |   ✅    |
| transport  | 传输方式：'stdio' 或 'sse'                  |   ✅     |   ✅    |
| env        | 环境变量（如 API Key）                       | 可选     | 可选    |
| headers    | HTTP 请求头（SSE 认证）                      |   ❌     | 可选    |
"""


# =============================================================================
# 二、Claude Desktop 配置
# =============================================================================
def claude_desktop_config():
    """
    Claude Desktop 是 Anthropic 官方桌面客户端，原生支持 MCP。

    配置文件位置：
      macOS:   ~/Library/Application Support/Claude/claude_desktop_config.json
      Windows: %APPDATA%/Claude/claude_desktop_config.json
               (如 C:\\Users\\你的用户名\\AppData\\Roaming\\Claude\\claude_desktop_config.json)
    """
    print("=" * 70)
    print("  二、Claude Desktop MCP 配置")
    print("=" * 70)
    print("""
【示例 A — 连接本地 Python MCP Server（stdio）】

将以下内容粘贴到 claude_desktop_config.json 中：

{
  "mcpServers": {
    "weather": {
      "command": "python",
      "args": [
        "C:/沃林AI课程/AI0226_0309课程/wolin_learn-master/mcp_examples/local_weather_server.py"
      ],
      "transport": "stdio"
    }
  }
}

配置完成后：
  1. 完全退出 Claude Desktop（托盘图标 → Quit）
  2. 重新启动 Claude Desktop
  3. 在对话中尝试：'东京天气怎么样？'
  4. Claude 会自动调用 get_weather 工具

【示例 B — 连接远程 MCP Server（SSE）】

{
  "mcpServers": {
    "amap": {
      "url": "https://mcp.amap.com/sse?key=你的高德Key",
      "transport": "sse"
    }
  }
}

【示例 C — 同时配置多个 MCP Server】

{
  "mcpServers": {
    "weather": {
      "command": "python",
      "args": [".../local_weather_server.py"],
      "transport": "stdio"
    },
    "amap": {
      "url": "https://mcp.amap.com/sse?key=你的高德Key",
      "transport": "sse"
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "transport": "stdio",
      "env": {
        "GITHUB_TOKEN": "ghp_xxxxxxxxxxxx"
      }
    }
  }
}

【调试技巧】
  1. 打开 Claude Desktop 的开发者工具（Ctrl+Shift+I 或 Cmd+Option+I）
  2. 查看 Console 标签，搜索 'mcp' 关键词
  3. 如果 Server 启动失败，会显示具体的错误信息
  4. 常见错误：
     - 'command not found' → 确认 python/npx 在 PATH 中
     - 'connection refused' → 检查 transport 类型是否正确
     - 'authentication failed' → 检查 API Key / Token
""")


# =============================================================================
# 三、Cursor 编辑器配置
# =============================================================================

def cursor_config():
    """
    Cursor 是流行的 AI 编辑器，支持在项目级别配置 MCP Server。

    配置文件位置：
      项目根目录/.cursor/mcp.json

    特点：
      - 项目级别：每个项目可以有自己的 MCP Server 配置
      - 与 Claude Desktop 格式兼容（同样基于 mcpServers 结构）
      - 支持 stdio 和 SSE 两种传输
    """
    print("=" * 70)
    print("  三、Cursor 编辑器 MCP 配置")
    print("=" * 70)
    print("""
【示例 A — 连接本地 Python MCP Server】

在项目根目录创建 .cursor/mcp.json：

{
  "mcpServers": {
    "weather": {
      "command": "python",
      "args": [
        "mcp_examples/local_weather_server.py"
      ],
      "transport": "stdio"
    },
    "order-system": {
      "command": "python",
      "args": [
        "mcp_examples/enterprise_api_mcp_demo.py"
      ],
      "transport": "stdio"
    }
  }
}

配置完成后：
  1. Cursor 会自动检测 .cursor/mcp.json 文件
  2. 在 Cursor 的 AI 对话（Cmd+L / Ctrl+L）中尝试提问
  3. Agent 模式（Cmd+I / Ctrl+I）同样可以使用 MCP 工具
  4. 如果 Server 未启动，Cursor 会提示配置错误

【示例 B — 连接远程 MCP Server】

{
  "mcpServers": {
    "amap": {
      "url": "https://mcp.amap.com/sse?key=你的高德Key",
      "transport": "sse"
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
      "transport": "stdio"
    }
  }
}

【Cursor 特有的技巧】
  1. 组合使用：codebase 索引 + MCP 工具 → 极强的代码理解能力
     例：配置 filesystem MCP，让 Agent 能读取任意项目文件
  2. 按项目配置：前端项目连 Figma MCP，后端项目连数据库 MCP
  3. 与 .cursorrules 配合：在 .cursorrules 中提示 Agent 何时使用 MCP 工具
""")


# =============================================================================
# 四、Codex 配置
# =============================================================================

def codex_config():
    """
    Codex（OpenAI 开源项目 Codex CLI / Codex Desktop）同样支持 MCP。

    Codex 的 MCP 配置通过插件系统实现：
      - 插件定义在 .codex-plugin/plugin.json 中
      - MCP Server 作为插件的一种 transport 类型
      - 也支持在 settings.json 中直接配置 mcpServers

    配置方式一：通过插件（推荐，可复用）
    配置方式二：通过 settings.json 直接配置
    """
    print("=" * 70)
    print("  四、Codex MCP 配置")
    print("=" * 70)
    print("""
【方式 A — 创建 MCP 插件（推荐）】

在 .codex-plugin/plugin.json 中定义：

{
  "name": "my-mcp-tools",
  "version": "1.0.0",
  "description": "我的 MCP 工具集（天气 + 高德地图）",
  "mcpServers": {
    "weather": {
      "command": "python",
      "args": [
        "mcp_examples/local_weather_server.py"
      ],
      "transport": "stdio"
    },
    "amap": {
      "url": "https://mcp.amap.com/sse?key=你的高德Key",
      "transport": "sse"
    }
  }
}

然后在 Codex 中启用该插件即可。

【方式 B — settings.json 直接配置】

在 Codex 的 settings.json 中添加：

{
  "mcpServers": {
    "weather": {
      "command": "python",
      "args": ["mcp_examples/local_weather_server.py"],
      "transport": "stdio"
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "transport": "stdio",
      "env": {
        "GITHUB_TOKEN": "ghp_xxxxxxxxxxxx"
      }
    }
  }
}

【Codex 特有的优势】
  1. 多 Agent 协作：可以为不同的子 Agent 配置不同的 MCP Server
     例：前端 Agent → Figma MCP，后端 Agent → 数据库 MCP
  2. Skill 体系：MCP 工具可以被 Skill 编排，形成复合能力
  3. 本项目的 mcp_examples 目录就是为 Codex 教学设计的
""")


# =============================================================================
# 五、三种客户端对比
# =============================================================================

def comparison_table():
    """
    横向对比三种 MCP 客户端的差异，帮助选型。
    """
    print("=" * 70)
    print("  五、三种客户端对比")
    print("=" * 70)
    print("""
| 特性               | Claude Desktop       | Cursor              | Codex               |
|-------------------|---------------------|--------------------|--------------------|
| 适用场景           | 日常 AI 对话         | 代码编写与审查       | 全栈 Agent 开发     |
| 配置方式           | 全局 JSON 文件       | 项目 .cursor/mcp.json | 插件 + settings.json |
| stdio 支持         | ✅                   | ✅                  | ✅                  |
| SSE 支持           | ✅                   | ✅                  | ✅                  |
| 多 Server 同时连接 | ✅ 一个配置文件       | ✅ 一个配置文件      | ✅ 插件/Agent 级别   |
| 环境变量注入       | ✅ env 字段          | ✅ env 字段         | ✅ env 字段         |
| 配置热重载         | 需重启客户端          | 自动检测            | 自动检测            |
| MCP Resources      | ✅                   | ✅                  | ✅                  |
| MCP Prompts        | ✅                   | ✅                  | ✅                  |
| 平台              | macOS / Windows      | macOS / Windows / Linux | macOS / Windows / Linux |
| 适合人群           | 非开发者、管理者      | 开发者              | Agent 开发者        |
""")


# =============================================================================
# 六、常见问题排查
# =============================================================================

def troubleshooting():
    """
    配置 MCP 客户端时最常见的问题和解决方法。
    """
    print("=" * 70)
    print("  六、常见问题排查")
    print("=" * 70)
    print("""
【问题 1】配置后客户端没有显示 MCP 工具
  原因：Server 启动失败或连接超时
  排查：
    1. 先在终端手动运行 Server 脚本，确认能正常启动
       python mcp_examples/local_weather_server.py
    2. 检查 JSON 语法是否正确（多余的逗号最常见）
    3. Claude Desktop：完全退出并重启（托盘图标 → Quit）
    4. Cursor/Codex：重新打开项目或重启编辑器

【问题 2】windows 上 Python 命令找不到
  原因：Windows 系统下可能是 'python' 而非 'python3'
  解决：把 command 改为 'python'（不要用 'python3'）
  验证：在 PowerShell 中运行 where.exe python 确认路径

【问题 3】SSE 连接失败（connection refused 或 401）
  原因：API Key 错误或网络不通
  排查：
    1. 在浏览器中访问 SSE URL，确认返回正常
    2. 检查 Key 是否过期
    3. 检查防火墙/代理设置

【问题 4】多个 Server 中某个启动失败，其他也受影响
  原因：部分客户端在初始化阶段会同时启动所有 Server
        如果某个失败，可能阻止整体初始化
  解决：
    1. 先单独测试每个 Server
    2. 将稳定的 Server 放在配置文件前面
    3. 暂时注释掉不稳定的 Server，逐个排查

【问题 5】工具参数验证失败
  原因：MCP Tool 定义的参数 Schema 与客户端期望不一致
  排查：
    1. 检查 Server 端 @mcp.tool() 的参数类型注解是否正确
    2. 确认使用了标准 Python 类型（str, int, float, bool）
    3. 避免使用自定义类型或复杂嵌套结构
""")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  MCP 客户端配置指南")
    print("  Claude Desktop / Cursor / Codex")
    print("=" * 70 + "\n")

    claude_desktop_config()
    print()
    cursor_config()
    print()
    codex_config()
    print()
    comparison_table()
    print()
    troubleshooting()

    print("\n" + "=" * 70)
    print("  本教程引用的 MCP Server：")
    print("    - local_weather_server.py（天气查询 + 空气质量）")
    print("    - enterprise_api_mcp_demo.py（订单管理系统）")
    print("  ")
    print("  这些 Server 均在本目录下，可直接用于上述配置。")
    print("=" * 70 + "\n")
