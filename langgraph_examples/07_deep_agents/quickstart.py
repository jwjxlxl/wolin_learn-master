# =============================================================================
# DeepAgents 快速入门 — 一行创建 Agent + 文件系统操作
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 用 create_deep_agent() 一行创建带完整能力的 Agent
#   ✅ 添加自定义工具到 DeepAgent
#   ✅ 配置 FilesystemBackend 让 Agent 读写文件
#   ✅ 理解 DeepAgent 与 create_react_agent() 的核心区别
#
# 运行前检查：
# 1. 已安装依赖：pip install deepagents langgraph langchain-core
# 2. 已安装 Ollama 并下载模型：ollama pull qwen3.5:2b
# =============================================================================

import sys
import os
import io
import tempfile
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 确保项目根目录可导入（用于 utils/ 共享模块）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from utils.model_utils import get_model


# =============================================================================
# 示例 1: 最简 DeepAgent — 一行创建 + 自定义工具
# =============================================================================

def simplest_deep_agent():
    """
    最简 DeepAgent：用 create_deep_agent() 一行创建，
    加入一个自定义工具，体验完整的 Agent 能力。

    与 create_react_agent() 的区别：
      - DeepAgent 自动内置了文件系统、任务规划等工具
      - create_react_agent() 只有纯工具调用循环
    """
    print(f"\n-- 示例 1: 最简 DeepAgent — 一行创建 + 自定义工具")

    # 1. 定义自定义工具
    @tool
    def calculate(expression: str) -> str:
        """执行数学表达式计算"""
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"计算错误: {e}"

    # 2. 获取模型
    model = get_model("qwen")
    if model is None:
        print("  【跳过】请安装 Ollama 并下载模型：ollama pull qwen3.5:2b")
        return

    try:
        from deepagents import create_deep_agent
    except ImportError:
        print("  【跳过】请安装 deepagents：pip install deepagents")
        return

    # 3. ★ 一行创建 DeepAgent！
    agent = create_deep_agent(
        model=model,
        tools=[calculate],
        system_prompt="你是一个智能助手。你可以使用计算工具执行数学运算。",
    )

    # 4. 测试
    questions = [
        "123 乘以 456 等于多少？请用计算工具算一下。",
    ]

    for q in questions:
        print(f"  【用户】{q}")
        result = agent.invoke({"messages": [HumanMessage(content=q)]})
        final_msg = result["messages"][-1]
        if hasattr(final_msg, 'content') and final_msg.content:
            print(f"  【回答】{final_msg.content[:120]}...")

        # 展示工具调用
        tool_calls = [
            m for m in result["messages"]
            if hasattr(m, 'tool_calls') and m.tool_calls
        ]
        if tool_calls:
            print(f"  【工具调用】共 {len(tool_calls)} 次工具调用")
            for m in tool_calls:
                for tc in m.tool_calls:
                    print(f"    - {tc['name']}({tc['args']})")
        print()


# =============================================================================
# 示例 2: 文件操作 — Agent 自主读写文件
# =============================================================================

def agent_with_filesystem():
    """
    文件操作是 DeepAgents 的核心特色。

    配置 FilesystemBackend 后，Agent 自动拥有：
      ls / read_file / write_file / edit_file / glob / grep

    演示：让 Agent 创建一个文件并写入内容。
    """
    print(f"\n-- 示例 2: 文件操作 — Agent 自主读写文件")

    try:
        from deepagents import create_deep_agent
        from deepagents.backends import FilesystemBackend
    except ImportError:
        print("  【跳过】请安装 deepagents：pip install deepagents")
        return

    # 1. 创建临时工作目录（安全沙箱）
    workspace = tempfile.mkdtemp(prefix="deepagent_workspace_")
    print(f"  【工作目录】{workspace}")

    # 2. 配置文件系统后端
    # virtual_mode=False: 允许在 root_dir 内使用绝对路径
    backend = FilesystemBackend(root_dir=workspace, virtual_mode=False)

    model = get_model("qwen")
    if model is None:
        print("  【跳过】请安装 Ollama 并下载模型：ollama pull qwen3.5:2b")
        return

    # 3. 创建带文件系统的 Agent
    agent = create_deep_agent(
        model=model,
        backend=backend,
        system_prompt=(
            "你是一个写作助手。你可以使用文件系统来读写文件。"
            "当用户要求创建文件时，请在工作目录{{workspace}}中操作。"
        ),
    )

    # 4. 让 Agent 创建文件
    # print("  【用户】请创建一个名为 hello.txt 的文件，里面写一句问候语。")
    result = agent.invoke({
        "messages": [HumanMessage(
            content="请创建一个名为 hello.txt 的文件，里面写一句问候语。"
        )]
    })

    # 5. 验证文件是否真的被创建
    created_file = os.path.join(workspace, "hello.txt")
    if os.path.exists(created_file):
        with open(created_file, "r", encoding="utf-8") as f:
            content = f.read()
        print(f"  【文件内容】{content.strip()}")
        print("  ✅ Agent 成功创建了 hello.txt！")
    else:
        print("  ⚠️ 文件未创建")
        print(f"  【目录内容】{os.listdir(workspace)}")

    # 6. 展示消息流（含调试信息）
    print(f"  【消息流】共 {len(result['messages'])} 条")
    for i, msg in enumerate(result["messages"]):
        msg_type = msg.__class__.__name__
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            preview = f"调用工具: {[tc['name'] for tc in msg.tool_calls]}"
        elif hasattr(msg, 'content') and msg.content:
            preview = msg.content[:80]
        else:
            preview = f"(空内容 — 可能模型未响应, 请确认 ollama serve 已启动)"
        print(f"    [{i}] {msg_type}: {preview}")

    # 清理
    # try:
    #     if os.path.exists(created_file):
    #         os.remove(created_file)
    #     os.rmdir(workspace)
    # except Exception:
    #     pass
    print()


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  DeepAgents 快速入门")
    print("  一行创建完整 Agent + 文件系统操作")
    print("=" * 70 + "\n")

    # simplest_deep_agent()
    agent_with_filesystem()

    print("=" * 70)
    print("  总结：")
    print("    create_deep_agent() = 一行获得完整 Agent 能力")
    print("    FilesystemBackend 让 Agent 可以安全地读写文件")
    print("    这是 DeepAgents 区别于其它 Agent 框架的核心优势")
    print("=" * 70 + "\n")
