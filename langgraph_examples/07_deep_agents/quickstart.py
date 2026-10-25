# =============================================================================
# DeepAgents 快速入门 — 一行创建 Agent + 文件系统操作
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 用 create_deep_agent() 一行创建带完整能力的 Agent
#   ✅ 配置 FilesystemBackend 让 Agent 读写文件
#   ✅ 理解 DeepAgent 与 create_react_agent() 的核心区别
#
# 运行前检查：
# 1. 已安装依赖：pip install deepagents langgraph langchain-core
# 2. 配置 .env 中的 ALIYUN_API_KEY（使用云端 Qwen 模型）
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
from utils.model_utils import get_qwen_client


# =============================================================================
# 示例: 文件操作 — Agent 自主读写文件
# =============================================================================

def agent_with_filesystem():
    """
    文件操作是 DeepAgents 的核心特色。

    配置 FilesystemBackend 后，Agent 自动拥有：
      ls / read_file / write_file / edit_file / glob / grep

    演示：让 Agent 创建一个文件并写入内容。
    """
    print(f"\n-- 示例: 文件操作 — Agent 自主读写文件")

    try:
        from deepagents import create_deep_agent
        from deepagents.backends import FilesystemBackend
    except ImportError:
        print("  【跳过】请安装 deepagents：pip install deepagents")
        return

    # 1. 创建临时工作目录（安全沙箱）
    workspace = tempfile.mkdtemp(prefix="deepagent_workspace_", dir="c:\\workspace")
    print(f"  【工作目录】{workspace}")

    # 2. 配置文件系统后端
    # virtual_mode=False: 允许在 root_dir 内使用绝对路径
    backend = FilesystemBackend(root_dir=workspace, virtual_mode=True)

    model = get_qwen_client()
    if model is None:
        print("  【跳过】请先在 .env 配置 ALIYUN_API_KEY")
        return

    # 3. 创建带文件系统的 Agent
    agent = create_deep_agent(
        model=model,
        backend=backend,
        system_prompt=(
            "你是一个写作助手。你可以使用文件系统来读写文件。"
            "当用户要求创建文件时，请在当前工作目录操作。"
        ),
    )

    # 4. 让 Agent 创建文件
    result = agent.invoke({
        "messages": [HumanMessage(
            content="在工作目录下创建一个cal.py文件，实现一个冒泡排序的函数"
        )]
    })


    # 6. 展示消息流（含调试信息）
    print(f"  【消息流】共 {len(result['messages'])} 条")
    for i, msg in enumerate(result["messages"]):
        msg_type = msg.__class__.__name__
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            preview = f"调用工具: {[tc['name'] for tc in msg.tool_calls]}"
        elif hasattr(msg, 'content') and msg.content:
            preview = msg.content
        else:
            preview = f"(空内容 — 可能模型未响应, 请确认 .env 已配置 ALIYUN_API_KEY)"
        print(f"    [{i}] {msg_type}: {preview}")

    print()


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  DeepAgents 快速入门")
    print("  一行创建完整 Agent + 文件系统操作")
    print("=" * 70 + "\n")

    agent_with_filesystem()

    print("=" * 70)
    print("  总结：")
    print("    create_deep_agent() = 一行获得完整 Agent 能力")
    print("    FilesystemBackend 让 Agent 可以安全地读写文件")
    print("    这是 DeepAgents 区别于其它 Agent 框架的核心优势")
    print("=" * 70 + "\n")
