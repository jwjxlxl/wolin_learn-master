"""
AutoGen 相声 Demo — 多 Agent 对话演示

AutoGen 简介：
AutoGen 是微软开源的多 Agent 对话框架，核心理念是让多个 AI Agent
通过自然的对话协作完成复杂任务。与单 Agent 调用不同，AutoGen 支持：
  - 多 Agent 编排：让不同角色（如开发者、测试员、产品经理）协同工作
  - 灵活的对话模式：一对一、群聊、轮转、基于条件的自动终止
  - 工具/函数调用：Agent 可以执行代码、调用 API、读写文件
  - 人类参与（Human-in-the-loop）：关键时刻由人类介入决策

本 Demo 展示了最基础的多 Agent 对话模式：
  1. 创建两个性格迥异的 Agent（浩天 vs 思锦）
  2. 用 RoundRobinGroupChat 组织轮转群聊
  3. 用 MaxMessageTermination 控制对话轮数
  4. 用 Console 实时打印对话流

运行方式：确保 .env 中已配置 ALIYUN_API_KEY
"""

import asyncio
import io
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Windows 终端兼容：强制 stdout 使用 UTF-8 编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ── 导入 AutoGen 核心组件 ─────────────────────────────────────
# AssistantAgent：最基础的 Agent 类型，接收系统提示 + 模型客户端即可对话
from autogen_agentchat.agents import AssistantAgent
# Console：将 Agent 对话流实时输出到终端的工具
from autogen_agentchat.ui import Console
# OpenAIChatCompletionClient：兼容 OpenAI 协议的模型客户端（支持 Qwen/GPT 等）
from autogen_ext.models.openai import OpenAIChatCompletionClient

# ── 配置 LLM（用通义千问） ─────────────────────────────────
# AutoGen 使用 OpenAI 兼容协议接入任何大模型
# 这里通过 DashScope（阿里云）的兼容接口调用 qwen-plus
model_client = OpenAIChatCompletionClient(
    model="qwen-plus",
    api_key=os.getenv("ALIYUN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # model_info 告诉 AutoGen 该模型的能力边界
    model_info={
        "vision": False,          # 不支持图像理解
        "function_calling": True, # 支持函数调用（工具使用）
        "json_output": True,      # 支持 JSON 格式输出
        "structured_output": True,# 支持结构化输出（Pydantic 等）
        "family": "unknown",      # 模型家族标识，影响内部路由策略
    },
)

# ── 创建逗哏 Agent ─────────────────────────────────────────
# AssistantAgent 是 AutoGen 的基础构建块：
#   - name：Agent 在对话中的称呼
#   - model_client：绑定的 LLM 客户端
#   - system_message：角色设定（决定说话风格和行为模式）
dougen = AssistantAgent(
    name="haotian",
    model_client=model_client,
    system_message="""你是一个暴躁的贴吧老哥，你叫浩天。
你的任务：
- 疯狂的贬低对方
- 每次发言不超过 50 字
- 喜欢抬杠，喜欢嘲讽

""",
)

# ── 创建捧哏 Agent ─────────────────────────────────────────
# 两个 Agent 共享同一个 model_client，但 system_message 不同
# 这就是多 Agent 协作的核心：不同角色 + 不同提示 = 不同行为
penggen = AssistantAgent(
    name="sijin",
    model_client=model_client,
    system_message="""你是一个知乎的海龟精英， 你叫思锦。
你的任务：
- 喜欢阴阳
- 每次发言不超过 50 字
- 喜欢装逼
""",
)


async def main():
    """
    主函数：组织群聊并启动对话流
    """
    # RoundRobinGroupChat：轮转群聊，让所有 Agent 按顺序依次发言
    from autogen_agentchat.teams import RoundRobinGroupChat
    # MaxMessageTermination：达到最大消息数时自动终止对话
    from autogen_agentchat.conditions import MaxMessageTermination

    # 创建轮转群聊（逗哏和捧哏交替发言）
    team = RoundRobinGroupChat(
        [dougen, penggen],              # 参与对话的 Agent 列表
        termination_condition=MaxMessageTermination(max_messages=20),  # 最多 20 条消息后终止
    )

    print("🎭 相声开始！\n")
    # team.run_stream() 返回一个异步迭代器，Console 会逐条格式化打印
    # task 参数是这次对话的主题/任务描述
    await Console(team.run_stream(task="成都只有一个名额，为了争抢这个名额吵架"))


asyncio.run(main())
