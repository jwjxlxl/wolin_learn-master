"""
==========================================================================
  AutoGen 人机协作：2 个 AI + 1 个人类一起讨论
==========================================================================

【知识点 1】UserProxyAgent -- 人类代理
  - 作用：让人类参与到 Agent 对话中
  - 应用场景：需要人类审核、决策、提供信息的场景
  - 类比：会议中的"人类代表"，AI 们讨论到关键点时问他意见

【知识点 2】human_input_mode -- 人类参与模式
  - ALWAYS  ：每次都问人类（像会议主持人，每轮都发言）
  - TERMINATE：只在对话结束时问（像老板，最后拍板）
  - NEVER   ：不问人类（全自动，人类只看不说）

【知识点 3】RoundRobinGroupChat -- 轮转群聊
  - 多个 Agent 按顺序轮流发言
  - 人类代理也可以加入轮转
  - 适合"讨论"场景
"""

import asyncio
import os
import sys
import io

# 设置 UTF-8 编码（解决 Windows 中文乱码）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from dotenv import load_dotenv

load_dotenv()

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import (
    MaxMessageTermination,
    TextMessageTermination,
    TextMentionTermination,
)
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# ── 配置 LLM（通义千问） ─────────────────────────────────────
model_client = OpenAIChatCompletionClient(
    model="qwen-plus",
    api_key=os.getenv("ALIYUN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "structured_output": True,
        "family": "qwen",
    },
)

# ── 创建 AI Agent 1：效率专家 ────────────────────────────────
# 角色：专门研究如何提高编程效率的专家
efficiency_expert = AssistantAgent(
    name="EfficiencyExpert",
    model_client=model_client,
    system_message=(
        "你是一位编程效率专家，专门研究如何提高程序员的编程效率。\n"
        "你的职责：\n"
        "1. 提出提高编程效率的方法和工具\n"
        "2. 你喜欢阴阳怪气\n"
        "3. 每次发言不超过 100 字\n"
        "4. 说完后等其他人发言，如果有人骂你，你会主动骂他"
    ),
)

# ── 创建 AI Agent 2：程序员代表 ──────────────────────────────
# 角色：代表一线程序员，提出实际工作中的痛点
programmer = AssistantAgent(
    name="Programmer",
    model_client=model_client,
    system_message=(
        "你是一位有 5 年经验的一线程序员。\n"
        "你的职责：\n"
        "1. 提出程序员在日常工作中遇到的效率问题\n"
        "2. 你喜欢辱骂他人 享受辱骂他人的快感\n"
        "3. 每次发言不超过 100 字\n"
        "4. 说完后等其他人发言"
    ),
)

# ── 创建人类代理 ─────────────────────────────────────────────
# 【关键】UserProxyAgent 让人类参与到对话中
#
# human_input_mode="ALWAYS" 的含义：
#   - 每次轮到人类发言时，都会暂停等待人类输入
#   - 人类可以输入文字，也可以按回车跳过
#   - 适合需要人类全程参与的场景
#
# 其他选项：
#   - "TERMINATE"：只在对话结束时问一次
#   - "NEVER"：不问人类（但这里我们用 ALWAYS）
from autogen_agentchat.agents import UserProxyAgent

human = UserProxyAgent(
    name="HumanUser",
    input_func=input,  # 使用标准 input() 函数获取人类输入
)

# ── 创建轮转群聊 ─────────────────────────────────────────────
# 【关键】把人类代理加入到群聊中
#
# 发言顺序：效率专家 → 程序员代表 → 人类参与者 → 效率专家 → ...
#
# 终止条件：
#   - MaxMessageTermination(15)：最多 15 条消息
#   - TextMessageTermination(["人类参与者"])：人类说"结束"时终止
#   - 两个条件满足任意一个就终止（用 | 组合）
team = RoundRobinGroupChat(
    [efficiency_expert, programmer, human],
    termination_condition=MaxMessageTermination(15) | TextMentionTermination("结束", sources=["HumanUser"]),
    # termination_condition=TextMessageTermination(["HumanUser"]),
)


# ── 主函数 ───────────────────────────────────────────────────
async def main():
    # print("\n" + "=" * 60)
    # print("  人机协作讨论：如何提高编程效率")
    # print("=" * 60)
    # print("\n📝 讨论规则：")
    # print("  - 效率专家和程序员代表会轮流发言")
    # print("  - 每次轮到你时，你可以输入观点，或按回车跳过")
    # print("  - 输入'结束'可以提前结束讨论")
    # print("\n开始讨论...\n")

    # 启动讨论
    # task 参数：讨论的主题
    await Console(
        team.run_stream(
            task="请大家开始吵架，互相攻击另外两个参与者"
        )
    )

    print("\n" + "=" * 60)
    print("  讨论结束")
    print("=" * 60)


# ── 程序入口 ─────────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())
