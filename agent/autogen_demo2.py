"""
==========================================================================
  多 Agent 协作代码审查 -- 基于 AutoGen 0.7.x（通义千问 qwen-plus）
==========================================================================


1. 什么是 Multi-Agent（多智能体）？
   - 单个 Agent = 一个 LLM + 一个角色设定（system message）
   - 多 Agent = 多个不同角色的 Agent 协作完成一个复杂任务
   - 类比：一个公司里有不同岗位的职员，各司其职、相互配合

2. AutoGen 框架的核心组件：
   - AssistantAgent     ：Agent 个体，拥有自己的角色设定和模型
   - RoundRobinGroupChat：轮转群聊，Agent 按顺序依次发言
   - TextMessage        ：Agent 之间传递的消息
   - Console            ：把对话过程打印到终端的辅助工具

3. 多 Agent 协作的工作流程（以代码审查为例）：
   +--------------+     +--------------+     +--------------+
   |  代码阅读者   |---->|   Bug 猎人    |---->| 审查总结者    |
   |  (读懂代码)   |     |  (找问题)     |     | (汇总结论)    |
   +--------------+     +--------------+     +--------------+
   第一步：理解代码      第二步：发现缺陷      第三步：给出建议

4. 为什么不用单个 Agent 一次性完成？
   - 单一 Agent 容易顾此失彼，遗漏某些视角
   - 拆分角色后，每个 Agent 的 system_message 更聚焦，输出质量更高
   - 后一个 Agent 能看到前一个 Agent 的完整输出，形成接力

==========================================================================
"""

import os
import sys
import io
import asyncio
from dotenv import load_dotenv

# --------------------------------------------------------------------------
# 【知识点 1】Windows 控制台编码
# --------------------------------------------------------------------------
# Windows 命令行默认使用 GBK 编码，无法显示 emoji 等 Unicode 字符。
# 把 stdout 包装成 UTF-8，这样 print() 就不会报编码错误。
# --------------------------------------------------------------------------
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

load_dotenv()

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient


# --------------------------------------------------------------------------
# 【知识点 2】模型客户端（Model Client）
# --------------------------------------------------------------------------
# AutoGen 0.7.x 把用什么 LLM 独立出来了，叫 model_client。
# 这样你可以轻松切换模型（GPT-4 / Claude / qwen / 本地模型等），
# 而不需要修改每个 Agent 的创建代码。
#
# 这里的配置等价于：
#   - 模型名称   ：qwen-plus（通义千问）
#   - API 地址   ：DashScope 兼容 OpenAI 协议的 endpoint
#   - 认证方式   ：从环境变量 DASHSCOPE_API_KEY 读取密钥
#
# model_info 是必填项，因为 autogen 需要知道模型的能力特征：
#   - function_calling：模型能否调用工具/函数
#   - json_output     ：模型能否输出结构化 JSON
#   - structured_output：模型能否遵循 JSON Schema 输出
#   - vision          ：模型能否理解图片
#   - family          ：模型系列名称（用于内部路由）
# --------------------------------------------------------------------------
model_client = OpenAIChatCompletionClient(
    model="qwen-plus",
    api_key=os.getenv("ALIYUN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_info={
        "vision": False,          # qwen-plus 在此配置下不支持视觉理解
        "function_calling": True, # 支持函数调用（工具调用）
        "json_output": True,      # 支持 JSON 格式输出
        "structured_output": True,# 支持结构化输出（JSON Schema）
        "family": "qwen",         # 模型系列
    },
)


# --------------------------------------------------------------------------
# 【知识点 3】AssistantAgent -- 单个智能体
# --------------------------------------------------------------------------
# 每个 Agent 由两个核心要素定义：
#   1) name（名称）       ：Agent 的身份标识，在对话中显示为发言者
#                           注意：只能是英文字母、数字、下划线、连字符
#                           中文名称会导致 autogen 内部校验失败！
#
#   2) system_message（系统提示词）：相当于给 Agent 设定的人设或岗位职责
#     - 它会被放在每条发送给 LLM 的消息最前面（role: system）
#     - LLM 会根据这个提示词调整回答的语气、视角、风格
#     - 越具体越好，比如你是一名资深 Python 工程师 比 你是程序员 效果更好
#
#   3) model_client（模型客户端）：指定该 Agent 使用哪个 LLM
# --------------------------------------------------------------------------

# 角色 1：代码阅读者
#   职责：通读代码，理解它在做什么，用简洁的语言描述出来
#   类比：新员工入职后先看代码，写一份代码读懂笔记给同事
code_reader = AssistantAgent(
    name="CodeReader",  # 英文命名！autogen 0.7.x 不允许中文名
    model_client=model_client,
    system_message=(
        "你是一名资深 Python 工程师，角色是【代码阅读者】。\n"
        "你的职责：\n"
        "1. 仔细阅读用户提供的代码\n"
        "2. 用简洁的中文解释这段代码的功能和逻辑\n"
        "3. 指出代码中使用的关键函数、变量和数据流向\n"
        "4. 每次发言控制在 150 字以内\n"
        "你的输出将作为后续 Bug 查找的基础，请务必准确、全面。"
    ),
)

# 角色 2：Bug 猎人
#   职责：基于代码阅读者的分析，专门挑毛病——逻辑错误、边界情况、安全隐患等
#   类比：QA 测试员，专门找代码的漏洞和缺陷
bug_finder = AssistantAgent(
    name="BugFinder",
    model_client=model_client,
    system_message=(
        "你是一名资深测试工程师，角色是【Bug 猎人】。\n"
        "你的职责：\n"
        "1. 阅读代码阅读者的分析\n"
        "2. 找出代码中的 Bug、逻辑错误和潜在问题\n"
        "3. 特别关注：边界情况（如输入为 0、负数、None、极大值等）\n"
        "4. 每个问题都说明【问题描述】+【可能导致什么后果】\n"
        "5. 每次发言控制在 200 字以内\n"
        "如果代码没有问题，也要明确说出未发现 Bug。"
    ),
)

# 角色 3：审查总结者
#   职责：汇总前两位的意见，给出最终结论和改进建议
#   类比：技术主管做 code review 总结
reviewer = AssistantAgent(
    name="Reviewer",
    model_client=model_client,
    system_message=(
        "你是一名技术主管，角色是【审查总结者】。\n"
        "你的职责：\n"
        "1. 阅读代码阅读者的分析和 Bug 猎人的发现\n"
        "2. 给出结构化的审查结论：\n"
        "   【代码功能】代码阅读者总结的功能\n"
        "   【发现问题】Bug 猎人找到的问题列表\n"
        "   【改进建议】针对每个问题的具体修复方案（给出代码示例）\n"
        "   【总体评价】代码质量评分（A/B/C/D）及理由\n"
        "3. 每次发言控制在 300 字以内\n"
        "你的输出是本次审查的最终结论。"
    ),
)


# --------------------------------------------------------------------------
# 【知识点 4】RoundRobinGroupChat -- 轮转群聊（多 Agent 协作的核心）
# --------------------------------------------------------------------------
# RoundRobinGroupChat 让多个 Agent 按顺序轮流发言，形成一个讨论组。
#
# 工作流程（以 3 个 Agent、max_messages=6 为例）：
#
#   第 1 轮：用户发起任务 --> CodeReader 回应（阅读代码）
#   第 2 轮：CodeReader 的输出 --> BugFinder 回应（找问题）
#   第 3 轮：BugFinder 的输出 --> Reviewer 回应（总结结论）
#   第 4 轮：Reviewer 的总结 --> CodeReader 补充...
#   ...
#   第 6 轮：达到 max_messages 上限 --> 自动终止
#
# 关键理解：
#   - 每个 Agent 都能看到前面所有 Agent 的发言（共享上下文）
#   - 发言顺序就是创建 team 时传入的 Agent 列表顺序
#   - termination_condition 决定何时结束对话
#      * MaxMessageTermination：达到消息条数就停（最常用）
#      * TextMentionTermination：出现特定关键词就停
#      * 多个条件可以组合（OrTermination / AndTermination）
# --------------------------------------------------------------------------

# 创建轮转群聊团队：3 个 Agent 按 CodeReader --> BugFinder --> Reviewer 顺序轮流发言
team = RoundRobinGroupChat(
    [code_reader, bug_finder, reviewer],
    termination_condition=MaxMessageTermination(max_messages=7),
    # max_messages=6 的含义：
#   总共最多产生 6 条 Agent 回复消息（不计用户初始消息）
#   3 个 Agent x 每人最多发言 2 次 = 6 条
#   如果希望多轮深入讨论，可以调大这个数字
)


# --------------------------------------------------------------------------
# 【知识点 5】异步执行与 Console 输出
# --------------------------------------------------------------------------
# AutoGen 0.7.x 的对话引擎是异步的（async/await），原因：
#   1. LLM API 调用是网络请求，异步可以避免阻塞
#   2. 多个 Agent 可能需要并行处理（在更复杂的编排中）
#
# asyncio.run(main()) 是 Python 标准库提供的异步入口
#
# Console() 的作用：
#   - 实时流式输出对话过程（像看直播一样，不是等全部完成才显示）
#   - 自动格式化每条消息，显示发言者名称
#   - 显示 Token 消耗统计
#
# team.run_stream(task='...') 发起一个任务：
#   - task 参数 = 用户的第一条消息（发给所有 Agent）
#   - 返回一个异步迭代器，Console 消费它来打印输出
# --------------------------------------------------------------------------
async def main():
    # 待审查的示例代码（故意写了一个 Bug：折扣计算错误）
    code = '''
def calculate_discount(price, discount):
    """计算折后价格。
    price: 原价（浮点数）
    discount: 折扣（0-100 之间的整数，如 20 表示打 8 折）
    返回：折后价格
    """
    return price * discount
'''

    # 任务描述：告诉团队要做什么
    task = (
        f"请审查以下 Python 代码，找出其中的问题并给出改进建议：\n\n"
        f"`python\n{code}`"
    )

    # Console 会实时打印每一条对话消息
    # 输出格式示例：
    #   ---------- TextMessage (CodeReader) ----------
    #   这段代码的功能是计算折后价格...
    #
    #   ---------- TextMessage (BugFinder) ----------
    #   我发现以下问题：
    #   1. 折扣计算逻辑错误...
    await Console(team.run_stream(task=task))


# --------------------------------------------------------------------------
# 程序入口
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print(" 多 Agent 协作代码审查开始")
    print("=" * 60)
    print()
    asyncio.run(main())
