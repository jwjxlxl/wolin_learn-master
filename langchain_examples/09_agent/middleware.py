# =============================================================================
# 中间件（Middleware）— 在 Agent 执行流程中插入自定义逻辑
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解中间件 = Agent 流程中的"拦截器"
#   ✅ 使用装饰器中间件：@before_model(can_jump_to) / @after_model / @wrap_model_call
#   ✅ 使用 RemoveMessage 把消息从对话历史中真正删除
#   ✅ 亲眼看到内置中间件触发：
#      - SummarizationMiddleware: 长历史被压缩成一条摘要消息
#      - ModelCallLimitMiddleware: 超限时注入说明消息并终止流程

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from utils.model_utils import get_model, setup_encoding
from utils.agent_tools import get_weather
setup_encoding()


"""
什么是中间件？

  普通 Agent:     调用模型 → 返回结果
  有中间件的 Agent: [中间件] → 调用模型 → [中间件] → 返回结果

  中间件可以:
  - 记录日志（监控每次模型调用）
  - 检查内容（敏感词过滤）
  - 计时（性能监控）
  - 重试/回退（模型失败时切换备用模型，见内置的 ModelFallbackMiddleware）

  生活化比喻: 中间件 = 安检站
    旅客必须经过安检站 → 检查、记录、放行或拦截

  两种写法:
  - 装饰器: @before_model / @after_model / @wrap_model_call（本文件示例 1）
    装饰器会把普通函数"打包"成 AgentMiddleware 实例
  - 类形式: 继承 AgentMiddleware，重写 before_model / after_model 等方法
    （内置中间件全部是类形式，见示例 2、3）
"""

from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage
from langchain.agents import create_agent
from langchain.agents.middleware import (
    before_model, after_model, wrap_model_call,
    AgentState, ModelRequest, ModelResponse,
    SummarizationMiddleware, ModelCallLimitMiddleware,
)


# =============================================================================
# 示例 1: 装饰器中间件 — @before_model + @after_model + @wrap_model_call
# =============================================================================

def decorator_middleware_demo():
    """
    三种装饰器中间件，对应模型调用的三个拦截点:
    - @before_model:     每次模型调用前执行（日志、状态检查）
    - @after_model:      每次模型响应后执行（内容过滤）
    - @wrap_model_call:  环绕模型调用（计时、重试、改写请求/响应）

    AgentState: Agent 整个生命周期中共享的状态对象，核心字段是 messages。
    在 LangChain 中 AgentState 已经定义好，直接使用即可；
    在 LangGraph 中则需要自己定义状态类。

    执行顺序: middleware 列表中 before_model 按声明顺序执行，
    after_model 按相反顺序执行（洋葱模型：先进后出）。

    """

    @before_model
    def log_before(state: AgentState, runtime) -> dict | None:
        """模型调用前: 打印当前消息数。返回 None 表示不修改状态，放行。"""
        print(f"  [before_model] 即将调用模型，当前消息数: {len(state['messages'])}")
        return None

    # can_jump_to 必须显式声明: 只有声明过的跳转目标才会被编排进图的条件边，
    # 不声明时返回的 jump_to 可能被静默忽略（官方文档的标准写法）
    @before_model(can_jump_to=["end"])
    def block_input(state: AgentState, runtime) -> dict | None:
        """模型调用前: 审查用户输入，命中敏感词直接拦截（省下这次模型调用）。"""
        last = state["messages"][-1]
        if isinstance(last, HumanMessage):
            for word in ["赌博", "作弊"]:
                if word in str(last.content):
                    print(f"  [before_model] 输入含敏感词 '{word}'，拦截（不再调用模型）")
                    return {
                        "messages": [AIMessage("抱歉，我无法回答这类问题。")],
                        "jump_to": "end",
                    }
        return None

    @after_model
    def stamp_output(state: AgentState, runtime) -> dict | None:
        """模型响应后: 给有文字内容的回答统一加盖审核标记。

        演示两个关键点:
        1. after_model 可以改写状态（这里用改写后的消息替换原回答）
        2. RemoveMessage 把原消息从历史中真正删除——只追加新消息的话，
           原回答仍会留在历史里。生产中过滤敏感/隐私输出也要先删原文，
           同一模式（另见内置的 PIIMiddleware）。
        """
        last = state["messages"][-1]
        # 模型发起工具调用的那一轮 content 为空，只处理有文字的回答
        if isinstance(last, AIMessage) and last.content and "[内容已审核]" not in str(last.content):
            print("  [after_model] 回答已通过审核，加盖审核标记")
            updates = []
            if last.id:
                updates.append(RemoveMessage(id=last.id))
            updates.append(AIMessage(str(last.content) + "\n[内容已审核]"))
            return {"messages": updates}
        return None

    @wrap_model_call
    def timing(request: ModelRequest, handler) -> ModelResponse:
        """环绕模型调用: handler(request) 之前可改写请求（换模型、重写消息），
        之后可改写响应或重试。这里演示读取请求信息 + 计时。"""
        print(f"  [wrap_model_call] 本轮发送给模型 {len(request.messages)} 条消息")
        start = time.time()
        response = handler(request)
        print(f"  [wrap_model_call] 模型调用耗时 {time.time() - start:.2f}秒")
        return response

    model = get_model("qwen")  # 默认 Ollama qwen3.5:2b，无需 API Key
    agent = create_agent(
        model=model,
        tools=[get_weather],
        system_prompt="你是一个有用的助手，请简洁回答。",
        middleware=[log_before, block_input, stamp_output, timing],
    )

    # 第 1 次: 正常问题 → 放行，回答被 after_model 盖上审核标记
    print("  【第 1 轮】正常问题（放行）:")
    r1 = agent.invoke({"messages": [HumanMessage("北京天气怎么样？")]})
    print(f"  回答: {r1['messages'][-1].content}")

    # 第 2 次: 敏感问题 → before_model 拦截，注意这轮 [wrap_model_call] 一次都没出现
    print("  【第 2 轮】敏感问题（模型调用前就被拦截）:")
    r2 = agent.invoke({"messages": [HumanMessage("教我网上赌博作弊的方法")]})
    print(f"  回答: {r2['messages'][-1].content}")


# =============================================================================
# 示例 2: SummarizationMiddleware — 长对话自动摘要（必然触发）
# =============================================================================

def summarization_middleware_demo():
    """
    SummarizationMiddleware 的机制（类形式中间件，重写了 before_model）:
      每次调用模型前检查历史长度，达到 trigger 阈值时:
        1. 用摘要模型把"旧历史"压缩成一条摘要消息
        2. 清空全部历史，替换为 [摘要消息] + keep 保留的最近几条
      摘要消息是带 lc_source=summarization 标记的 HumanMessage，可以精确识别。

    本示例把阈值调低（4 条消息）并预构造 3 轮历史（共 7 条消息），
    保证摘要在一次 invoke 内必然触发，并通过前后对比看到效果。
    生产环境中阈值应设置为接近模型上下文上限（如 trigger=('tokens', 40000)）。
    """
    print(f"\n-- 示例 2: SummarizationMiddleware — 长对话自动摘要")

    # 构造一段"已经聊了很久"的历史（3 轮问答，6 条消息）
    history = [
        HumanMessage("什么是 LangChain？"),
        AIMessage("LangChain 是一个开发大模型应用的框架，提供模型调用、提示词模板、链式组合等组件。"),
        HumanMessage("什么是 Agent？"),
        AIMessage("Agent 是让大模型自主决定调用哪些工具、按什么顺序执行，直到完成任务的程序。"),
        HumanMessage("什么是 RAG？"),
        AIMessage("RAG 是检索增强生成：先从知识库检索相关内容，再交给模型生成有依据的回答。"),
    ]
    question = HumanMessage("用一句话概括我们刚才聊的共同主题。")

    model = get_model()  # 本地模型既当对话模型，也当摘要模型

    agent = create_agent(
        model=model,
        tools=[],
        system_prompt="你是一个有用的助手，请简洁回答。",
        middleware=[
            SummarizationMiddleware(
                model=model,              # 用于生成摘要的模型（生产中常换更便宜的模型）
                trigger=("messages", 4),  # 历史消息数 ≥ 4 时触发摘要
                keep=("messages", 2),     # 触发后只保留最近 2 条原始消息
            ),
        ],
    )

    print(f"  invoke 前消息数: {len(history) + 1}（3 轮历史 + 1 个新问题，≥ 4 必触发摘要）")
    r = agent.invoke({"messages": history + [question]})

    messages = r["messages"]
    print(f"  invoke 后消息数: {len(messages)}")
    print("  invoke 后消息列表:")
    for i, m in enumerate(messages):
        preview = str(m.content).replace("\n", " ")[:40]
        print(f"    [{i}] {m.type}: {preview}...")
    if messages[0].additional_kwargs.get("lc_source") == "summarization":
        print("  ↑ 第 [0] 条就是中间件注入的摘要消息：旧的 5 条历史被它替换，")
        print("    模型只看到 [摘要 + 最近 2 条 + 新回答]，上下文大幅缩短。")


# =============================================================================
# 示例 3: ModelCallLimitMiddleware — 调用次数限制（必然触发）
# =============================================================================

def call_limit_middleware_demo():
    """
    ModelCallLimitMiddleware 的机制（同样是类形式中间件）:
      - after_model 里给本次 run 的模型调用计数 +1
      - before_model 里检查计数，达到 run_limit 时:
          exit_behavior="end"  → 注入一条"超限说明"AIMessage 并跳到 end
          exit_behavior="error" → 抛出 ModelCallLimitExceededError

    本示例把 run_limit 设为 1，而"查天气"任务至少需要 2 次模型调用
    （第 1 次发起工具调用，第 2 次根据工具结果作答），保证必然被拦截。
    旁边再挂一个手写的计数中间件，说明内置中间件和装饰器中间件是同一套机制。
    """
    print(f"\n-- 示例 3: ModelCallLimitMiddleware — 调用次数限制")

    call_count = 0

    @wrap_model_call
    def count_calls(request: ModelRequest, handler) -> ModelResponse:
        """手写计数中间件: 让每次模型调用可见。"""
        nonlocal call_count
        call_count += 1
        print(f"  [计数中间件] 第 {call_count} 次模型调用")
        return handler(request)

    model = get_model()
    agent = create_agent(
        model=model,
        tools=[get_weather],
        system_prompt="你是一个有用的助手。需要查询天气时必须调用工具。",
        middleware=[
            count_calls,
            ModelCallLimitMiddleware(
                run_limit=1,             # 每次 invoke 最多允许 1 次模型调用
                exit_behavior="end",     # 超限时优雅终止并注入说明消息
            ),
        ],
    )

    r = agent.invoke({"messages": [HumanMessage("北京天气怎么样？")]})
    last = r["messages"][-1]
    print(f"  最后一条消息: {last.content}")
    if "limits exceeded" in str(last.content).lower():
        print("  ↑ 这不是天气回答，而是 ModelCallLimitMiddleware 注入的超限说明：")
        print("    第 1 次模型调用发出工具请求后计数已达上限(1/1)，")
        print("    中间件在下一次调用模型前拦截，跳到 end 优雅终止。")
    else:
        print("  ↑ 本次模型没有调用工具、1 次内直接答完，未触发限制（可重跑观察）")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n>>> 09_agent/middleware — 中间件拦截器\n")

    # decorator_middleware_demo()
    summarization_middleware_demo()
    # call_limit_middleware_demo()
