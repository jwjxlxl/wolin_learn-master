# =============================================================================
# 中间件（Middleware）— 在 Agent 执行流程中插入自定义逻辑
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解中间件 = Agent 流程中的"拦截器"
#   ✅ 使用装饰器中间件：@before_model, @after_model, @wrap_model_call
#   ✅ 使用内置中间件：SummarizationMiddleware, ModelCallLimitMiddleware
# =============================================================================

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)


"""
什么是中间件？

  普通 Agent:     调用模型 → 返回结果
  有中间件的 Agent: [中间件] → 调用模型 → [中间件] → 返回结果

  中间件可以:
  - 记录日志（监控每次模型调用）
  - 检查内容（敏感词过滤）
  - 计时（性能监控）
  - 重试/回退（模型失败时切换备用模型）

  生活化比喻: 中间件 = 安检站
    旅客必须经过安检站 → 检查、记录、放行或拦截
"""

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent
from langchain.agents.middleware import (
    before_model, after_model, wrap_model_call,
    AgentState, ModelRequest, ModelResponse,
)
from langchain_ollama import ChatOllama


# =============================================================================
# 示例 1: 装饰器中间件 — @before_model + @wrap_model_call
# =============================================================================

def decorator_middleware_demo():
    """
    三种装饰器中间件:
    - @before_model: 每次模型调用前执行（日志、状态检查）
    - @after_model:  每次模型响应后执行（内容过滤）
    - @wrap_model_call: 包装模型调用（计时、重试）
    """
    print(f"\n-- 示例 1: 装饰器中间件 — 日志 + 计时 + 敏感词过滤")

    @before_model
    def log_before(state: AgentState, runtime) -> dict | None:
        """模型调用前: 打印当前消息数。"""
        print(f"  [before_model] 即将调用模型，消息数: {len(state.get('messages', []))}")
        return None

    @after_model
    def check_response(state: AgentState, runtime) -> dict | None:
        """模型响应后: 检查敏感词。"""
        last = state["messages"][-1]
        if hasattr(last, "content") and last.content:
            for word in ["BLOCKED", "禁止回答"]:
                if word in last.content:
                    print(f"  [after_model] 检测到敏感词 '{word}'，终止")
                    return {"messages": [AIMessage("抱歉，我无法回答这个问题。")], "jump_to": "end"}
        return None

    @wrap_model_call
    def timing(request: ModelRequest, handler) -> ModelResponse:
        """记录每次模型调用的耗时。"""
        import time
        start = time.time()
        response = handler(request)
        print(f"  [wrap_model_call] 耗时 {time.time() - start:.2f}秒")
        return response

    @tool
    def get_weather(city: str) -> str:
        """查询天气。"""
        return {"北京": "晴，25°C", "上海": "多云，28°C"}.get(city, "暂无数据")

    model = ChatOllama(model="qwen3.5:2b")
    agent = create_agent(model=model, tools=[get_weather],
                         system_prompt="你是一个有用的助手，请简洁回答。",
                         middleware=[log_before, check_response, timing])

    r = agent.invoke({"messages": [HumanMessage("北京天气怎么样？")]})
    print(f"  回答: {r['messages'][-1].content}")


# =============================================================================
# 示例 2: 内置中间件 — Summarization + ModelCallLimit
# =============================================================================

def builtin_middleware_demo():
    """
    LangChain 预置的中间件——无需自己实现。

    SummarizationMiddleware: 对话过长时自动摘要，保留关键信息
    ModelCallLimitMiddleware: 限制模型调用次数，防止无限循环
    """
    from langchain.agents.middleware import SummarizationMiddleware, ModelCallLimitMiddleware

    print(f"\n-- 示例 2: 内置中间件 — 摘要 + 调用限制")

    @tool
    def get_weather(city: str) -> str:
        """查询天气。"""
        return {"北京": "晴，25°C", "上海": "多云"}.get(city, "暂无数据")

    model = ChatOllama(model="qwen3.5:2b")

    # 用本地轻量模型做摘要（省钱）
    summary_model = ChatOllama(model="qwen3.5:2b")

    agent = create_agent(
        model=model, tools=[get_weather],
        system_prompt="你是一个有用的助手。",
        middleware=[
            SummarizationMiddleware(
                model=summary_model,        # 用于生成摘要的模型
                max_tokens_before_summary=4000,  # 超过 4000 token 触发
                messages_to_keep=10,         # 保留最近 10 条
            ),
            ModelCallLimitMiddleware(
                run_limit=10,       # 每次运行最多 10 次模型调用
                exit_behavior="end",  # 达到限制时优雅终止
            ),
        ],
    )

    r = agent.invoke({"messages": [HumanMessage("北京天气怎么样？")]})
    print(f"  回答: {r['messages'][-1].content}")
    print("  内置中间件已就绪: 长对话自动摘要 + 调用次数限制")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n>>> 09_agent/middleware — 中间件拦截器\n")

    decorator_middleware_demo()
    builtin_middleware_demo()

    # 接下来学习: human_in_the_loop.py（人在回路）
