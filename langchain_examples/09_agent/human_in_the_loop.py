import sys
import os
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import model_untils

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


# =============================================================================
# 人在回路 (Human-in-the-Loop) — 完整中断与恢复流程
# =============================================================================
#
# 核心原理：
#   当 Agent 调用被标记为 interrupt_on 的工具时，invoke() 会返回一个
#   包含 __interrupt__ 字段的结果，而不会真正执行该工具。
#   此时 Agent 的执行状态已保存在 checkpointer 中。
#
#   流程：
#     1. agent.invoke(...)  → 运行到中断点，自动暂停
#     2. 检查 result.get("__interrupt__") → 获取待审批的工具调用
#     3. 人类做出决策 → approve / edit / reject
#     4. agent.invoke(Command(resume=...), config=...) → 从断点恢复
#
# 注意：
#   - 必须传入 checkpointer，否则无法保存/恢复中断状态
#   - 恢复调用必须使用相同的 config（相同的 thread_id）
# =============================================================================


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送一封邮件给指定收件人。"""
    print(f"\n  📧 正在发送邮件到: {to}")
    print(f"     主题: {subject}")
    print(f"     正文: {body}")
    return f"邮件已成功发送至 {to}，主题: {subject}"


def human_decision_loop(interrupt) -> dict:
    """
    等待人类输入，直到获得有效决策。

    这是人在回路的核心交互函数 — 阻塞等待，循环校验。

    Args:
        interrupt: Interrupt 对象，其 value 字段包含 action_requests 和 review_configs

    Returns:
        符合格式的决策字典，用于 Command(resume=...)
        格式: {"decisions": [{"type": "approve"}, ...]}
    """
    # 从 interrupt.value 中提取工具信息
    value = interrupt.value
    action_request = value["action_requests"][0]  # 取第一个请求
    tool_name = action_request["name"]
    tool_args = action_request["args"]
    review_config = value["review_configs"][0]
    allowed = review_config["allowed_decisions"]

    print(f"\n{'=' * 50}")
    print(f"⏸️  中断 — 工具调用待审批")
    print(f"{'=' * 50}")
    print(f"  工具: {tool_name}")
    print(f"  参数:")
    for key, val in tool_args.items():
        print(f"    {key}: {val}")
    print(f"{'=' * 50}")

    # 根据 allowed_decisions 动态显示可用选项
    options = []
    if "approve" in allowed:
        options.append("[a]pprove 批准")
    if "edit" in allowed:
        options.append("[e]dit 编辑")
    if "reject" in allowed:
        options.append("[r]eject 拒绝")
    print(f"  请输入决策: {' | '.join(options)}")
    print(f"{'=' * 50}")

    while True:
        choice = input("\n👤 你的决定 > ").strip().lower()

        # 批准 — 直接放行，使用原始参数执行工具
        if choice in ("a", "approve") and "approve" in allowed:
            return {"decisions": [{"type": "approve"}]}

        # 拒绝 — 不让工具执行，把拒绝原因返回给 Agent
        elif choice in ("r", "reject") and "reject" in allowed:
            reason = input("👤 请输入拒绝原因 > ").strip()
            return {
                "decisions": [{
                    "type": "reject",
                    "message": reason or "人类操作员拒绝了此操作",
                }]
            }

        # 编辑后批准 — 人类可以修改工具参数后再放行
        elif choice in ("e", "edit") and "edit" in allowed:
            print(f"\n  当前参数值（留空表示不修改）:")
            edited_args = {}
            for key, val in tool_args.items():
                new_value = input(f"    {key} [{val}] > ").strip()
                edited_args[key] = new_value if new_value else val
            return {
                "decisions": [{
                    "type": "edit",
                    "edited_action": {
                        "name": tool_name,
                        "args": edited_args,
                    },
                }]
            }

        else:
            print(f"  ⚠️  无效输入，请输入 {' / '.join([o.split(']')[0] + ']' for o in options])}")


def run_with_human_approval():
    """
    运行 Agent 并等待人类审批邮件发送。

    这是完整的交互流程：
      1. 用户提出发邮件的请求
      2. Agent 准备发送邮件 → 触发中断
      3. 程序暂停，显示待审批信息
      4. 人类选择 approve/edit/reject
      5. Agent 从断点恢复，根据决策继续执行
    """
    print("=" * 60)
    print("人在回路 — 邮件发送审批")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. 配置人在回路中间件
    # -------------------------------------------------------------------------
    hitl = HumanInTheLoopMiddleware(
        interrupt_on={
            # 发送邮件需要审批
            "send_email": {
                "allowed_decisions": ["approve", "edit", "reject"],
            },
        }
    )

    # -------------------------------------------------------------------------
    # 2. 获取模型
    # -------------------------------------------------------------------------
    model = model_untils.get_qwen_client()
    if model is None:
        print("【跳过】未配置阿里云 API Key，无法运行此示例")
        return

    # -------------------------------------------------------------------------
    # 3. 创建 Agent
    # -------------------------------------------------------------------------
    agent = create_agent(
        model,
        tools=[send_email],
        system_prompt="你是一个邮件助手，帮用户发送邮件。",
        middleware=[hitl],
        checkpointer=InMemorySaver(),
    )

    # -------------------------------------------------------------------------
    # 4. 运行并处理中断
    # -------------------------------------------------------------------------
    config = {"configurable": {"thread_id": "email_approval_thread"}}

    user_input = input("\n💬 请输入你的请求（例如：帮我给 boss@company.com 发邮件，主题是工作汇报，内容为本周任务已完成）\n> ")
    if not user_input.strip():
        user_input = "帮我给 test@example.com 发邮件，主题是'测试邮件'，内容为'这是一封测试邮件'"

    print(f"\n🚀 Agent 开始运行...")

    # 第一次运行：Agent 会走到 send_email 工具的调用前，然后被中断
    result = agent.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
    )

    # 检查是否有待处理的中断
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        # 没有中断 — 可能 Agent 直接回答了，没有调用工具
        print(f"\n【Agent 回答】{result['messages'][-1].content}")
        return

    # 有中断 — 进入人在回路审批流程
    for interrupt in interrupts:
        decision = human_decision_loop(interrupt)

        print(f"\n📋 人类决策: {decision}")

        # 恢复 Agent 执行，将决策传回
        result = agent.invoke(
            Command(resume=decision),
            config=config,
        )

        # 打印最终结果
        print(f"\n{'=' * 50}")
        print(f"【Agent 最终回答】{result['messages'][-1].content}")
        print(f"{'=' * 50}")


if __name__ == '__main__':
    run_with_human_approval()
