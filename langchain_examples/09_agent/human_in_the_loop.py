# =============================================================================
# 人在回路（Human-in-the-Loop）— 关键操作需要人类审批
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解 HITL = AI 执行关键操作前需人类确认
#   ✅ 使用 HumanInTheLoopMiddleware 配置审批策略
#   ✅ 处理中断（interrupt）→ 人类决策 → 恢复执行
#
# 需要: ALIYUN_API_KEY 环境变量
# =============================================================================

from langchain_examples.utils import get_qwen_client
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


"""
什么是 Human-in-the-Loop？

  某些操作太重要了，不能让 AI 自己决定——比如发邮件、转账、删数据。
  这时候需要"人类审批"——AI 准备好参数，人类点头才执行。

  生活化比喻: HITL = 双人复核制
    财务大额转账 → 需要主管签字
    Agent 调用敏感工具 → 需要人类点头
"""


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送一封邮件给指定收件人。"""
    print(f"\n  📧 正在发送: {subject} → {to}")
    print(f"     正文: {body[:80]}...")
    return f"邮件已成功发送至 {to}"


def human_decision(interrupt) -> dict:
    """等待人类输入，直到获得有效决策（approve / edit / reject）。"""
    value = interrupt.value
    action = value["action_requests"][0]
    tool_name = action["name"]
    tool_args = action["args"]
    allowed = value["review_configs"][0]["allowed_decisions"]

    print(f"\n{'─' * 40}")
    print(f"⏸️  中断 — 工具调用待审批")
    print(f"  工具: {tool_name}")
    for k, v in tool_args.items():
        print(f"    {k}: {v}")

    options = []
    if "approve" in allowed: options.append("[a] 批准")
    if "edit" in allowed:    options.append("[e] 编辑后批准")
    if "reject" in allowed:  options.append("[r] 拒绝")
    print(f"  决策: {' | '.join(options)}")
    print(f"{'─' * 40}")

    while True:
        choice = input("\n👤 你的决定 > ").strip().lower()

        if choice in ("a", "approve") and "approve" in allowed:
            return {"decisions": [{"type": "approve"}]}

        elif choice in ("r", "reject") and "reject" in allowed:
            reason = input("👤 拒绝原因 > ").strip()
            return {"decisions": [{"type": "reject",
                    "message": reason or "人类操作员拒绝了此操作"}]}

        elif choice in ("e", "edit") and "edit" in allowed:
            print("  当前参数（留空 = 不修改）:")
            edited = {}
            for k, v in tool_args.items():
                new = input(f"    {k} [{v}] > ").strip()
                edited[k] = new if new else v
            return {"decisions": [{"type": "edit",
                    "edited_action": {"name": tool_name, "args": edited}}]}

        print(f"  ⚠️ 无效输入，请选择: {' / '.join([o.split(']')[0] + ']' for o in options])}")


def run():
    """完整的 HITL 流程：用户请求 → Agent 准备 → 中断 → 人类审批 → 恢复执行。"""
    print(f"\n-- 人在回路 — 邮件发送审批")

    hitl = HumanInTheLoopMiddleware(interrupt_on={
        "send_email": {"allowed_decisions": ["approve", "edit", "reject"]},
    })

    model = get_qwen_client()
    if model is None:
        return

    agent = create_agent(model=model, tools=[send_email],
                         system_prompt="你是一个邮件助手，帮用户发送邮件。",
                         middleware=[hitl], checkpointer=InMemorySaver())

    config = {"configurable": {"thread_id": "email_approval"}}
    user_input = input("\n💬 请输入邮件请求（如: 帮我给 boss@co.com 发邮件...）\n> ")
    if not user_input.strip():
        user_input = "帮我给 test@example.com 发邮件，主题=测试，内容=这是一封测试邮件"

    result = agent.invoke({"messages": [HumanMessage(content=user_input)]}, config)

    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        print(f"\nAgent 直接回答: {result['messages'][-1].content}")
        return

    for interrupt in interrupts:
        decision = human_decision(interrupt)
        result = agent.invoke(Command(resume=decision), config)
        print(f"\n✅ Agent: {result['messages'][-1].content}")


if __name__ == '__main__':
    print("\n>>> 09_agent/human_in_the_loop — 人在回路审批\n")
    print("⚠️ 本文件需要 ALIYUN_API_KEY 环境变量\n")
    run()
