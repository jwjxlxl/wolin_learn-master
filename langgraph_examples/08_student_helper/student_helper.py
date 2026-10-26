# =============================================================================
# 08_student_helper — 学生助手「智教小 E」（LangGraph 长时运行 + 人工介入）
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 用 LangGraph 构建一个多节点、带循环的学生辅导 Agent
#   ✅ 用 interrupt() + Checkpointer + thread_id 实现「暂停 → 恢复」的长时运行
#   ✅ 理解 HITL（Human-in-the-loop）：图在学生作答处暂停，回答后从同一处继续
#
# 需求（见同目录 require.md）：
#   1. 出题       ：根据知识点从题库抽题
#   2. 作答       ：学生回答
#   3. 判题       ：判断对错并给出解析
#   4. 情感识别   ：检测学生情绪，低落时给予鼓励
#   5. 知识讲解   ：答错后讲解知识点
#
# 运行前检查：
# 1. 已安装依赖：pip install langgraph langchain-core
# 2. 本示例为确定性演示（无需 API Key），直接运行：
#      python student_helper.py
# =============================================================================

import sys
import os
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 确保项目根目录可导入（用于 utils/ 共享模块）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.messages import HumanMessage, AIMessage


# =============================================================================
# 题库：根据知识点组织，出题节点按顺序从中抽题
# =============================================================================

QUESTION_BANK = [
    {
        "knowledge_point": "一元二次方程——因式分解",
        "question": "方程 x² − 5x + 6 = 0 的两个根是（　）",
        "options": "A. 2 和 3\nB. −2 和 −3\nC. 1 和 6\nD. −1 和 −6",
        "answer": "A",
        "explanation": "因式分解得 (x−2)(x−3)=0，所以 x=2 或 x=3。",
    },
    {
        "knowledge_point": "一元二次方程——开平方法",
        "question": "方程 x² − 4 = 0 的解是（　）",
        "options": "A. x=2\nB. x=−2\nC. x=±2\nD. x=4",
        "answer": "C",
        "explanation": "移项得 x²=4，两边开平方得 x=±2。",
    },
    {
        "knowledge_point": "一元二次方程——判别式",
        "question": "方程 x² + 2x + 3 = 0 的根的情况是（　）",
        "options": "A. 有两个不等实根\nB. 有两个相等实根\nC. 无实根\nD. 无法判断",
        "answer": "C",
        "explanation": "判别式 Δ = 2² − 4×1×3 = −8 < 0，所以方程无实数根。",
    },
]

MAX_QUESTIONS = len(QUESTION_BANK)  # 最多出 3 道题


# =============================================================================
# State 定义：贯穿整个辅导流程的共享状态
# =============================================================================

class TutorState(TypedDict):
    messages: Annotated[list, add_messages]  # 对话历史（add_messages 自动追加）
    current_question: str    # 当前题目（题干 + 选项）
    correct_answer: str      # 当前题正确答案
    knowledge_point: str     # 当前题知识点
    student_answer: str      # 学生作答（保留上一轮，供下一轮情感识别使用）
    is_correct: bool         # 判题结果
    explanation: str         # 判题解析
    emotion: str             # 情绪：positive / neutral / negative
    encourage_message: str   # 鼓励语（情感识别节点生成）
    question_index: int      # 已出题数（0 起，用于控制最多 3 题）


# =============================================================================
# Node 定义
# =============================================================================

def draw_question_node(state: TutorState) -> dict:
    """节点 1：出题 —— 从题库按顺序抽题。"""
    idx = state.get("question_index", 0)
    item = QUESTION_BANK[idx]  # should_continue 保证 idx < MAX_QUESTIONS
    question = f"{item['question']}\n\n{item['options']}"
    print(f"  [出题] 第 {idx + 1} 题（{item['knowledge_point']}）")
    return {
        "current_question": question,
        "correct_answer": item["answer"],
        "knowledge_point": item["knowledge_point"],
        "question_index": idx + 1,
        "messages": [AIMessage(content=f"📝 第 {idx + 1} 题（{item['knowledge_point']}）：\n{question}")],
    }


def emotion_detect_node(state: TutorState) -> dict:
    """节点 2：情感识别 —— 根据上一轮表现判断情绪，低落时生成鼓励语。

    注意：本节点位于「出题」之后、「作答」之前，因此识别的不是本轮作答，
    而是上一轮作答的结果（首轮无历史，默认 neutral）。
    """
    if not state.get("student_answer"):
        # 首轮：还没有作答历史
        emotion, encourage = "neutral", "请仔细读题，选出正确答案。"
    elif state.get("is_correct"):
        emotion, encourage = "positive", "上一题答得不错，继续保持！"
    else:
        emotion, encourage = "negative", "别灰心，答错很正常，我们再接再厉！"

    print(f"  [情感识别] 情绪：{emotion}")
    return {"emotion": emotion, "encourage_message": encourage}


def student_answer_node(state: TutorState) -> dict:
    """节点 3：作答 —— 暂停等待学生输入（HITL 的核心）。

    interrupt() 会在此处暂停整个图，把 payload 返回给调用方（前端/API）。
    学生作答后，调用 graph.invoke(Command(resume={"answer": "B"}), config) 恢复，
    interrupt() 就会返回 {"answer": "B"}。
    """
    response = interrupt({
        "type": "student_answer",
        "question": state["current_question"],
        "encourage": state.get("encourage_message", "请回答："),
    })
    # 恢复后 response 即外部传入的 resume 值
    answer = response.get("answer", "") if isinstance(response, dict) else ""
    print(f"  [作答] 学生答案：{answer}")
    return {
        "student_answer": answer,
        "messages": [HumanMessage(content=answer)],
    }


def grade_answer_node(state: TutorState) -> dict:
    """节点 4：判题 —— 与题库正确答案比对，给出解析。"""
    student = state.get("student_answer", "").strip().upper()
    correct = state.get("correct_answer", "").strip().upper()
    is_correct = student == correct

    idx = state.get("question_index", 1) - 1  # 当前题在题库中的索引
    item = QUESTION_BANK[idx]
    if is_correct:
        explanation = f"✅ 回答正确！{item['explanation']}"
    else:
        explanation = f"❌ 回答错误。正确答案是 {correct}。\n解析：{item['explanation']}"

    print(f"  [判题] {'正确' if is_correct else '错误'}")
    return {
        "is_correct": is_correct,
        "explanation": explanation,
        "messages": [AIMessage(content=explanation)],
    }


def should_continue(state: TutorState) -> Literal["explain_knowledge", "finish"]:
    """条件路由：出满 MAX_QUESTIONS 题就结束，否则讲解知识点并进入下一题。"""
    if state.get("question_index", 0) >= MAX_QUESTIONS:
        return "finish"
    return "explain_knowledge"


def explain_knowledge_node(state: TutorState) -> dict:
    """节点 5：知识讲解 —— 答错时详细讲解知识点。"""
    if state.get("is_correct"):
        return {"messages": [AIMessage(content="✅ 回答正确，继续加油！")]}

    idx = state.get("question_index", 1) - 1
    item = QUESTION_BANK[idx]
    content = (
        f"💡 知识点讲解（{item['knowledge_point']}）：\n"
        f"{item['explanation']}\n"
        f"建议：回顾「{item['knowledge_point']}」的基本概念，再做几道同类题巩固。"
    )
    print("  [知识讲解] 答错，展开讲解")
    return {"messages": [AIMessage(content=content)]}


# =============================================================================
# 构建图
# =============================================================================

builder = StateGraph(TutorState)

builder.add_node("draw_question", draw_question_node)
builder.add_node("emotion_detect", emotion_detect_node)
builder.add_node("student_answer", student_answer_node)
builder.add_node("grade_answer", grade_answer_node)
builder.add_node("explain_knowledge", explain_knowledge_node)

# 循环体：出题 → 情感识别 → 作答 → 判题
builder.add_edge(START, "draw_question")
builder.add_edge("draw_question", "emotion_detect")
builder.add_edge("emotion_detect", "student_answer")
builder.add_edge("student_answer", "grade_answer")

# 判题后：出满 3 题结束，否则讲解知识点后回到出题
builder.add_conditional_edges(
    "grade_answer",
    should_continue,
    {
        "explain_knowledge": "explain_knowledge",
        "finish": END,
    },
)
builder.add_edge("explain_knowledge", "draw_question")

# 编译：必须挂 Checkpointer，interrupt() 才能「暂停 → 恢复」跨 invoke 保留状态
graph = builder.compile(checkpointer=MemorySaver())


# =============================================================================
# 运行演示：用预置答案模拟学生作答，展示 interrupt 的「暂停 → 恢复」
# =============================================================================

def run_demo():
    """模拟一轮完整辅导：出题 → 情感识别 → 作答(暂停) → 判题 → 讲解 → 下一题。"""
    # thread_id 标识一次会话，该会话的状态都保存在检查点里
    config = {"configurable": {"thread_id": "tutor-session-001"}}

    # 模拟学生依次作答（实际项目中来自前端 / API）
    simulated_answers = ["B", "C", "A"]

    # 首次启动：跑到 student_answer 的 interrupt() 处暂停
    result = graph.invoke({"messages": [], "question_index": 0}, config=config)

    for i, answer in enumerate(simulated_answers):
        snapshot = graph.get_state(config)
        if snapshot.next == ():  # 已结束
            break

        # —— 暂停在 student_answer：展示题目与鼓励语 ——
        print("\n" + "─" * 50)
        print(f"📝 题目：\n{result.get('current_question', '')}")
        print(f"💬 {result.get('encourage_message', '')}")

        # —— 学生作答（演示用预置答案；真实项目替换为 input() / API 调用）——
        print(f"🧑 学生答案：{answer}")
        result = graph.invoke(Command(resume={"answer": answer}), config=config)

        # —— 展示上一题的判题与解析 ——
        print(f"\n{result.get('explanation', '')}")

    # 完整对话回放：体现 messages 自动累计 + 答错时的知识讲解
    print("\n" + "=" * 50)
    print("📜 本轮完整记录：")
    for msg in result.get("messages", []):
        if hasattr(msg, "content") and msg.content:
            icon = "🤖" if getattr(msg, "type", "") == "ai" else "🧑"
            print(f"{icon} {msg.content}")
    print("=" * 50)
    print("🎉 今日练习结束！")


if __name__ == "__main__":
    print("=" * 60)
    print("  学生助手「智教小 E」—— LangGraph 长时运行（HITL）演示")
    print("  流程：出题 → 情感识别 → 作答(暂停) → 判题 → 讲解 → 下一题")
    print("=" * 60)

    run_demo()

    print("\n💡 思考：")
    print("  1. 换一个 thread_id 重跑，就是另一个「学生」的独立会话。")
    print("  2. 去掉 checkpointer，interrupt() 将无法「暂停 → 恢复」——体会检查点的作用。")
    print("  3. 用 interrupt_before=[\"student_answer\"] 也能实现 HITL（在进入节点前暂停，")
    print("     答案通过 invoke({\"student_answer\": ...}) 注入状态），本文件用的是节点内 interrupt()。")
