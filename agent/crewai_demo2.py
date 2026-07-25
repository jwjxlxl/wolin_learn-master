"""
CrewAI 实战：旅行规划团队
安装依赖：pip install crewai
"""

from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM
import os
from dotenv import load_dotenv

load_dotenv()

llm = LLM(
    model="qwen-plus",
    api_key=os.getenv("ALIYUN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ── 创建 Agent 1：旅行顾问 ───────────────────────────────────
travel_advisor = Agent(
    role="资深旅行顾问",
    goal="根据用户需求，推荐最佳旅行目的地和时间",
    backstory="""你是一位资深旅行顾问，去过 50 多个国家。
你擅长根据用户的预算、兴趣和时间，推荐最合适的旅行方案。
你特别了解亚洲和欧洲的旅游路线。""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

# ── 创建 Agent 2：预算规划师 ─────────────────────────────────
budget_planner = Agent(
    role="旅行预算规划师",
    goal="制定详细的旅行预算，包括机票、酒店、餐饮、景点门票",
    backstory="""你是一位专业的旅行预算规划师，擅长控制旅行成本。
你能在保证旅行质量的前提下，帮用户省钱。
你特别擅长找性价比高的酒店和机票。""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

# ── 创建 Agent 3：行程规划师 ─────────────────────────────────
itinerary_planner = Agent(
    role="行程规划师",
    goal="制定详细的每日行程安排，包括景点、餐厅、交通",
    backstory="""你是一位经验丰富的行程规划师，擅长安排紧凑但不累的行程。
你特别了解各个景点的开放时间和最佳游览顺序。
你能根据用户的体力和兴趣，安排最合理的行程。""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

# ── 创建 Task 1：推荐目的地 ──────────────────────────────────
recommend_task = Task(
    description="""用户需求：
- 目的地：日本
- 时间：5 天
- 预算：10000 元
- 兴趣：美食、文化、购物

请推荐最佳的旅行路线和必去的景点。""",
    expected_output="""一份旅行推荐报告，包含：
- 推荐的城市和景点（3-5 个）
- 每个景点的亮点（50 字以内）
- 建议的游览顺序""",
    agent=travel_advisor,
)

# ── 创建 Task 2：制定预算 ────────────────────────────────────
budget_task = Task(
    description="""根据旅行推荐，制定详细的预算方案。
包括：
1. 机票（往返）
2. 酒店（4 晚）
3. 餐饮（5 天）
4. 景点门票
5. 交通（市内）
6. 购物预算

总预算控制在 10000 元以内。""",
    expected_output="""一份详细的预算表，包含：
- 各项费用的明细
- 总计金额
- 省钱建议""",
    agent=budget_planner,
    context=[recommend_task],  # 依赖推荐任务的输出
)

# ── 创建 Task 3：安排行程 ────────────────────────────────────
itinerary_task = Task(
    description="""根据推荐和预算，制定详细的 5 天行程。
包括：
1. 每天的景点安排
2. 餐厅推荐（午餐、晚餐）
3. 交通方式
4. 注意事项

要求：行程紧凑但不累，每天步行不超过 15000 步。""",
    expected_output="""一份详细的行程安排表，包含：
- 5 天的详细行程（每天 3-4 个活动）
- 每个活动的时间、地点、交通方式
- 餐厅推荐（人均消费）
- 注意事项""",
    agent=itinerary_planner,
    context=[recommend_task, budget_task],  # 依赖前两个任务
)

# ── 创建团队并启动 ───────────────────────────────────────────
crew = Crew(
    agents=[travel_advisor, budget_planner, itinerary_planner],
    tasks=[recommend_task, budget_task, itinerary_task],
    process=Process.sequential,
    verbose=True,
)

if __name__ == "__main__":
    print("✈️ 开始规划日本之旅...\n")
    result = crew.kickoff()

    print("\n" + "=" * 50)
    print("✅ 旅行规划完成！")
    print("=" * 50)
    print("\n最终行程：")
    print(result)