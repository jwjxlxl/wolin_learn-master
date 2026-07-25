"""
CrewAI 层级执行：经理 Agent 协调团队
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

# ── 创建经理 Agent ───────────────────────────────────────────
manager = Agent(
    role="项目经理",
    goal="协调团队，确保任务高质量完成",
    backstory="""你是一位经验丰富的项目经理，擅长协调团队。
你能根据团队成员的能力，合理分配任务。
你会监控任务进度，及时处理问题。""",
    verbose=True,
    allow_delegation=True,  # 允许委派任务
    llm=llm,
)

# ── 创建团队成员 ─────────────────────────────────────────────
researcher = Agent(
    role="研究员",
    goal="收集资料",
    backstory="你是研究员，擅长收集资料",
    verbose=True,
    llm=llm,
)

writer = Agent(
    role="写手",
    goal="写文章",
    backstory="你是写手，擅长写文章",
    verbose=True,
    llm=llm,
)

# ── 创建任务 ─────────────────────────────────────────────────
research_task = Task(
    description="研究 AI 最新进展",
    expected_output="研究报告",
    agent=researcher,
)

write_task = Task(
    description="写技术博客",
    expected_output="博客文章",
    agent=writer,
)

# ── 创建团队（层级执行） ─────────────────────────────────────
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.hierarchical,  # 层级执行
    manager_agent=manager,  # 指定经理 Agent
    verbose=True,
)

if __name__ == "__main__":
    print("🚀 开始层级执行...\n")
    result = crew.kickoff()
    print("\n最终结果：")
    print(result)