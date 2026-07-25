"""
CrewAI 实战：写技术博客
安装依赖：pip install crewai

运行前准备：
1. 设置环境变量 DASHSCOPE_API_KEY（阿里云百炼 API Key）
2. 确保已安装依赖：pip install python-dotenv
"""

# ── 导入 CrewAI 核心组件 ─────────────────────────────────────
from crewai import Agent, Task, Crew, Process

# ── 配置 LLM：使用 Qwen（阿里云 DashScope） ──────────────────
from crewai.llm import LLM
import os
from dotenv import load_dotenv

load_dotenv()

llm = LLM(
    model="qwen-plus",
    api_key=os.getenv("ALIYUN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


# ── 创建 Agent 1：研究员 ─────────────────────────────────────
# role: 角色（这个 Agent 是干什么的）
# goal: 目标（这个 Agent 要达成什么）
# backstory: 背景（这个 Agent 的经历，影响它的行为方式）
researcher = Agent(
    role="AI 技术研究员",
    goal="收集和整理 AI 领域的最新进展和重要论文",
    backstory="""你是一位资深的 AI 技术研究员，有 10 年研究经验。
你擅长从海量信息中筛选出最有价值的内容，并用通俗易懂的语言解释复杂概念。
你特别关注大语言模型、Agent、RAG 等前沿技术。""",
    verbose=True,  # 开启详细输出，可以看到 Agent 的思考过程
    allow_delegation=False,  # 不允许委派任务给其他 Agent
    llm=llm,  # 使用 Qwen
)

# ── 创建 Agent 2：写手 ───────────────────────────────────────
writer = Agent(
    role="技术博客写手",
    goal="根据研究资料，撰写高质量的技术博客文章",
    backstory="""你是一位技术博客写手，擅长把复杂的技术概念用通俗的语言写出来。
你的文章风格：清晰、有趣、有深度，深受开发者喜爱。
你特别擅长写教程类文章，能让读者快速上手。""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

# ── 创建 Agent 3：校对 ───────────────────────────────────────
editor = Agent(
    role="内容编辑",
    goal="检查文章的逻辑、语法和可读性，确保文章质量",
    backstory="""你是一位资深内容编辑，有 15 年编辑经验。
你擅长发现文章中的逻辑漏洞、语法错误和表达不清的地方。
你对文章质量要求很高，但也懂得平衡专业性和可读性。""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

# ── 创建 Task 1：研究任务 ────────────────────────────────────
# description: 任务描述（要做什么）
# expected_output: 期望的输出（要输出什么）
# agent: 执行任务的 Agent
research_task = Task(
    description="""研究 AI 领域的最新进展，重点关注：
1. 大语言模型的最新发展（如 GPT-4、Claude、Qwen）
2. Agent 技术的最新突破
3. RAG（检索增强生成）的应用场景
请整理成结构化的研究报告。""",
    expected_output="""一份结构化的研究报告，包含：
- 3-5 个重要进展
- 每个进展的简要说明（100 字以内）
- 相关论文或项目的链接（如果有）""",
    agent=researcher,  # 由研究员执行
)

# ── 创建 Task 2：写作任务 ────────────────────────────────────
write_task = Task(
    description="""根据研究报告，撰写一篇技术博客文章。
要求：
1. 标题吸引人
2. 开头用一个有趣的故事或案例引入
3. 正文分 3-4 个小节，每节讲一个重点
4. 结尾总结要点，给出行动建议
5. 字数 1500-2000 字""",
    expected_output="""一篇完整的技术博客文章，包含：
- 标题
- 引言（故事/案例）
- 正文（3-4 个小节）
- 结论（总结 + 行动建议）""",
    agent=writer,  # 由写手执行
    context=[research_task],  # 依赖研究任务的输出
)

# ── 创建 Task 3：校对任务 ────────────────────────────────────
edit_task = Task(
    description="""检查文章的质量，重点关注：
1. 逻辑是否清晰
2. 语法是否正确
3. 表达是否通俗易懂
4. 是否有事实错误
请给出修改建议。""",
    expected_output="""一份校对报告，包含：
- 文章评分（1-10 分）
- 发现的问题列表
- 修改建议
- 最终版本（如果问题不大，直接修改）""",
    agent=editor,  # 由编辑执行
    context=[write_task],  # 依赖写作任务的输出
)

# ── 创建 Crew（团队） ────────────────────────────────────────
# agents: 团队成员列表
# tasks: 任务列表（按顺序执行）
# process: 执行流程（sequential = 顺序执行，hierarchical = 层级执行）
# verbose: 是否输出详细日志
crew = Crew(
    agents=[researcher, writer, editor],  # 团队成员
    tasks=[research_task, write_task, edit_task],  # 任务列表
    process=Process.sequential,  # 顺序执行：研究 → 写作 → 校对
    verbose=True,
)

# ── 启动团队 ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 开始创作技术博客...\n")

    # 启动团队，执行任务
    result = crew.kickoff()

    print("\n" + "=" * 50)
    print("✅ 创作完成！")
    print("=" * 50)
    print("\n最终结果：")
    print(result)