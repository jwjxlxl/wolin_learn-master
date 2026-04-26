# =============================================================================
# 研究助手
# =============================================================================
#  
# 用途：实战项目 - 信息搜索与总结助手
#
# 核心概念：
#   - 多步骤信息处理
#   - 信息聚合与总结
#   - 实用 AI 助手构建
# =============================================================================

# -----------------------------------------------------------------------------
# 运行前检查
# -----------------------------------------------------------------------------
# 1. 已安装 Ollama 服务
# 2. 已下载模型：ollama pull qwen3:4b
# 3. 已完成前面章节的学习
# -----------------------------------------------------------------------------

# 设置 UTF-8 编码（Windows 专用）
import sys
import io
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer,
    encoding='utf-8',
    errors='replace',
    line_buffering=True
)



# =============================================================================
# 第一部分：项目概述
# =============================================================================
"""
项目目标

🎯 构建一个研究助手
   - 帮助用户收集、整理、总结信息
   - 支持多步骤分析
   - 输出结构化报告

📋 功能特性
   - 信息搜索与筛选
   - 内容总结与提炼
   - 多角度分析
   - 结构化输出

📊 系统架构

   用户主题
       ↓
   分解子问题 → 并行搜索 → 信息聚合
       ↓
   分析总结 → 生成报告 → 输出
"""


# =============================================================================
# 示例 1: 简单信息总结助手
# =============================================================================

class SimpleSummarizer:
    """
    简单信息总结助手

    对输入的长文本进行总结
    """

    def __init__(self):
        """初始化总结助手"""
        from langchain_core.prompts import PromptTemplate
        from langchain_ollama import ChatOllama
        from langchain_core.output_parsers import StrOutputParser

        self.model = ChatOllama(model="qwen3:4b")

        # 总结提示模板
        self.summary_prompt = PromptTemplate.from_template("""
请总结以下内容，提取核心要点。

要求：
1. 用简洁的语言概括主旨
2. 列出 3-5 个关键要点
3. 保持原意，不添加个人观点

内容：
{content}

总结：""")

        self.summary_chain = self.summary_prompt | self.model | StrOutputParser()

        # 提取要点提示
        self.extract_prompt = PromptTemplate.from_template("""
请从以下内容中提取关键信息，按类别整理。

内容：
{content}

请按以下格式输出：
【核心概念】
- ...

【关键事实】
- ...

【重要数据】
- ...

【行动建议】
- ...

提取结果：""")

        self.extract_chain = self.extract_prompt | self.model | StrOutputParser()

    def summarize(self, content, max_length=200):
        """
        总结文本

        Args:
            content: 待总结的文本
            max_length: 最大长度

        Returns:
            总结结果
        """
        return self.summary_chain.invoke({"content": content})

    def extract_key_points(self, content):
        """
        提取关键要点

        Args:
            content: 待提取的文本

        Returns:
            结构化要点
        """
        return self.extract_chain.invoke({"content": content})


def demo_summarizer():
    """演示总结助手"""
    print("=" * 60)
    print("示例 1: 简单信息总结助手")
    print("=" * 60)

    summarizer = SimpleSummarizer()

    # 测试文本
    test_content = """
    人工智能（Artificial Intelligence，简称 AI）是计算机科学的一个重要分支，
    致力于研究如何让计算机具有人类的智能特征，如学习、推理、感知、理解语言等。

    机器学习是 AI 的核心技术之一，它让计算机通过数据学习规律，而不需要显式编程。
    深度学习是机器学习的一个子领域，使用神经网络模拟人脑的工作方式，
    在图像识别、自然语言处理等领域取得了突破性进展。

    大语言模型（Large Language Model，简称 LLM）是基于深度学习的技术，
    通过在大量文本上训练，获得强大的语言理解和生成能力。
    代表性的模型包括 GPT 系列、Claude、Qwen 等。

    AI 的应用非常广泛：
    - 医疗领域：辅助诊断、药物研发
    - 交通领域：自动驾驶、路径规划
    - 金融领域：风险评估、欺诈检测
    - 教育领域：个性化学习、智能辅导
    - 客服领域：智能问答、自动回复

    未来 AI 将继续发展，但也带来了一些挑战，如就业影响、隐私保护、算法偏见等。
    需要在技术发展和伦理治理之间找到平衡。
    """

    print("原文内容：\n")
    print(test_content)
    print("\n" + "-" * 40 + "\n")

    # 总结
    print("【总结】\n")
    summary = summarizer.summarize(test_content)
    print(summary)
    print()

    print("-" * 40 + "\n")

    # 提取要点
    print("【关键要点】\n")
    key_points = summarizer.extract_key_points(test_content)
    print(key_points)
    print()


# =============================================================================
# 示例 2: 多步骤研究助手
# =============================================================================

class ResearchAssistant:
    """
    多步骤研究助手

    对研究主题进行多角度分析
    """

    def __init__(self):
        """初始化研究助手"""
        from langchain_core.prompts import PromptTemplate
        from langchain_ollama import ChatOllama
        from langchain_core.output_parsers import StrOutputParser

        self.model = ChatOllama(model="qwen3:4b")

        # 步骤 1: 定义问题
        self.define_prompt = PromptTemplate.from_template("""
请分析研究主题"{topic}"，列出应该了解的 5 个关键问题。

格式：
1. ...
2. ...
3. ...
4. ...
5. ...

关键问题：""")

        self.define_chain = self.define_prompt | self.model | StrOutputParser()

        # 步骤 2: 解释概念
        self.explain_prompt = PromptTemplate.from_template("""
请解释以下概念，让初学者也能理解。

概念：{concept}

要求：
1. 用通俗的语言解释
2. 给一个生活中的例子
3. 说明应用场景

解释：""")

        self.explain_chain = self.explain_prompt | self.model | StrOutputParser()

        # 步骤 3: 优缺点分析
        self.analysis_prompt = PromptTemplate.from_template("""
请分析{topic}的优缺点。

【优点】
- ...

【缺点】
- ...

【适用场景】
- ...

【不适用场景】
- ...

分析结果：""")

        self.analysis_chain = self.analysis_prompt | self.model | StrOutputParser()

        # 步骤 4: 总结报告
        self.report_prompt = PromptTemplate.from_template("""
请根据以下研究内容，生成一份完整的研究简报。

研究主题：{topic}
关键问题：{questions}
概念解释：{explanations}
分析结果：{analysis}

格式：
# {topic}研究简报

## 核心概念
...

## 关键要点
...

## 优缺点分析
...

## 总结建议
...

简报：""")

        self.report_chain = self.report_prompt | self.model | StrOutputParser()

    def research(self, topic):
        """
        对主题进行完整研究

        Args:
            topic: 研究主题

        Returns:
            研究报告
        """
        print(f"\n开始研究：{topic}\n")
        print("=" * 50)

        # 步骤 1: 定义关键问题
        print("步骤 1: 定义关键问题...")
        questions = self.define_chain.invoke({"topic": topic})
        print(f"关键问题:\n{questions}\n")

        # 步骤 2: 解释核心概念
        print("步骤 2: 解释核心概念...")
        concept = topic.split()[0] if " " in topic else topic
        explanations = self.explain_chain.invoke({"concept": concept})
        print(f"概念解释:\n{explanations}\n")

        # 步骤 3: 优缺点分析
        print("步骤 3: 优缺点分析...")
        analysis = self.analysis_chain.invoke({"topic": topic})
        print(f"分析结果:\n{analysis}\n")

        # 步骤 4: 生成报告
        print("步骤 4: 生成研究报告...")
        report = self.report_chain.invoke({
            "topic": topic,
            "questions": questions,
            "explanations": explanations,
            "analysis": analysis
        })

        return report


def demo_research_assistant():
    """演示研究助手"""
    print("=" * 60)
    print("示例 2: 多步骤研究助手")
    print("=" * 60)

    assistant = ResearchAssistant()

    # 研究主题
    topics = [
        "机器学习入门",
        "大语言模型应用",
    ]

    for topic in topics:
        print(f"\n{'=' * 60}")
        print(f"研究主题：{topic}")
        print("=" * 60)

        report = assistant.research(topic)

        print("\n" + "=" * 50)
        print("研究报告")
        print("=" * 50)
        print(report)
        print()


# =============================================================================
# 示例 3: 学习规划助手
# =============================================================================

class LearningPlanAssistant:
    """
    学习规划助手

    帮助初学者制定学习计划
    """

    def __init__(self):
        """初始化学习规划助手"""
        from langchain_core.prompts import PromptTemplate
        from langchain_ollama import ChatOllama
        from langchain_core.output_parsers import StrOutputParser

        self.model = ChatOllama(model="qwen3:4b")

        # 学习路径生成
        self.path_prompt = PromptTemplate.from_template("""
我是一名{level}学习者，想学习{topic}。

请为我设计一个学习路径，包括：

【阶段 1: 基础入门】（预计时间：X 周）
- 学习目标
- 核心概念
- 推荐资源

【阶段 2: 进阶提升】（预计时间：X 周）
- 学习目标
- 核心技能
- 实践项目

【阶段 3: 熟练应用】（预计时间：X 周）
- 学习目标
- 高级主题
- 实战建议

我的背景：{background}

学习路径：""")

        self.path_chain = self.path_prompt | self.model | StrOutputParser()

        # 资源推荐
        self.resource_prompt = PromptTemplate.from_template("""
请为学习{topic}推荐资源。

要求：
1. 适合{level}水平
2. 包含多种类型（书籍、视频、实践）
3. 注明难度和预计学习时间

推荐格式：
【书籍】
- 《书名》：适合人群，特点

【在线课程】
- 课程名：平台，链接（如有）

【实践项目】
- 项目名称：难度，预计时间

【社区/论坛】
- 名称：特点

资源推荐：""")

        self.resource_chain = self.resource_prompt | self.model | StrOutputParser()

    def create_learning_plan(self, topic, level="零基础", background=""):
        """
        创建学习计划

        Args:
            topic: 学习主题
            level: 当前水平
            background: 个人背景

        Returns:
            学习计划
        """
        print(f"\n为'{topic}'创建学习计划...\n")

        # 生成学习路径
        print("生成学习路径...")
        path = self.path_chain.invoke({
            "topic": topic,
            "level": level,
            "background": background
        })
        print(f"\n{path}\n")

        # 推荐资源
        print("推荐学习资源...")
        resources = self.resource_chain.invoke({
            "topic": topic,
            "level": level
        })
        print(f"\n{resources}\n")

        return {"path": path, "resources": resources}


def demo_learning_plan():
    """演示学习规划助手"""
    print("=" * 60)
    print("示例 3: 学习规划助手")
    print("=" * 60)

    assistant = LearningPlanAssistant()

    # 创建学习计划
    plan = assistant.create_learning_plan(
        topic="Python 编程",
        level="零基础",
        background="完全没有编程经验，想转行做数据分析"
    )


# =============================================================================
# 示例 4: 综合研究报告生成器
# =============================================================================

class ReportGenerator:
    """
    综合研究报告生成器

    生成结构完整、格式规范的研究报告
    """

    def __init__(self):
        """初始化报告生成器"""
        from langchain_core.prompts import PromptTemplate
        from langchain_ollama import ChatOllama
        from langchain_core.output_parsers import StrOutputParser

        self.model = ChatOllama(model="qwen3:4b")

        # 大纲生成
        self.outline_prompt = PromptTemplate.from_template("""
请为"{topic}"主题的研究报告生成大纲。

要求：
1. 结构清晰，层次分明
2. 包含 4-6 个主要章节
3. 每章有 2-4 个小节

大纲：""")

        self.outline_chain = self.outline_prompt | self.model | StrOutputParser()

        # 章节撰写
        self.section_prompt = PromptTemplate.from_template("""
请撰写研究报告的以下章节：

主题：{topic}
章节：{section}
要点：{points}

要求：
1. 内容详实，有深度
2. 语言专业但不晦涩
3. 适当举例说明

章节内容：""")

        self.section_chain = self.section_prompt | self.model | StrOutputParser()

        # 摘要生成
        self.abstract_prompt = PromptTemplate.from_template("""
请为以下研究报告写一份摘要。

报告内容：
{content}

要求：
1. 200 字以内
2. 概括核心内容
3. 突出主要发现

摘要：""")

        self.abstract_chain = self.abstract_prompt | self.model | StrOutputParser()

    def generate_outline(self, topic):
        """生成报告大纲"""
        return self.outline_chain.invoke({"topic": topic})

    def write_section(self, topic, section, points):
        """撰写章节"""
        return self.section_chain.invoke({
            "topic": topic,
            "section": section,
            "points": points
        })

    def write_abstract(self, content):
        """写摘要"""
        return self.abstract_chain.invoke({"content": content})

    def generate_full_report(self, topic):
        """
        生成完整报告

        Args:
            topic: 报告主题

        Returns:
            完整报告字符串
        """
        print(f"\n开始生成报告：{topic}\n")
        print("=" * 50)

        # 步骤 1: 生成大纲
        print("步骤 1: 生成大纲...")
        outline = self.generate_outline(topic)
        print(f"\n{outline}\n")
        print("-" * 40 + "\n")

        # 步骤 2: 撰写主要章节（简化版，只写一个示例章节）
        print("步骤 2: 撰写章节（示例）...")
        section_content = self.write_section(
            topic=topic,
            section="核心概念",
            points="定义、特点、分类"
        )
        print(f"\n{section_content}\n")
        print("-" * 40 + "\n")

        # 步骤 3: 写摘要
        print("步骤 3: 生成摘要...")
        abstract = self.write_abstract(f"{topic}研究报告\n\n{section_content}")
        print(f"\n【摘要】{abstract}\n")

        return {
            "topic": topic,
            "outline": outline,
            "sample_section": section_content,
            "abstract": abstract
        }


def demo_report_generator():
    """演示报告生成器"""
    print("=" * 60)
    print("示例 4: 综合研究报告生成器")
    print("=" * 60)

    generator = ReportGenerator()

    # 生成报告
    report = generator.generate_full_report("人工智能在医疗领域的应用")

    print("\n" + "=" * 50)
    print("报告生成完成！")
    print("=" * 50)


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  研究助手 - Research Assistant")
    print("  说明：信息搜索与总结助手")
    print("=" * 70 + "\n")

    print("【运行前检查】")
    print("  1. 已安装 Ollama 服务")
    print("  2. 已下载模型：ollama pull qwen3:4b")
    print("  3. 已完成前面章节的学习")
    print()

    # 运行示例
    demo_summarizer()
    # demo_research_assistant()  # 耗时较长，按需运行
    # demo_learning_plan()       # 耗时较长，按需运行
    # demo_report_generator()    # 耗时较长，按需运行

    print("=" * 70)
    print("  恭喜完成 LangChain 基础教程！")
    print("  建议复习重点概念，多动手实践")
    print("=" * 70 + "\n")
