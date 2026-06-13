# =============================================================================
# 实战项目（二）— 研究助手：多步骤信息处理
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 构建多步骤信息处理流程（总结→提取→分析→报告）
#   ✅ 封装可复用的研究助手机器人类
# =============================================================================

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser


# =============================================================================
# 示例 1: 简单总结助手 — 长文本总结 + 要点提取
# =============================================================================

class SimpleSummarizer:
    """对长文本进行总结和要点提取。"""

    def __init__(self):
        self.model = ChatOllama(model="qwen3.5:2b")

        self.summary_prompt = PromptTemplate.from_template("""总结以下内容，用简洁语言概括主旨，并列出 3-5 个关键要点。

内容:
{content}

总结:""")
        self.summary_chain = self.summary_prompt | self.model | StrOutputParser()

    def summarize(self, content: str) -> str:
        return self.summary_chain.invoke({"content": content})


def demo_summarizer():
    """演示总结助手。"""
    print(f"\n-- 示例 1: 简单总结助手")

    summarizer = SimpleSummarizer()

    text = """人工智能（AI）是计算机科学的重要分支，研究如何让计算机具有学习、推理、感知等智能特征。
机器学习是 AI 的核心技术，让计算机从数据中学习规律。深度学习使用多层神经网络模拟人脑，
在图像识别、自然语言处理等领域取得突破性进展。大语言模型（LLM）如 GPT、Qwen 等，
基于海量文本训练，展现出强大的语言理解和生成能力。"""

    print(f"原文: {text[:60]}...\n")
    print(f"总结:\n{summarizer.summarize(text)}")


# =============================================================================
# 示例 2: 多步骤研究助手 — 分步研究 + 生成报告
# =============================================================================

class ResearchAssistant:
    """
    对研究主题进行多角度分析，生成结构化报告。

    流程: 定义关键问题 → 解释核心概念 → 优缺点分析 → 生成报告
    每一步的输出是下一步的输入 —— 典型的 Sequential Chain 实战应用。
    """

    def __init__(self):
        self.model = ChatOllama(model="qwen3.5:2b")
        self.parser = StrOutputParser()

        self.define_c = (
            PromptTemplate.from_template('分析研究主题"{topic}"，列出 5 个关键问题。\n关键问题:')
            | self.model | self.parser
        )
        self.explain_c = (
            PromptTemplate.from_template('用通俗语言解释"{concept}"，让初学者也能理解。\n解释:')
            | self.model | self.parser
        )
        self.analyze_c = (
            PromptTemplate.from_template("分析{topic}的优缺点。\n【优点】\n- ...\n\n【缺点】\n- ...\n\n分析:")
            | self.model | self.parser
        )
        self.report_c = (
            PromptTemplate.from_template("""根据以下研究内容生成简报。

主题: {topic}
关键问题: {questions}
概念解释: {explanations}
分析: {analysis}

格式: 【核心概念】... 【关键要点】... 【总结建议】...

简报:""")
            | self.model | self.parser
        )

    def research(self, topic: str) -> str:
        print(f"  步骤1: 定义关键问题...")
        questions = self.define_c.invoke({"topic": topic})
        print(f"  步骤2: 解释核心概念...")
        explanations = self.explain_c.invoke({"concept": topic})
        print(f"  步骤3: 优缺点分析...")
        analysis = self.analyze_c.invoke({"topic": topic})
        print(f"  步骤4: 生成报告...")
        return self.report_c.invoke({
            "topic": topic, "questions": questions,
            "explanations": explanations, "analysis": analysis,
        })


def demo_research():
    """演示多步骤研究助手。"""
    print(f"\n-- 示例 2: 多步骤研究助手")

    assistant = ResearchAssistant()
    report = assistant.research("机器学习入门")
    print(f"\n研究报告:\n{report}")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n>>> 08_project/research_assistant — 研究助手\n")

    demo_summarizer()
    demo_research()

    # 接下来学习: 09_agent/tools.py（Agent 工具定义）
